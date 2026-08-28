"""JSONL export and import.

Implemented by work unit U8. Signatures are frozen; see ``_store.SQLAlchemyChatStore``.

Line format: ``{"conversation_key": str, "seq": int, "run_id": str | null, "message": {...}}``
with ``message`` being the stored blob. Import validates every line's message through
``ModelMessagesTypeAdapter`` before inserting and allocates fresh sequence numbers per
conversation; both directions stream (no full-table loads into memory).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import IO, TYPE_CHECKING, cast

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from ._models import BaseConversation
from ._serialize import dump_message, extract_denorm, load_messages, message_content_hash, sanitize_payload

if TYPE_CHECKING:
    from ._store import SQLAlchemyChatStore

__all__ = ['export_jsonl', 'import_jsonl']

#: Rows fetched per round trip while streaming an export.
_EXPORT_CHUNK_SIZE = 500


async def export_jsonl(
    store: SQLAlchemyChatStore,
    destination: IO[bytes] | os.PathLike[str] | str,
    *,
    conversation_keys: Sequence[str] | None = None,
    since: datetime | None = None,
    session: AsyncSession | None = None,
) -> int:
    """Write matching messages as JSONL; returns the number of lines written."""
    async with store.sessions.scope(session) as (db, _owned):
        if isinstance(destination, (str, os.PathLike)):
            # Local file IO is deliberately synchronous: exports write small buffered chunks.
            with open(destination, 'wb') as sink:  # noqa: ASYNC230
                return await _export_lines(store, db, sink, conversation_keys=conversation_keys, since=since)
        return await _export_lines(store, db, destination, conversation_keys=conversation_keys, since=since)


async def _export_lines(
    store: SQLAlchemyChatStore,
    db: AsyncSession,
    sink: IO[bytes],
    *,
    conversation_keys: Sequence[str] | None,
    since: datetime | None,
) -> int:
    conversation_cls = store.models.conversation
    message_cls = store.models.message
    statement = (
        sa.select(conversation_cls.conversation_key, message_cls.seq, message_cls.run_id, message_cls.message)
        .join_from(message_cls, conversation_cls, message_cls.conversation_pk == conversation_cls.id)
        .order_by(conversation_cls.conversation_key, message_cls.seq)
    )
    if conversation_keys is not None:
        statement = statement.where(conversation_cls.conversation_key.in_(list(conversation_keys)))
    if since is not None:
        statement = statement.where(message_cls.created_at >= since)

    count = 0
    result = await db.stream(statement.execution_options(yield_per=_EXPORT_CHUNK_SIZE))
    async for conversation_key, seq, run_id, payload in result.tuples():
        record = {'conversation_key': conversation_key, 'seq': seq, 'run_id': run_id, 'message': payload}
        sink.write(json.dumps(record, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))
        sink.write(b'\n')
        count += 1
    return count


async def import_jsonl(
    store: SQLAlchemyChatStore,
    source: IO[bytes] | os.PathLike[str] | str,
    *,
    session: AsyncSession | None = None,
) -> int:
    """Read a JSONL export and persist it; returns the number of messages imported."""
    async with store.sessions.scope(session) as (db, _owned):
        if isinstance(source, (str, os.PathLike)):
            # Local file IO is deliberately synchronous: imports read small buffered lines.
            with open(source, 'rb') as lines:  # noqa: ASYNC230
                return await _import_lines(store, db, lines)
        return await _import_lines(store, db, source)


@dataclass
class _ConversationState:
    """Per-conversation import bookkeeping; the max seq is fetched once and advanced locally."""

    conversation: BaseConversation
    next_seq: int
    imported: int = 0
    latest_timestamp: datetime | None = None


async def _import_lines(store: SQLAlchemyChatStore, db: AsyncSession, lines: IO[bytes]) -> int:
    dialect_name = db.get_bind().dialect.name
    message_factory = cast('Callable[..., object]', store.models.message)
    states: dict[str, _ConversationState] = {}
    imported = 0

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        record_raw: object = json.loads(line)
        if not isinstance(record_raw, dict):
            raise _line_error('expected an object per line')
        record = cast('dict[str, object]', record_raw)
        conversation_key = record.get('conversation_key')
        if not isinstance(conversation_key, str):
            raise _line_error('"conversation_key" must be a string')
        run_id_raw = record.get('run_id')
        run_id = run_id_raw if isinstance(run_id_raw, str) else None
        payload = cast('dict[str, object]', record.get('message'))

        # Validation raises pydantic's ValidationError on corrupt payloads; the validated
        # message is re-dumped so imported blobs are normalized to current adapter output.
        [message] = load_messages([payload])
        stored = sanitize_payload(dump_message(message), dialect_name=dialect_name, custom=store.json_sanitizer)
        denorm = extract_denorm(message)

        state = states.get(conversation_key)
        if state is None:
            state = await _conversation_state(store, db, conversation_key)
            states[conversation_key] = state

        db.add(
            message_factory(
                conversation_pk=state.conversation.id,
                seq=state.next_seq,
                kind=denorm.kind,
                message=dict(stored),
                content_hash=message_content_hash(message),
                run_id=run_id,
                message_timestamp=denorm.message_timestamp,
                model_name=denorm.model_name,
                provider_name=denorm.provider_name,
                provider_response_id=denorm.provider_response_id,
                finish_reason=denorm.finish_reason,
                state=denorm.state,
                input_tokens=denorm.input_tokens,
                output_tokens=denorm.output_tokens,
                cache_read_tokens=denorm.cache_read_tokens,
                cache_write_tokens=denorm.cache_write_tokens,
                cost=denorm.cost,
                has_user_prompt=denorm.has_user_prompt,
                tool_call_count=denorm.tool_call_count,
            )
        )
        state.next_seq += 1
        state.imported += 1
        imported += 1
        if denorm.message_timestamp is not None and (
            state.latest_timestamp is None or _as_utc(denorm.message_timestamp) > _as_utc(state.latest_timestamp)
        ):
            state.latest_timestamp = denorm.message_timestamp

    for state in states.values():
        conversation = state.conversation
        conversation.message_count = conversation.message_count + state.imported
        current = conversation.last_activity_at
        if state.latest_timestamp is not None and (
            current is None or _as_utc(state.latest_timestamp) > _as_utc(current)
        ):
            conversation.last_activity_at = state.latest_timestamp
    await db.flush()
    return imported


async def _conversation_state(
    store: SQLAlchemyChatStore, db: AsyncSession, conversation_key: str
) -> _ConversationState:
    conversation_cls = store.models.conversation
    existing = (
        await db.execute(sa.select(conversation_cls).where(conversation_cls.conversation_key == conversation_key))
    ).scalar_one_or_none()
    if existing is None:
        conversation_factory = cast('Callable[..., BaseConversation]', store.models.conversation)
        created = conversation_factory(conversation_key=conversation_key)
        db.add(created)
        await db.flush()
        return _ConversationState(conversation=created, next_seq=1)

    message_cls = store.models.message
    max_seq = await db.scalar(sa.select(sa.func.max(message_cls.seq)).where(message_cls.conversation_pk == existing.id))
    return _ConversationState(conversation=existing, next_seq=(max_seq or 0) + 1)


def _line_error(detail: str) -> ValueError:
    """Corrupt-line failures are data errors, so they surface as ``ValueError``."""
    return ValueError(f'invalid JSONL line: {detail}')


def _as_utc(value: datetime) -> datetime:
    """Treat naive datetimes as UTC; SQLite returns naive values even for timezone columns."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
