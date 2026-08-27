"""Read path: history loading with turn windowing, conversation listing, transcripts.

Implemented by work unit U3. Signatures are frozen; see ``_store.SQLAlchemyChatStore``.

Implementation notes:
- ``load_history`` orders by ``seq``; ``max_turns`` selects the last N turns via a backward
  chunked scan over the message payloads (newest first), counting turn boundaries with the
  same rule as ``_turns.slice_to_recent_turns``: a boundary is a request that carries a
  user prompt and no tool-return/retry-prompt parts, so tool-call/tool-return pairs are
  never split (a deferred/HITL resume request continues its turn instead of starting one).
- When ``sanitize`` is true (or the store was built with ``trusted_history=False`` and
  ``sanitize`` is None), pass the loaded history through
  ``pydantic_ai.messages.sanitize_messages`` before returning.
- Unknown conversation keys diverge intentionally: ``load_history`` returns ``[]`` (loading
  an empty history must be cheap and non-fatal), while ``get_transcript`` raises
  ``ConversationNotFoundError`` (transcripts are explicit lookups).
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast, overload

import sqlalchemy as sa
from pydantic_ai.messages import ModelMessage, sanitize_messages
from sqlalchemy.ext.asyncio import AsyncSession

from ._exceptions import ConversationNotFoundError
from ._serialize import load_messages
from ._turns import messages_to_text

if TYPE_CHECKING:
    from ._models import BaseMessage
    from ._store import SQLAlchemyChatStore

from ._types import ConversationRecord

__all__ = ['get_transcript', 'list_conversations', 'load_history']

#: Page size of the backward scan that resolves a ``max_turns`` window.
_WINDOW_CHUNK_SIZE = 200

_NON_BOUNDARY_PART_KINDS = ('tool-return', 'retry-prompt')


async def _resolve_conversation_pk(
    store: SQLAlchemyChatStore, db: AsyncSession, conversation_key: str
) -> uuid.UUID | None:
    """Resolve a conversation key to its primary key, or ``None`` when unknown.

    ``conversation_key`` is globally unique on the shipped models; hosts that relax that
    constraint (e.g. per-tenant uniqueness) must scope their sessions accordingly, since
    the store resolves the first match.
    """
    conversation = store.models.conversation
    query = sa.select(conversation.id).where(conversation.conversation_key == conversation_key).limit(1)
    return await db.scalar(query)


def _payload_is_turn_boundary(payload: Mapping[str, object]) -> bool:
    """Apply the ``_turns`` boundary rule to a raw (not yet validated) message payload."""
    if payload.get('kind') != 'request':
        return False
    parts = payload.get('parts')
    if not isinstance(parts, list):
        return False
    has_user_prompt = False
    for part in cast('list[object]', parts):
        if not isinstance(part, dict):
            continue
        part_kind = cast('dict[str, object]', part).get('part_kind')
        if part_kind == 'user-prompt':
            has_user_prompt = True
        elif part_kind in _NON_BOUNDARY_PART_KINDS:
            return False
    return has_user_prompt


async def _fetch_payloads(
    db: AsyncSession, message: type[BaseMessage], conversation_pk: uuid.UUID, max_turns: int | None
) -> list[Mapping[str, object]]:
    """Fetch message payloads in ``seq`` order, windowed to the last ``max_turns`` turns."""
    if max_turns is None:
        query = sa.select(message.message).where(message.conversation_pk == conversation_pk).order_by(message.seq.asc())
        return list((await db.scalars(query)).all())

    collected: list[Mapping[str, object]] = []
    boundaries = 0
    offset = 0
    while True:
        chunk_query = (
            sa.select(message.message)
            .where(message.conversation_pk == conversation_pk)
            .order_by(message.seq.desc())
            .limit(_WINDOW_CHUNK_SIZE)
            .offset(offset)
        )
        chunk = (await db.scalars(chunk_query)).all()
        for payload in chunk:
            collected.append(payload)
            if _payload_is_turn_boundary(payload):
                boundaries += 1
                if boundaries >= max_turns:
                    collected.reverse()
                    return collected
        if len(chunk) < _WINDOW_CHUNK_SIZE:
            # Fewer boundaries than requested: the whole history is within the window.
            collected.reverse()
            return collected
        offset += len(chunk)


async def _load_for_conversation(
    store: SQLAlchemyChatStore,
    db: AsyncSession,
    conversation_pk: uuid.UUID,
    *,
    max_turns: int | None,
    sanitize: bool | None,
) -> list[ModelMessage]:
    payloads = await _fetch_payloads(db, store.models.message, conversation_pk, max_turns)
    loaded = load_messages(payloads)
    effective_sanitize = sanitize if sanitize is not None else not store.trusted_history
    if effective_sanitize:
        return sanitize_messages(loaded)
    return loaded


async def load_history(
    store: SQLAlchemyChatStore,
    *,
    conversation_key: str,
    max_turns: int | None = None,
    sanitize: bool | None = None,
    session: AsyncSession | None = None,
) -> list[ModelMessage]:
    """Load a conversation's history in order, optionally windowed to the last ``max_turns`` turns.

    Note that ``sanitize_messages`` also strips dangling trailing tool calls; a HITL resume
    that needs those pending calls should load with ``sanitize=False`` or a trusted store.
    """
    if max_turns is not None and max_turns <= 0:
        return []
    async with store.sessions.scope(session) as (db, _owned):
        conversation_pk = await _resolve_conversation_pk(store, db, conversation_key)
        if conversation_pk is None:
            return []
        return await _load_for_conversation(store, db, conversation_pk, max_turns=max_turns, sanitize=sanitize)


async def get_transcript(
    store: SQLAlchemyChatStore,
    *,
    conversation_key: str,
    session: AsyncSession | None = None,
) -> str:
    async with store.sessions.scope(session) as (db, _owned):
        conversation_pk = await _resolve_conversation_pk(store, db, conversation_key)
        if conversation_pk is None:
            raise ConversationNotFoundError(f'unknown conversation key: {conversation_key!r}')
        history = await _load_for_conversation(store, db, conversation_pk, max_turns=None, sanitize=None)
    return messages_to_text(history)


@overload
def _aware_utc(value: datetime) -> datetime: ...
@overload
def _aware_utc(value: None) -> None: ...
def _aware_utc(value: datetime | None) -> datetime | None:
    """Attach UTC to naive datetimes; SQLite returns naive values that are stored as UTC."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


async def list_conversations(
    store: SQLAlchemyChatStore,
    *,
    since: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession | None = None,
) -> list[ConversationRecord]:
    """List conversation headers, most recently active first.

    ``since`` intentionally excludes conversations whose ``last_activity_at`` is NULL.
    Ordering is ``last_activity_at`` descending with NULLS LAST (supported on the two
    tested dialects: SQLite >= 3.30 and PostgreSQL), tie-broken by ``id`` for stable paging.
    """
    if since is not None and since.tzinfo is not None:
        since = since.astimezone(timezone.utc)
    conversation = store.models.conversation
    query = sa.select(
        conversation.id,
        conversation.conversation_key,
        conversation.conversation_id,
        conversation.created_at,
        conversation.updated_at,
        conversation.message_count,
        conversation.first_activity_at,
        conversation.last_activity_at,
        conversation.total_input_tokens,
        conversation.total_output_tokens,
        conversation.total_cost,
    )
    if since is not None:
        query = query.where(conversation.last_activity_at >= since)
    query = (
        query.order_by(sa.nulls_last(conversation.last_activity_at.desc()), conversation.id).limit(limit).offset(offset)
    )
    async with store.sessions.scope(session) as (db, _owned):
        rows = (await db.execute(query)).all()
    return [
        ConversationRecord(
            id=row.id,
            conversation_key=row.conversation_key,
            conversation_id=row.conversation_id,
            created_at=_aware_utc(cast('datetime', row.created_at)),
            updated_at=_aware_utc(cast('datetime', row.updated_at)),
            message_count=row.message_count,
            first_activity_at=_aware_utc(cast('datetime | None', row.first_activity_at)),
            last_activity_at=_aware_utc(cast('datetime | None', row.last_activity_at)),
            total_input_tokens=row.total_input_tokens,
            total_output_tokens=row.total_output_tokens,
            total_cost=row.total_cost,
        )
        for row in rows
    ]
