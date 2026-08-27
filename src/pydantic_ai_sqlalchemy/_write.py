"""Write path: save runs and message batches, allocate sequences, maintain rollups.

Implemented by work unit U2. Signatures are frozen; see ``_store.SQLAlchemyChatStore``.

Implementation contract:
- Allocate ``seq`` as ``max(seq) + 1`` inside the transaction; insert the batch inside a
  SAVEPOINT (``session.begin_nested()``) and retry (up to 5 times, small jitter) on
  ``IntegrityError`` from the ``(conversation_pk, seq)`` unique constraint, so a
  caller-owned outer transaction survives collisions. Raise ``SequenceAllocationError``
  after the last retry.
- Idempotency: before inserting a batch with a ``run_id``, load existing content hashes for
  ``(conversation_pk, run_id)`` and skip the already-persisted multiset prefix.
- Maintain conversation rollups (message_count, first/last activity, token/cost totals) and
  upsert the run row (state, model, duration, counters) in the same transaction.
- ``final_message_id`` forces the primary key of the LAST ModelResponse row of the batch.
- Call ``_tool_calls.extract_and_store`` after inserting rows (no-op until U5 lands).
- Never mutate the input messages.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from pydantic_ai.messages import ModelMessage
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from . import _tool_calls
from ._exceptions import SequenceAllocationError, StoreError
from ._models import BaseConversation, BaseMessage, BaseRun
from ._serialize import dump_message, extract_denorm, message_content_hash, sanitize_payload
from ._types import MessageDenorm, SaveResult

if TYPE_CHECKING:
    from pydantic_ai.run import AgentRunResult

    from ._store import SQLAlchemyChatStore

__all__ = ['get_or_create_conversation', 'save_messages', 'save_run']

_MAX_SEQ_ATTEMPTS = 5
_MAX_RETRY_JITTER_SECONDS = 0.05
#: Model names stamped onto synthetic responses (e.g. by ``save_partial_run``); their zero
#: usage must not pollute token/cost rollups.
_SYNTHETIC_MODEL_NAMES = frozenset({'interrupted', 'error'})


async def get_or_create_conversation(
    store: SQLAlchemyChatStore,
    session: AsyncSession,
    *,
    conversation_key: str,
    conversation_id: str | None = None,
) -> Any:
    """Return the conversation row for ``conversation_key``, creating it if absent (flush, no commit)."""
    async with store.sessions.scope(session) as (db, _owned):
        return await _get_or_create_conversation(
            store, db, conversation_key=conversation_key, conversation_id=conversation_id
        )


async def save_run(
    store: SQLAlchemyChatStore,
    *,
    result: AgentRunResult[Any],
    conversation_key: str | None = None,
    agent_name: str | None = None,
    only_new: bool = True,
    final_message_id: uuid.UUID | None = None,
    session: AsyncSession | None = None,
) -> SaveResult:
    """Persist ``result.new_messages()`` (or ``all_messages()`` when ``only_new=False``)."""
    messages = list(result.new_messages() if only_new else result.all_messages())
    key = conversation_key
    if key is None:
        resolved = getattr(result, 'conversation_id', None)
        key = resolved if isinstance(resolved, str) else None
    if key is None:
        raise StoreError('save_run could not resolve a conversation key: pass conversation_key explicitly')
    raw_run_id = getattr(result, 'run_id', None)
    run_id = raw_run_id if isinstance(raw_run_id, str) else None
    return await _save_batch(
        store,
        messages=messages,
        conversation_key=key,
        run_id=run_id,
        agent_name=agent_name,
        final_message_id=final_message_id,
        session=session,
    )


async def save_messages(
    store: SQLAlchemyChatStore,
    *,
    messages: Sequence[ModelMessage],
    conversation_key: str,
    run_id: str | None = None,
    agent_name: str | None = None,
    final_message_id: uuid.UUID | None = None,
    session: AsyncSession | None = None,
) -> SaveResult:
    """Persist an explicit message batch into ``conversation_key``."""
    return await _save_batch(
        store,
        messages=messages,
        conversation_key=conversation_key,
        run_id=run_id,
        agent_name=agent_name,
        final_message_id=final_message_id,
        session=session,
    )


# -- internals ----------------------------------------------------------------------------


async def _save_batch(
    store: SQLAlchemyChatStore,
    *,
    messages: Sequence[ModelMessage],
    conversation_key: str,
    run_id: str | None,
    agent_name: str | None,
    final_message_id: uuid.UUID | None,
    session: AsyncSession | None,
) -> SaveResult:
    message_cls = store.models.message
    async with store.sessions.scope(session) as (db, _owned):
        dialect_name = db.get_bind().dialect.name
        payloads = [
            sanitize_payload(dump_message(message), dialect_name=dialect_name, custom=store.json_sanitizer)
            for message in messages
        ]
        denorms = [extract_denorm(message) for message in messages]
        hashes = [message_content_hash(message) for message in messages]

        conversation_id = next((denorm.conversation_id for denorm in denorms if denorm.conversation_id), None)
        conversation = await _get_or_create_conversation(
            store, db, conversation_key=conversation_key, conversation_id=conversation_id
        )
        conversation_pk = conversation.id

        skipped = 0
        if run_id is not None:
            skipped = await _count_persisted_prefix(
                db, message_cls, conversation_pk=conversation_pk, run_id=run_id, hashes=hashes
            )
        if skipped == len(messages):
            return SaveResult(
                conversation_key=conversation_key,
                conversation_pk=conversation_pk,
                run_id=run_id,
                message_ids=(),
                skipped=skipped,
            )

        kept_messages = list(messages[skipped:])
        kept_payloads = payloads[skipped:]
        kept_denorms = denorms[skipped:]
        kept_hashes = hashes[skipped:]

        rows = await _insert_batch_with_retry(
            db,
            message_cls,
            conversation_pk=conversation_pk,
            payloads=kept_payloads,
            denorms=kept_denorms,
            hashes=kept_hashes,
            run_id=run_id,
            final_message_id=final_message_id,
        )

        _apply_conversation_rollups(conversation, kept_denorms)
        if run_id is not None:
            await _upsert_run(
                db,
                store.models.run,
                run_id=run_id,
                conversation_pk=conversation_pk,
                agent_name=agent_name,
                denorms=kept_denorms,
            )
        await db.flush()
        await _tool_calls.extract_and_store(store, db, rows=rows, messages=kept_messages)

        return SaveResult(
            conversation_key=conversation_key,
            conversation_pk=conversation_pk,
            run_id=run_id,
            message_ids=tuple(row.id for row in rows),
            skipped=skipped,
        )


async def _get_or_create_conversation(
    store: SQLAlchemyChatStore,
    db: AsyncSession,
    *,
    conversation_key: str,
    conversation_id: str | None,
) -> BaseConversation:
    conversation_cls = store.models.conversation
    conversation = await db.scalar(
        sa.select(conversation_cls).where(conversation_cls.conversation_key == conversation_key)
    )
    if conversation is None:
        conversation = conversation_cls()
        conversation.conversation_key = conversation_key
        conversation.message_count = 0
        conversation.total_input_tokens = 0
        conversation.total_output_tokens = 0
        now = _utcnow()
        conversation.created_at = now
        conversation.updated_at = now
        if conversation_id is not None:
            conversation.conversation_id = conversation_id
        db.add(conversation)
        await db.flush()
    elif conversation_id is not None and conversation.conversation_id != conversation_id:
        conversation.conversation_id = conversation_id
    return conversation


async def _count_persisted_prefix(
    db: AsyncSession,
    message_cls: type[BaseMessage],
    *,
    conversation_pk: uuid.UUID,
    run_id: str,
    hashes: Sequence[str],
) -> int:
    """Length of the incoming-batch prefix already persisted for this run (multiset match)."""
    existing = await db.scalars(
        sa.select(message_cls.content_hash)
        .where(message_cls.conversation_pk == conversation_pk, message_cls.run_id == run_id)
        .order_by(message_cls.seq)
    )
    remaining = Counter(existing.all())
    skipped = 0
    for content_hash in hashes:
        if remaining[content_hash] <= 0:
            break
        remaining[content_hash] -= 1
        skipped += 1
    return skipped


async def _next_seq(db: AsyncSession, message_cls: type[BaseMessage], conversation_pk: uuid.UUID) -> int:
    current = await db.scalar(
        sa.select(sa.func.coalesce(sa.func.max(message_cls.seq), 0)).where(
            message_cls.conversation_pk == conversation_pk
        )
    )
    return int(current or 0) + 1


async def _insert_batch_with_retry(
    db: AsyncSession,
    message_cls: type[BaseMessage],
    *,
    conversation_pk: uuid.UUID,
    payloads: Sequence[Mapping[str, object]],
    denorms: Sequence[MessageDenorm],
    hashes: Sequence[str],
    run_id: str | None,
    final_message_id: uuid.UUID | None,
) -> list[BaseMessage]:
    """Insert the batch at ``max(seq) + 1`` inside a savepoint, retrying on seq collisions.

    The savepoint keeps a caller-owned outer transaction alive when a concurrent writer wins
    the ``(conversation_pk, seq)`` unique constraint; each retry re-reads ``max(seq)``.
    """
    last_error: IntegrityError | None = None
    for attempt in range(_MAX_SEQ_ATTEMPTS):
        base_seq = await _next_seq(db, message_cls, conversation_pk)
        rows = _build_rows(
            message_cls,
            conversation_pk=conversation_pk,
            base_seq=base_seq,
            payloads=payloads,
            denorms=denorms,
            hashes=hashes,
            run_id=run_id,
            final_message_id=final_message_id,
        )
        try:
            async with db.begin_nested():
                db.add_all(rows)
                await db.flush()
        except IntegrityError as error:
            last_error = error
            if attempt < _MAX_SEQ_ATTEMPTS - 1:
                await asyncio.sleep(random.uniform(0, _MAX_RETRY_JITTER_SECONDS))
            continue
        return rows
    raise SequenceAllocationError(
        f'could not allocate message sequences for conversation {conversation_pk} after {_MAX_SEQ_ATTEMPTS} attempts'
    ) from last_error


def _build_rows(
    message_cls: type[BaseMessage],
    *,
    conversation_pk: uuid.UUID,
    base_seq: int,
    payloads: Sequence[Mapping[str, object]],
    denorms: Sequence[MessageDenorm],
    hashes: Sequence[str],
    run_id: str | None,
    final_message_id: uuid.UUID | None,
) -> list[BaseMessage]:
    rows: list[BaseMessage] = []
    for offset, (payload, denorm, content_hash) in enumerate(zip(payloads, denorms, hashes, strict=True)):
        row = message_cls()
        setattr(row, 'conversation_pk', conversation_pk)  # noqa: B010  # declared_attr, no typed setter
        row.seq = base_seq + offset
        row.kind = denorm.kind
        row.message = dict(payload)
        row.content_hash = content_hash
        row.run_id = denorm.run_id if denorm.run_id is not None else run_id
        row.message_timestamp = denorm.message_timestamp
        row.created_at = _utcnow()
        row.model_name = denorm.model_name
        row.provider_name = denorm.provider_name
        row.provider_response_id = denorm.provider_response_id
        row.finish_reason = denorm.finish_reason
        row.state = denorm.state
        row.input_tokens = denorm.input_tokens
        row.output_tokens = denorm.output_tokens
        row.cache_read_tokens = denorm.cache_read_tokens
        row.cache_write_tokens = denorm.cache_write_tokens
        row.cost = denorm.cost
        row.has_user_prompt = denorm.has_user_prompt
        row.tool_call_count = denorm.tool_call_count
        rows.append(row)
    if final_message_id is not None:
        for row in reversed(rows):
            if row.kind == 'response':
                row.id = final_message_id
                break
    return rows


def _apply_conversation_rollups(conversation: BaseConversation, denorms: Sequence[MessageDenorm]) -> None:
    conversation.message_count += len(denorms)
    conversation.updated_at = _utcnow()
    timestamps = [denorm.message_timestamp for denorm in denorms if denorm.message_timestamp is not None]
    if timestamps:
        if conversation.first_activity_at is None:
            conversation.first_activity_at = min(timestamps)
        conversation.last_activity_at = max(timestamps)
    responses = _countable_responses(denorms)
    conversation.total_input_tokens += sum(denorm.input_tokens or 0 for denorm in responses)
    conversation.total_output_tokens += sum(denorm.output_tokens or 0 for denorm in responses)
    cost = _cost_sum(responses)
    if cost is not None:
        conversation.total_cost = (conversation.total_cost or Decimal(0)) + cost


async def _upsert_run(
    db: AsyncSession,
    run_cls: type[BaseRun],
    *,
    run_id: str,
    conversation_pk: uuid.UUID,
    agent_name: str | None,
    denorms: Sequence[MessageDenorm],
) -> None:
    run = await db.get(run_cls, run_id)
    if run is None:
        run = run_cls()
        run.run_id = run_id
        run.message_count = 0
        run.model_request_count = 0
        run.tool_call_count = 0
        run.input_tokens = 0
        run.output_tokens = 0
        run.cache_read_tokens = 0
        run.cache_write_tokens = 0
        db.add(run)
    setattr(run, 'conversation_pk', conversation_pk)  # noqa: B010  # declared_attr, no typed setter
    if agent_name is not None:
        run.agent_name = agent_name

    request_times = [
        denorm.message_timestamp
        for denorm in denorms
        if denorm.kind == 'request' and denorm.message_timestamp is not None
    ]
    response_times = [
        denorm.message_timestamp
        for denorm in denorms
        if denorm.kind == 'response' and denorm.message_timestamp is not None
    ]
    if run.started_at is None and request_times:
        run.started_at = min(request_times)
    if response_times:
        run.finished_at = max(response_times)
    if run.started_at is not None and run.finished_at is not None:
        run.duration_ms = int((_as_utc(run.finished_at) - _as_utc(run.started_at)).total_seconds() * 1000)

    responses = _countable_responses(denorms)
    if responses:
        last = responses[-1]
        run.state = last.state
        run.model_name = last.model_name
        run.provider_name = last.provider_name

    run.message_count += len(denorms)
    run.model_request_count += sum(1 for denorm in denorms if denorm.kind == 'response')
    run.tool_call_count += sum(denorm.tool_call_count for denorm in denorms)
    run.input_tokens += sum(denorm.input_tokens or 0 for denorm in responses)
    run.output_tokens += sum(denorm.output_tokens or 0 for denorm in responses)
    run.cache_read_tokens += sum(denorm.cache_read_tokens or 0 for denorm in responses)
    run.cache_write_tokens += sum(denorm.cache_write_tokens or 0 for denorm in responses)
    cost = _cost_sum(responses)
    if cost is not None:
        run.cost = (run.cost or Decimal(0)) + cost


def _countable_responses(denorms: Sequence[MessageDenorm]) -> list[MessageDenorm]:
    """Response denorms that count toward token/cost rollups (synthetic responses excluded)."""
    return [
        denorm for denorm in denorms if denorm.kind == 'response' and denorm.model_name not in _SYNTHETIC_MODEL_NAMES
    ]


def _cost_sum(responses: Sequence[MessageDenorm]) -> Decimal | None:
    costs = [denorm.cost for denorm in responses if denorm.cost is not None]
    if not costs:
        return None
    return sum(costs, Decimal(0))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """Treat naive datetimes (e.g. SQLite round-trips) as UTC so arithmetic never mixes awareness."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
