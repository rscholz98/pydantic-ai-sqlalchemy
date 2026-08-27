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

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import ModelMessage
from sqlalchemy.ext.asyncio import AsyncSession

from ._types import SaveResult

if TYPE_CHECKING:
    from pydantic_ai.run import AgentRunResult

    from ._store import SQLAlchemyChatStore

__all__ = ['get_or_create_conversation', 'save_messages', 'save_run']


async def get_or_create_conversation(
    store: SQLAlchemyChatStore,
    session: AsyncSession,
    *,
    conversation_key: str,
    conversation_id: str | None = None,
) -> Any:
    """Return the conversation row for ``conversation_key``, creating it if absent (flush, no commit)."""
    raise NotImplementedError('implemented in unit U2')


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
    raise NotImplementedError('implemented in unit U2')


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
    raise NotImplementedError('implemented in unit U2')
