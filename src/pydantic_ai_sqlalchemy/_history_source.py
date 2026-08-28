"""Harness ``HistorySource`` read seam: run listing and per-run history.

Implemented by work unit U9. Signatures are frozen; see ``_store.SQLAlchemyChatStore``.

The store's ``list_runs``/``run_history`` methods must structurally satisfy the
``pydantic_ai_harness.conversation_search.HistorySource`` runtime-checkable protocol.
The harness import stays lazy and optional; ``pydantic_ai_sqlalchemy.RunRecord`` is the
structural stand-in when the harness is absent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast

import sqlalchemy as sa
from pydantic_ai.messages import ModelMessage
from sqlalchemy.ext.asyncio import AsyncSession

from ._serialize import load_messages
from ._types import RunRecord

if TYPE_CHECKING:
    from ._store import SQLAlchemyChatStore

__all__ = ['list_runs', 'run_history']


def _coerce_metadata(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in cast('dict[object, object]', raw).items()}


def _sort_key(record: RunRecord) -> tuple[int, datetime]:
    """Ascending ``started_at`` with ``None`` values last (portable across backends)."""
    if record.started_at is None:
        # The sentinel is never compared against real timestamps (the leading int differs).
        return 1, datetime.min.replace(tzinfo=timezone.utc)
    return 0, record.started_at


async def list_runs(
    store: SQLAlchemyChatStore,
    *,
    conversation_key: str | None = None,
    session: AsyncSession | None = None,
) -> list[RunRecord]:
    """All runs (optionally scoped to one conversation) sorted by ``started_at`` ascending."""
    run_cls = store.models.run
    conversation_cls = store.models.conversation
    statement = sa.select(run_cls, conversation_cls.conversation_id, conversation_cls.conversation_key).join(
        conversation_cls, run_cls.conversation_pk == conversation_cls.id
    )
    if conversation_key is not None:
        statement = statement.where(conversation_cls.conversation_key == conversation_key)

    async with store.sessions.scope(session) as (db, _owned):
        rows = (await db.execute(statement)).all()

    records = [
        RunRecord(
            run_id=run.run_id,
            conversation_id=row_conversation_id or row_conversation_key,
            parent_run_id=run.parent_run_id,
            agent_name=run.agent_name,
            metadata=_coerce_metadata(run.run_metadata),
            started_at=run.started_at,
            finished_at=run.finished_at,
            state=run.state,
            model_name=run.model_name,
            input_tokens=int(run.input_tokens),
            output_tokens=int(run.output_tokens),
            cost=run.cost,
        )
        for run, row_conversation_id, row_conversation_key in rows
    ]
    records.sort(key=_sort_key)
    return records


async def run_history(
    store: SQLAlchemyChatStore,
    *,
    run_id: str,
    session: AsyncSession | None = None,
) -> list[ModelMessage]:
    """The messages of one run, ordered by ``seq``."""
    message_cls = store.models.message
    statement = sa.select(message_cls.message).where(message_cls.run_id == run_id).order_by(message_cls.seq.asc())

    async with store.sessions.scope(session) as (db, _owned):
        payloads = (await db.execute(statement)).scalars().all()

    return load_messages(payloads)
