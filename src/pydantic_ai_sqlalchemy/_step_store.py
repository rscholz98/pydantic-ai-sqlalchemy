"""``SQLAlchemyStepStore``: full pydantic-ai-harness ``StepStore`` protocol over SQL tables.

Implemented by work unit U13. Signatures are frozen and mirror the harness protocol
exactly (all async, kw-only after the first argument). The class must structurally
satisfy ``pydantic_ai_harness.step_persistence.StepStore`` (runtime-checkable).

Implementation contract:
- Tables are the concrete models in ``_step_models`` (mirroring the official SQLite DDL):
  event/snapshot ordering uses the autoincrement ``seq``, never ``step_index`` (it resets
  across runs).
- ``register_run`` must surface a duplicate ``run_id`` as an error (PK violation): an
  explicit ``run_id`` is single-shot.
- Snapshots store the whole history via ``ModelMessagesTypeAdapter`` semantics
  (``_serialize.dump_message`` per message).
- ``pydantic-ai-harness`` is an optional dependency (extra ``harness``); import its types
  lazily inside methods and raise a clear ``StoreError`` when it is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from ._base import Base
from ._session import SessionProvider
from ._step_models import StepEvent, StepRun, StepSnapshot, StepToolEffect

if TYPE_CHECKING:
    from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, ToolEffectRecord
    from pydantic_ai_harness.step_persistence import StepEvent as HarnessStepEvent

__all__ = ['SQLAlchemyStepStore']

import sqlalchemy as sa

_STEP_TABLES = cast(
    'tuple[sa.Table, ...]', (StepRun.__table__, StepEvent.__table__, StepSnapshot.__table__, StepToolEffect.__table__)
)


class SQLAlchemyStepStore:
    """Async SQLAlchemy backend for pydantic-ai-harness step persistence."""

    def __init__(self, bind: AsyncEngine | async_sessionmaker[AsyncSession]) -> None:
        self._sessions = SessionProvider(bind)

    async def create_tables(self) -> None:
        async with self._sessions.scope() as (session, _owned):
            connection = await session.connection()
            await connection.run_sync(
                lambda sync_connection: Base.metadata.create_all(sync_connection, tables=list(_STEP_TABLES))
            )

    async def register_run(self, record: RunRecord) -> None:
        raise NotImplementedError('implemented in unit U13')

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        raise NotImplementedError('implemented in unit U13')

    async def list_runs(
        self, *, parent_run_id: str | None = None, conversation_id: str | None = None
    ) -> list[RunRecord]:
        raise NotImplementedError('implemented in unit U13')

    async def append_event(self, event: HarnessStepEvent) -> None:
        raise NotImplementedError('implemented in unit U13')

    async def list_events(self, *, run_id: str) -> list[HarnessStepEvent]:
        raise NotImplementedError('implemented in unit U13')

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        raise NotImplementedError('implemented in unit U13')

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        raise NotImplementedError('implemented in unit U13')

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        raise NotImplementedError('implemented in unit U13')

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        raise NotImplementedError('implemented in unit U13')

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        raise NotImplementedError('implemented in unit U13')
