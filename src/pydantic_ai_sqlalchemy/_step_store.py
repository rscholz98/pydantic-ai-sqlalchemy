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

from types import ModuleType
from typing import TYPE_CHECKING, cast

from pydantic import TypeAdapter
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from ._base import Base
from ._exceptions import StoreError
from ._serialize import dump_message, load_messages
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


def _harness() -> ModuleType:
    """Import the harness step-persistence module lazily; it is an optional dependency."""
    try:
        from pydantic_ai_harness import step_persistence
    except ImportError as exc:
        raise StoreError('pydantic-ai-harness is not installed; install pydantic-ai-sqlalchemy[harness]') from exc
    return step_persistence


_METADATA_ADAPTER: TypeAdapter[dict[str, str]] = TypeAdapter(dict[str, str])
_ENUM_ADAPTERS: dict[str, TypeAdapter[str]] = {}


def _validate_enum(sp: ModuleType, name: str, value: str) -> str:
    """Validate a stored string against a harness ``Literal`` type (e.g. ``EventKind``).

    Mirrors the reference stores' loud read path: an unknown stored value raises
    ``pydantic.ValidationError`` (a ``ValueError`` subclass) instead of leaking through.
    """
    adapter = _ENUM_ADAPTERS.get(name)
    if adapter is None:
        adapter = cast('TypeAdapter[str]', TypeAdapter(getattr(sp, name)))
        _ENUM_ADAPTERS[name] = adapter
    return adapter.validate_python(value)


def _run_record(sp: ModuleType, row: StepRun) -> RunRecord:
    return sp.RunRecord(
        run_id=row.run_id,
        conversation_id=row.conversation_id,
        parent_run_id=row.parent_run_id,
        agent_name=row.agent_name,
        metadata=_METADATA_ADAPTER.validate_python(row.run_metadata),
        started_at=row.started_at,
    )


def _event_record(sp: ModuleType, row: StepEvent) -> HarnessStepEvent:
    return sp.StepEvent(
        run_id=row.run_id,
        kind=_validate_enum(sp, 'EventKind', row.kind),
        step_index=row.step_index,
        timestamp=row.timestamp,
        conversation_id=row.conversation_id,
        parent_run_id=row.parent_run_id,
        agent_name=row.agent_name,
        tool_call_id=row.tool_call_id,
        tool_name=row.tool_name,
        error=row.error,
        metadata=_METADATA_ADAPTER.validate_python(row.event_metadata),
    )


def _snapshot_record(sp: ModuleType, row: StepSnapshot) -> ContinuableSnapshot:
    return sp.ContinuableSnapshot(
        run_id=row.run_id,
        step_index=row.step_index,
        messages=load_messages(cast('list[dict[str, object]]', row.messages)),
        conversation_id=row.conversation_id,
        parent_run_id=row.parent_run_id,
        agent_name=row.agent_name,
        timestamp=row.timestamp,
        state=_validate_enum(sp, 'SnapshotState', row.state),
    )


def _tool_effect_record(sp: ModuleType, row: StepToolEffect) -> ToolEffectRecord:
    return sp.ToolEffectRecord(
        tool_call_id=row.tool_call_id,
        tool_name=row.tool_name,
        run_id=row.run_id,
        status=_validate_enum(sp, 'ToolEffectStatus', row.status),
        started_at=row.started_at,
        ended_at=row.ended_at,
        idempotency_key=row.idempotency_key,
        effect_summary=row.effect_summary,
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
        _harness()
        async with self._sessions.scope() as (db, _owned):
            db.add(
                StepRun(
                    run_id=record.run_id,
                    conversation_id=record.conversation_id,
                    parent_run_id=record.parent_run_id,
                    agent_name=record.agent_name,
                    run_metadata=dict(record.metadata),
                    started_at=record.started_at,
                )
            )
            # A duplicate run_id violates the primary key; explicit run_id reuse is a
            # harness contract violation, so the IntegrityError propagates on purpose.
            await db.flush()

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        sp = _harness()
        async with self._sessions.scope() as (db, _owned):
            row = await db.get(StepRun, run_id)
            return None if row is None else _run_record(sp, row)

    async def list_runs(
        self, *, parent_run_id: str | None = None, conversation_id: str | None = None
    ) -> list[RunRecord]:
        sp = _harness()
        statement = sa.select(StepRun).order_by(StepRun.started_at.asc(), StepRun.run_id.asc())
        if parent_run_id is not None:
            statement = statement.where(StepRun.parent_run_id == parent_run_id)
        if conversation_id is not None:
            statement = statement.where(StepRun.conversation_id == conversation_id)
        async with self._sessions.scope() as (db, _owned):
            rows = (await db.scalars(statement)).all()
            return [_run_record(sp, row) for row in rows]

    async def append_event(self, event: HarnessStepEvent) -> None:
        _harness()
        async with self._sessions.scope() as (db, _owned):
            db.add(
                StepEvent(
                    run_id=event.run_id,
                    kind=event.kind,
                    step_index=event.step_index,
                    timestamp=event.timestamp,
                    conversation_id=event.conversation_id,
                    parent_run_id=event.parent_run_id,
                    agent_name=event.agent_name,
                    tool_call_id=event.tool_call_id,
                    tool_name=event.tool_name,
                    error=event.error,
                    event_metadata=dict(event.metadata),
                )
            )
            await db.flush()

    async def list_events(self, *, run_id: str) -> list[HarnessStepEvent]:
        sp = _harness()
        statement = sa.select(StepEvent).where(StepEvent.run_id == run_id).order_by(StepEvent.seq.asc())
        async with self._sessions.scope() as (db, _owned):
            rows = (await db.scalars(statement)).all()
            return [_event_record(sp, row) for row in rows]

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        _harness()
        async with self._sessions.scope() as (db, _owned):
            db.add(
                StepSnapshot(
                    run_id=snapshot.run_id,
                    step_index=snapshot.step_index,
                    conversation_id=snapshot.conversation_id,
                    parent_run_id=snapshot.parent_run_id,
                    agent_name=snapshot.agent_name,
                    timestamp=snapshot.timestamp,
                    state=snapshot.state,
                    messages=[dump_message(message) for message in snapshot.messages],
                )
            )
            await db.flush()

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        sp = _harness()
        statement = sa.select(StepSnapshot).where(StepSnapshot.run_id == run_id).order_by(StepSnapshot.seq.desc())
        if not include_interrupted:
            statement = statement.where(StepSnapshot.state == 'complete')
        async with self._sessions.scope() as (db, _owned):
            row = (await db.scalars(statement.limit(1))).first()
            return None if row is None else _snapshot_record(sp, row)

    async def list_snapshots(self, *, run_id: str, include_interrupted: bool = False) -> list[ContinuableSnapshot]:
        """Return every snapshot for the run in ``seq`` (capture) order.

        Additive beyond the harness ``StepStore`` protocol: the conversation-search
        ``SnapshotStore`` surface expects it. Skips states other than ``complete``
        unless ``include_interrupted`` is set, matching ``latest_snapshot``.
        """
        sp = _harness()
        statement = sa.select(StepSnapshot).where(StepSnapshot.run_id == run_id).order_by(StepSnapshot.seq.asc())
        if not include_interrupted:
            statement = statement.where(StepSnapshot.state == 'complete')
        async with self._sessions.scope() as (db, _owned):
            rows = (await db.scalars(statement)).all()
            return [_snapshot_record(sp, row) for row in rows]

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        _harness()
        async with self._sessions.scope() as (db, _owned):
            row = await db.get(StepToolEffect, (record.run_id, record.tool_call_id))
            if row is None:
                db.add(
                    StepToolEffect(
                        run_id=record.run_id,
                        tool_call_id=record.tool_call_id,
                        tool_name=record.tool_name,
                        status=record.status,
                        started_at=record.started_at,
                        ended_at=record.ended_at,
                        idempotency_key=record.idempotency_key,
                        effect_summary=record.effect_summary,
                    )
                )
            else:
                row.status = record.status
                row.ended_at = record.ended_at
                row.idempotency_key = record.idempotency_key
                row.effect_summary = record.effect_summary
            await db.flush()

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        sp = _harness()
        async with self._sessions.scope() as (db, _owned):
            row = await db.get(StepToolEffect, (run_id, tool_call_id))
            return None if row is None else _tool_effect_record(sp, row)

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        sp = _harness()
        statement = (
            sa.select(StepToolEffect)
            .where(StepToolEffect.run_id == run_id, StepToolEffect.status == 'started')
            .order_by(StepToolEffect.started_at.asc(), StepToolEffect.tool_call_id.asc())
        )
        async with self._sessions.scope() as (db, _owned):
            rows = (await db.scalars(statement)).all()
            return [_tool_effect_record(sp, row) for row in rows]
