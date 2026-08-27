"""ORM models backing the pydantic-ai-harness ``StepStore`` protocol.

These mirror the official SQLite DDL shipped by pydantic-ai-harness step-persistence
(runs, events, snapshots, tool_effects). They are concrete package-owned models; the
step store is an ecosystem-interop surface, not a host-extension surface.
"""

from __future__ import annotations

from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from ._base import Base, autoincrement_pk_variant, json_variant

__all__ = ['StepEvent', 'StepRun', 'StepSnapshot', 'StepToolEffect']


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StepRun(Base):
    __tablename__ = 'pai_sp_runs'

    run_id: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    conversation_id: Mapped[str | None] = mapped_column(sa.String(64), index=True)
    parent_run_id: Mapped[str | None] = mapped_column(sa.String(64), index=True)
    agent_name: Mapped[str | None] = mapped_column(sa.String(255))
    run_metadata: Mapped[dict[str, object]] = mapped_column(json_variant(), nullable=False, default=dict)
    started_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )


class StepEvent(Base):
    __tablename__ = 'pai_sp_events'
    __table_args__ = (sa.Index('ix_pai_sp_events_run_seq', 'run_id', 'seq'),)

    seq: Mapped[int] = mapped_column(autoincrement_pk_variant(), primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    kind: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    step_index: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=_utcnow)
    conversation_id: Mapped[str | None] = mapped_column(sa.String(64))
    parent_run_id: Mapped[str | None] = mapped_column(sa.String(64))
    agent_name: Mapped[str | None] = mapped_column(sa.String(255))
    tool_call_id: Mapped[str | None] = mapped_column(sa.String(128))
    tool_name: Mapped[str | None] = mapped_column(sa.String(255))
    error: Mapped[str | None] = mapped_column(sa.Text)
    event_metadata: Mapped[dict[str, object]] = mapped_column(json_variant(), nullable=False, default=dict)


class StepSnapshot(Base):
    __tablename__ = 'pai_sp_snapshots'
    __table_args__ = (sa.Index('ix_pai_sp_snapshots_run_seq', 'run_id', 'seq'),)

    seq: Mapped[int] = mapped_column(autoincrement_pk_variant(), primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    step_index: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    conversation_id: Mapped[str | None] = mapped_column(sa.String(64))
    parent_run_id: Mapped[str | None] = mapped_column(sa.String(64))
    agent_name: Mapped[str | None] = mapped_column(sa.String(255))
    timestamp: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=_utcnow)
    state: Mapped[str] = mapped_column(sa.String(16), nullable=False, default='complete', server_default='complete')
    #: The whole message history as one JSON array written with ModelMessagesTypeAdapter semantics.
    messages: Mapped[list[object]] = mapped_column(json_variant(), nullable=False)


class StepToolEffect(Base):
    __tablename__ = 'pai_sp_tool_effects'

    run_id: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    tool_call_id: Mapped[str] = mapped_column(sa.String(128), primary_key=True)
    tool_name: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    status: Mapped[str] = mapped_column(sa.String(16), nullable=False)
    started_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=_utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    idempotency_key: Mapped[str | None] = mapped_column(sa.String(255))
    effect_summary: Mapped[str | None] = mapped_column(sa.Text)
