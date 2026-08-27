"""ORM model bases, default concrete models and the ModelSpec binding.

Design rules (mirroring how pydantic-ai stores messages officially):
- One row per ``ModelMessage``; the full payload lives in a JSON blob written with
  ``ModelMessagesTypeAdapter`` semantics. Parts are never shredded into per-kind tables.
- Only query keys are denormalized into real columns (timestamps, model, usage, cost, ...).
- Host applications subclass the abstract bases onto their own declarative ``Base`` to add
  columns (tenancy, foreign keys, enums); the package ships ready-made default models too.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import ClassVar

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, declared_attr, mapped_column

from ._base import Base, json_variant

__all__ = [
    'DEFAULT_SPEC',
    'BaseConversation',
    'BaseMessage',
    'BaseRun',
    'BaseToolCall',
    'DefaultConversation',
    'DefaultMessage',
    'DefaultRun',
    'DefaultToolCall',
    'ModelSpec',
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BaseConversation:
    """Abstract conversation header: identity plus rollups maintained by the write path."""

    __abstract__ = True

    __tablename__: ClassVar[str]

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(), primary_key=True, default=uuid.uuid4)
    conversation_key: Mapped[str] = mapped_column(sa.String(255), nullable=False, unique=True, index=True)
    conversation_id: Mapped[str | None] = mapped_column(sa.String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utcnow, server_default=sa.func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow, server_default=sa.func.now()
    )
    message_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0, server_default='0')
    first_activity_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    last_activity_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), index=True)
    total_input_tokens: Mapped[int] = mapped_column(sa.BigInteger, nullable=False, default=0, server_default='0')
    total_output_tokens: Mapped[int] = mapped_column(sa.BigInteger, nullable=False, default=0, server_default='0')
    total_cost: Mapped[Decimal | None] = mapped_column(sa.Numeric(18, 8))


class BaseMessage:
    """Abstract message row: one ``ModelMessage`` blob plus denormalized query columns."""

    __abstract__ = True

    __tablename__: ClassVar[str]

    #: Table name of the conversation table this message table points at; override when the
    #: host renames the conversation table.
    __pai_conversation_table__: ClassVar[str] = 'pai_conversations'

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(), primary_key=True, default=uuid.uuid4)

    @declared_attr
    def conversation_pk(cls) -> Mapped[uuid.UUID]:
        return mapped_column(
            sa.ForeignKey(f'{cls.__pai_conversation_table__}.id', ondelete='CASCADE'), nullable=False, index=True
        )

    seq: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    kind: Mapped[str] = mapped_column(sa.String(8), nullable=False)
    message: Mapped[dict[str, object]] = mapped_column(json_variant(), nullable=False)
    content_hash: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    run_id: Mapped[str | None] = mapped_column(sa.String(64), index=True)

    @declared_attr
    def parent_id(cls) -> Mapped[uuid.UUID | None]:
        # Reserved for future session-tree support (append-only log with replay); unused in v1.
        return mapped_column(sa.ForeignKey(f'{cls.__tablename__}.id', ondelete='SET NULL'), nullable=True)

    message_timestamp: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, default=_utcnow, server_default=sa.func.now()
    )
    model_name: Mapped[str | None] = mapped_column(sa.String(255))
    provider_name: Mapped[str | None] = mapped_column(sa.String(64))
    provider_response_id: Mapped[str | None] = mapped_column(sa.String(128))
    finish_reason: Mapped[str | None] = mapped_column(sa.String(32))
    state: Mapped[str | None] = mapped_column(sa.String(16))
    input_tokens: Mapped[int | None] = mapped_column(sa.BigInteger)
    output_tokens: Mapped[int | None] = mapped_column(sa.BigInteger)
    cache_read_tokens: Mapped[int | None] = mapped_column(sa.BigInteger)
    cache_write_tokens: Mapped[int | None] = mapped_column(sa.BigInteger)
    cost: Mapped[Decimal | None] = mapped_column(sa.Numeric(18, 8))
    has_user_prompt: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False, server_default=sa.false())
    tool_call_count: Mapped[int] = mapped_column(sa.SmallInteger, nullable=False, default=0, server_default='0')

    @declared_attr.directive
    def __table_args__(cls) -> tuple[sa.schema.SchemaItem, ...]:
        return (
            sa.UniqueConstraint('conversation_pk', 'seq', name=f'uq_{cls.__tablename__}_conversation_seq'),
            sa.Index(f'ix_{cls.__tablename__}_conv_run_hash', 'conversation_pk', 'run_id', 'content_hash'),
            sa.Index(f'ix_{cls.__tablename__}_message_timestamp', 'message_timestamp'),
            sa.Index(f'ix_{cls.__tablename__}_model_name', 'model_name'),
        )


class BaseRun:
    """Abstract run row: one ``Agent.run`` call with rollups for run-level analytics."""

    __abstract__ = True

    __tablename__: ClassVar[str]

    __pai_conversation_table__: ClassVar[str] = 'pai_conversations'

    run_id: Mapped[str] = mapped_column(sa.String(64), primary_key=True)

    @declared_attr
    def conversation_pk(cls) -> Mapped[uuid.UUID]:
        return mapped_column(
            sa.ForeignKey(f'{cls.__pai_conversation_table__}.id', ondelete='CASCADE'), nullable=False, index=True
        )

    parent_run_id: Mapped[str | None] = mapped_column(sa.String(64))
    agent_name: Mapped[str | None] = mapped_column(sa.String(255))
    started_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), index=True)
    finished_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(sa.Integer)
    state: Mapped[str | None] = mapped_column(sa.String(16))
    model_name: Mapped[str | None] = mapped_column(sa.String(255))
    provider_name: Mapped[str | None] = mapped_column(sa.String(64))
    message_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0, server_default='0')
    model_request_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0, server_default='0')
    tool_call_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0, server_default='0')
    input_tokens: Mapped[int] = mapped_column(sa.BigInteger, nullable=False, default=0, server_default='0')
    output_tokens: Mapped[int] = mapped_column(sa.BigInteger, nullable=False, default=0, server_default='0')
    cache_read_tokens: Mapped[int] = mapped_column(sa.BigInteger, nullable=False, default=0, server_default='0')
    cache_write_tokens: Mapped[int] = mapped_column(sa.BigInteger, nullable=False, default=0, server_default='0')
    cost: Mapped[Decimal | None] = mapped_column(sa.Numeric(18, 8))
    run_metadata: Mapped[dict[str, object] | None] = mapped_column(json_variant())


class BaseToolCall:
    """Abstract extracted tool call, populated when ``extract_tool_calls`` is enabled."""

    __abstract__ = True

    __tablename__: ClassVar[str]

    __pai_conversation_table__: ClassVar[str] = 'pai_conversations'
    __pai_message_table__: ClassVar[str] = 'pai_messages'

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(), primary_key=True, default=uuid.uuid4)

    @declared_attr
    def message_pk(cls) -> Mapped[uuid.UUID]:
        return mapped_column(sa.ForeignKey(f'{cls.__pai_message_table__}.id', ondelete='CASCADE'), nullable=False)

    @declared_attr
    def conversation_pk(cls) -> Mapped[uuid.UUID]:
        return mapped_column(
            sa.ForeignKey(f'{cls.__pai_conversation_table__}.id', ondelete='CASCADE'), nullable=False, index=True
        )

    run_id: Mapped[str | None] = mapped_column(sa.String(64))
    tool_call_id: Mapped[str] = mapped_column(sa.String(128), nullable=False)
    tool_name: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    args: Mapped[dict[str, object] | None] = mapped_column(json_variant())
    called_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    #: 'called' until a return is seen, then 'returned' | 'error' | 'unanswered'.
    status: Mapped[str] = mapped_column(sa.String(16), nullable=False, default='called', server_default='called')

    @declared_attr.directive
    def __table_args__(cls) -> tuple[sa.schema.SchemaItem, ...]:
        return (
            sa.Index(f'ix_{cls.__tablename__}_tool_name_called_at', 'tool_name', 'called_at'),
            sa.Index(f'ix_{cls.__tablename__}_conv_tool_call_id', 'conversation_pk', 'tool_call_id'),
        )


class DefaultConversation(BaseConversation, Base):
    __tablename__ = 'pai_conversations'


class DefaultMessage(BaseMessage, Base):
    __tablename__ = 'pai_messages'


class DefaultRun(BaseRun, Base):
    __tablename__ = 'pai_runs'


class DefaultToolCall(BaseToolCall, Base):
    __tablename__ = 'pai_tool_calls'


@dataclass(frozen=True)
class ModelSpec:
    """Binds the four concrete model classes the store operates on."""

    conversation: type[BaseConversation]
    message: type[BaseMessage]
    run: type[BaseRun]
    tool_call: type[BaseToolCall]

    def tables(self) -> tuple[sa.Table, ...]:
        return tuple(cls.__table__ for cls in (self.conversation, self.message, self.run, self.tool_call))  # type: ignore[attr-defined]


DEFAULT_SPEC = ModelSpec(
    conversation=DefaultConversation, message=DefaultMessage, run=DefaultRun, tool_call=DefaultToolCall
)
