"""Frozen result and analytics row types returned by the store."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal


def _empty_str_dict() -> dict[str, str]:
    return {}


__all__ = [
    'ConversationRecord',
    'ConversationUsage',
    'CostRow',
    'DailyUsage',
    'MessageDenorm',
    'ModelUsage',
    'PurgeReport',
    'RunRecord',
    'RunStats',
    'SaveResult',
    'ToolUsage',
]


@dataclass(frozen=True, kw_only=True)
class SaveResult:
    """Outcome of a save call: what was written, what was skipped as already persisted."""

    conversation_key: str
    conversation_pk: uuid.UUID
    run_id: str | None
    message_ids: tuple[uuid.UUID, ...]
    skipped: int = 0


@dataclass(frozen=True, kw_only=True)
class ConversationRecord:
    """Read-model of one conversation header row."""

    id: uuid.UUID
    conversation_key: str
    conversation_id: str | None
    created_at: datetime
    updated_at: datetime
    message_count: int
    first_activity_at: datetime | None
    last_activity_at: datetime | None
    total_input_tokens: int
    total_output_tokens: int
    total_cost: Decimal | None


@dataclass(frozen=True, kw_only=True)
class RunRecord:
    """Read-model of one agent run, structurally compatible with pydantic-ai-harness ``RunRecord``."""

    run_id: str
    conversation_id: str | None = None
    parent_run_id: str | None = None
    agent_name: str | None = None
    metadata: dict[str, str] = field(default_factory=_empty_str_dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    state: str | None = None
    model_name: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: Decimal | None = None


@dataclass(frozen=True, kw_only=True)
class MessageDenorm:
    """Denormalized query columns extracted from one ``ModelMessage`` at write time."""

    kind: str
    message_timestamp: datetime | None
    run_id: str | None
    conversation_id: str | None
    model_name: str | None
    provider_name: str | None
    provider_response_id: str | None
    finish_reason: str | None
    state: str | None
    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    cost: Decimal | None
    has_user_prompt: bool
    tool_call_count: int


@dataclass(frozen=True, kw_only=True)
class PurgeReport:
    """Row counts affected (or that would be affected, when ``dry_run``) by a purge."""

    conversations: int
    messages: int
    runs: int
    tool_calls: int
    dry_run: bool


@dataclass(frozen=True, kw_only=True)
class DailyUsage:
    day: date
    model_requests: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost: Decimal | None
    active_conversations: int


@dataclass(frozen=True, kw_only=True)
class ModelUsage:
    model_name: str | None
    provider_name: str | None
    model_requests: int
    input_tokens: int
    output_tokens: int
    cost: Decimal | None


@dataclass(frozen=True, kw_only=True)
class ConversationUsage:
    conversation_key: str
    message_count: int
    input_tokens: int
    output_tokens: int
    cost: Decimal | None
    first_activity_at: datetime | None
    last_activity_at: datetime | None


@dataclass(frozen=True, kw_only=True)
class RunStats:
    state: str | None
    run_count: int
    avg_duration_ms: float | None
    input_tokens: int
    output_tokens: int
    cost: Decimal | None


@dataclass(frozen=True, kw_only=True)
class ToolUsage:
    tool_name: str
    call_count: int
    returned_count: int
    error_count: int
    unanswered_count: int


@dataclass(frozen=True, kw_only=True)
class CostRow:
    """One row of a cost report; ``group`` is the day, model name or conversation key."""

    group: str
    model_requests: int
    input_tokens: int
    output_tokens: int
    cost: Decimal | None
