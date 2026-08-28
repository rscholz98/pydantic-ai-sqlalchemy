"""Tool-call extraction into the analytics side table.

Implemented by work unit U5. The signature is frozen because the write path (U2) calls
this seam after inserting message rows.

Behaviour:
- ``rows`` are the persisted ORM message instances, index-aligned with ``messages``.
- Every ``tool-call``/``builtin-tool-call`` part of a response creates one tool-call row
  (status 'called'). Answers update status: ``ToolReturnPart`` to 'returned' (or
  'unanswered' for the synthetic close note), ``RetryPromptPart`` tied to a tool to
  'error', and builtin returns (which ride inside responses) to 'returned'.
- Answers match in-batch calls first, then the most recent still-'called' row of the
  conversation from earlier saves via ``(conversation_pk, tool_call_id)``; scoping to the
  most recent row keeps providers that reuse counter-style ids from flipping older calls.
- Respects ``store.extract_tool_calls_enabled``; stays a no-op when disabled.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import sqlalchemy as sa
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, ToolReturnPart
from sqlalchemy.ext.asyncio import AsyncSession

from ._partial import UNANSWERED_TOOL_NOTE

if TYPE_CHECKING:
    from ._models import BaseMessage, BaseToolCall
    from ._store import SQLAlchemyChatStore

__all__ = ['extract_and_store']

# Mirrors the part kinds counted by _serialize.extract_denorm (tool_call_count); keep in sync.
_CALL_PART_KINDS = ('tool-call', 'builtin-tool-call')
_BUILTIN_RETURN_KIND = 'builtin-tool-return'


def _coerce_args(raw: str | dict[str, object] | None) -> dict[str, object] | None:
    """Normalize a tool-call ``args`` value to the JSON column shape.

    Dicts pass through and JSON-object strings are parsed; any other string is kept under
    a ``raw`` key so no argument payload is ever silently dropped.
    """
    if raw is None or isinstance(raw, dict):
        return raw
    try:
        parsed: object = json.loads(raw)
    except ValueError:
        return {'raw': raw}
    if isinstance(parsed, dict):
        return cast('dict[str, object]', parsed)
    return {'raw': raw}


def _answer_status(part: object) -> tuple[str, str] | None:
    """Return ``(tool_call_id, new_status)`` when a request ``part`` answers a tool call."""
    if isinstance(part, ToolReturnPart):
        status = 'unanswered' if part.content == UNANSWERED_TOOL_NOTE else 'returned'
        return part.tool_call_id, status
    if isinstance(part, RetryPromptPart) and part.tool_name is not None:
        return part.tool_call_id, 'error'
    return None


def _recency_key(record: BaseToolCall) -> tuple[int, float]:
    called_at = record.called_at
    return (0, 0.0) if called_at is None else (1, called_at.timestamp())


async def _apply_answer(
    session: AsyncSession,
    tool_call_cls: type[BaseToolCall],
    batch_calls: dict[tuple[uuid.UUID, str], BaseToolCall],
    *,
    conversation_pk: uuid.UUID,
    tool_call_id: str,
    status: str,
) -> None:
    """Resolve one answer: in-batch call first, else the newest still-'called' earlier row."""
    in_batch = batch_calls.get((conversation_pk, tool_call_id))
    if in_batch is not None:
        if in_batch.status == 'called':
            in_batch.status = status
        return
    result = await session.execute(
        sa.select(tool_call_cls).where(
            tool_call_cls.conversation_pk == conversation_pk,
            tool_call_cls.tool_call_id == tool_call_id,
            tool_call_cls.status == 'called',
        )
    )
    candidates = list(result.scalars())
    if candidates:
        max(candidates, key=_recency_key).status = status


async def extract_and_store(
    store: SQLAlchemyChatStore,
    session: AsyncSession,
    *,
    rows: Sequence[BaseMessage],
    messages: Sequence[ModelMessage],
) -> int:
    """Extract tool calls from a just-persisted batch; returns the number of rows written."""
    if not store.extract_tool_calls_enabled:
        return 0

    tool_call_cls = store.models.tool_call
    inserted = 0
    batch_calls: dict[tuple[uuid.UUID, str], BaseToolCall] = {}

    for row, message in zip(rows, messages, strict=True):
        if isinstance(message, ModelResponse):
            for part in message.parts:
                part_kind = getattr(part, 'part_kind', None)
                if part_kind in _CALL_PART_KINDS:
                    tool_call_id = getattr(part, 'tool_call_id', None)
                    tool_name = getattr(part, 'tool_name', None)
                    if not isinstance(tool_call_id, str) or not isinstance(tool_name, str):
                        continue
                    raw_args_attr: object = getattr(part, 'args', None)
                    raw_args: str | dict[str, object] | None = None
                    if isinstance(raw_args_attr, str):
                        raw_args = raw_args_attr
                    elif isinstance(raw_args_attr, dict):
                        raw_args = cast('dict[str, object]', raw_args_attr)
                    record = tool_call_cls()
                    record.message_pk = row.id
                    record.conversation_pk = row.conversation_pk
                    record.run_id = row.run_id
                    record.tool_call_id = tool_call_id
                    record.tool_name = tool_name
                    record.args = _coerce_args(raw_args)
                    record.called_at = row.message_timestamp
                    record.status = 'called'
                    session.add(record)
                    inserted += 1
                    batch_calls[(row.conversation_pk, tool_call_id)] = record
                elif part_kind == _BUILTIN_RETURN_KIND:
                    tool_call_id = getattr(part, 'tool_call_id', None)
                    if isinstance(tool_call_id, str):
                        await _apply_answer(
                            session,
                            tool_call_cls,
                            batch_calls,
                            conversation_pk=row.conversation_pk,
                            tool_call_id=tool_call_id,
                            status='returned',
                        )
        else:
            for part in message.parts:
                answer = _answer_status(part)
                if answer is None:
                    continue
                tool_call_id, status = answer
                await _apply_answer(
                    session,
                    tool_call_cls,
                    batch_calls,
                    conversation_pk=row.conversation_pk,
                    tool_call_id=tool_call_id,
                    status=status,
                )

    await session.flush()
    return inserted
