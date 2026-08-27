"""Tool-call extraction into the analytics side table.

Implemented by work unit U5. The signature is frozen because the write path (U2) calls
this seam after inserting message rows.

Behaviour:
- ``rows`` are the persisted ORM message instances, index-aligned with ``messages``.
- Every ``tool-call``/``builtin-tool-call`` part of a response creates one tool-call row
  (status 'called'); answering parts of request messages update status to 'returned',
  'error' (``RetryPromptPart``) or 'unanswered' (synthetic close note). Answers match
  in-batch calls first, then earlier saves via ``(conversation_pk, tool_call_id)``.
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

_CALL_PART_KINDS = ('tool-call', 'builtin-tool-call')


def _coerce_args(raw: object) -> dict[str, object] | None:
    """Normalize a tool-call ``args`` value to the JSON column shape.

    Dicts pass through, JSON-object strings are parsed, anything else stringly is kept
    under a ``raw`` key so no argument payload is ever silently dropped.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        return cast('dict[str, object]', raw)
    if isinstance(raw, str):
        try:
            parsed: object = json.loads(raw)
        except ValueError:
            return {'raw': raw}
        if isinstance(parsed, dict):
            return cast('dict[str, object]', parsed)
        return {'raw': raw}
    return {'raw': str(raw)}


def _answer_status(part: object) -> tuple[str, str] | None:
    """Return ``(tool_call_id, new_status)`` when ``part`` answers a tool call, else ``None``."""
    if isinstance(part, ToolReturnPart):
        status = 'unanswered' if part.content == UNANSWERED_TOOL_NOTE else 'returned'
        return part.tool_call_id, status
    if isinstance(part, RetryPromptPart) and part.tool_name is not None:
        return part.tool_call_id, 'error'
    return None


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
                if getattr(part, 'part_kind', None) not in _CALL_PART_KINDS:
                    continue
                tool_call_id = getattr(part, 'tool_call_id', None)
                tool_name = getattr(part, 'tool_name', None)
                if not isinstance(tool_call_id, str) or not isinstance(tool_name, str):
                    continue
                record = tool_call_cls()
                record.message_pk = row.id
                record.conversation_pk = row.conversation_pk
                record.run_id = row.run_id
                record.tool_call_id = tool_call_id
                record.tool_name = tool_name
                record.args = _coerce_args(getattr(part, 'args', None))
                record.called_at = row.message_timestamp
                record.status = 'called'
                session.add(record)
                inserted += 1
                batch_calls[(row.conversation_pk, tool_call_id)] = record
        else:
            for part in message.parts:
                answer = _answer_status(part)
                if answer is None:
                    continue
                tool_call_id, status = answer
                in_batch = batch_calls.get((row.conversation_pk, tool_call_id))
                if in_batch is not None:
                    if in_batch.status == 'called':
                        in_batch.status = status
                    continue
                await session.execute(
                    sa.update(tool_call_cls)
                    .where(
                        tool_call_cls.conversation_pk == row.conversation_pk,
                        tool_call_cls.tool_call_id == tool_call_id,
                        tool_call_cls.status == 'called',
                    )
                    .values(status=status)
                )

    await session.flush()
    return inserted
