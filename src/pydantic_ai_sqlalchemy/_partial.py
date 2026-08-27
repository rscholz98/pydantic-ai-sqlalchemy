"""Partial-run utilities: interrupted turns and unanswered deferred tool calls.

Implemented by work unit U4. Signatures are frozen.

Implementation contract:
- Never mutate input messages (``MessageHistoryMutatedWarning``); build new objects, using
  ``dataclasses.replace`` where a copy of an existing message is needed.
- ``synthesize_interrupted_response`` returns a ``ModelResponse`` with a single ``TextPart``
  and ``model_name`` set to the marker so analytics can filter synthetic rows out.
- ``close_unanswered_tool_calls`` appends a ``ModelRequest`` of synthetic ``ToolReturnPart``s
  for every tool call in ``messages`` that has no matching return, so persisted history never
  ends on an unanswered tool call (which would break the next run).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelMessage, ModelResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ._types import SaveResult

if TYPE_CHECKING:
    from ._store import SQLAlchemyChatStore

__all__ = ['close_unanswered_tool_calls', 'save_partial_run', 'synthesize_interrupted_response']

INTERRUPTED_MODEL_NAME = 'interrupted'
INTERRUPTED_TEXT = '[interrupted] Client disconnected before the answer completed'
UNANSWERED_TOOL_NOTE = 'Tool call was interrupted before a result was produced'


def synthesize_interrupted_response(
    *, model_name: str = INTERRUPTED_MODEL_NAME, text: str = INTERRUPTED_TEXT
) -> ModelResponse:
    raise NotImplementedError('implemented in unit U4')


def close_unanswered_tool_calls(
    messages: Sequence[ModelMessage], *, note: str = UNANSWERED_TOOL_NOTE
) -> list[ModelMessage]:
    """Return a new list; unchanged when every tool call already has a return."""
    raise NotImplementedError('implemented in unit U4')


async def save_partial_run(
    store: SQLAlchemyChatStore,
    *,
    messages: Sequence[ModelMessage],
    conversation_key: str,
    run_id: str | None = None,
    agent_name: str | None = None,
    mark_interrupted: bool = True,
    close_deferred_tools: bool = True,
    session: AsyncSession | None = None,
) -> SaveResult:
    """Persist a cancelled/failed run's partial messages, closed and marked as configured."""
    raise NotImplementedError('implemented in unit U4')
