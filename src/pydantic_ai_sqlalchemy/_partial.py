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

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolReturnPart
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
    return ModelResponse(parts=[TextPart(content=text)], model_name=model_name)


def _answer_most_recent(open_calls: list[tuple[str, str]], tool_call_id: str) -> None:
    """Mark the most recent still-open call with ``tool_call_id`` as answered."""
    for index in range(len(open_calls) - 1, -1, -1):
        if open_calls[index][0] == tool_call_id:
            del open_calls[index]
            return


def close_unanswered_tool_calls(
    messages: Sequence[ModelMessage], *, note: str = UNANSWERED_TOOL_NOTE
) -> list[ModelMessage]:
    """Return a new list; unchanged when every tool call already has a return.

    Only function tool calls (part_kind ``tool-call``) are closed. Builtin tool calls
    (part_kind ``builtin-tool-call``) are skipped entirely: their returns live inside the
    ``ModelResponse`` itself as builtin return parts, so a synthetic function
    ``ToolReturnPart`` in a ``ModelRequest`` would be invalid history for them.

    Messages are walked in order and a return or retry answers the most recent open call
    with its ``tool_call_id``, so providers that reuse ids across turns still get the last,
    genuinely unanswered call closed.
    """
    open_calls: list[tuple[str, str]] = []  # (tool_call_id, tool_name), oldest first
    for message in messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if part.part_kind == 'tool-call':
                    open_calls.append((part.tool_call_id, part.tool_name))
        else:
            for part in message.parts:
                if part.part_kind == 'tool-return' or part.part_kind == 'retry-prompt':
                    _answer_most_recent(open_calls, part.tool_call_id)

    if not open_calls:
        return list(messages)
    closing_request = ModelRequest(
        parts=[
            ToolReturnPart(tool_name=tool_name, content=note, tool_call_id=tool_call_id)
            for tool_call_id, tool_name in open_calls
        ]
    )
    return [*messages, closing_request]


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
    transformed: list[ModelMessage] = list(messages)
    if close_deferred_tools:
        transformed = close_unanswered_tool_calls(transformed)
    if mark_interrupted and transformed:
        transformed.append(synthesize_interrupted_response())
    from . import _write

    return await _write.save_messages(
        store,
        messages=transformed,
        conversation_key=conversation_key,
        run_id=run_id,
        agent_name=agent_name,
        session=session,
    )
