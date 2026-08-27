"""Tool-call extraction into the analytics side table.

Work unit U5 replaces the no-op body below. The signature is frozen because the write
path (U2) calls this seam after inserting message rows.

Implementation contract (U5):
- ``rows`` are the persisted ORM message instances, index-aligned with ``messages``.
- For every ``ToolCallPart``/``NativeToolCallPart`` create one tool-call row (status
  'called'); match returns within the batch and across earlier batches of the same
  conversation via ``(conversation_pk, tool_call_id)`` and update status to 'returned',
  'error' (RetryPromptPart) or 'unanswered' (synthetic close).
- Respect ``store.extract_tool_calls_enabled``; stay a no-op when disabled.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelMessage
from sqlalchemy.ext.asyncio import AsyncSession

if TYPE_CHECKING:
    from ._models import BaseMessage
    from ._store import SQLAlchemyChatStore

__all__ = ['extract_and_store']


async def extract_and_store(
    store: SQLAlchemyChatStore,
    session: AsyncSession,
    *,
    rows: Sequence[BaseMessage],
    messages: Sequence[ModelMessage],
) -> int:
    """Extract tool calls from a just-persisted batch; returns the number of rows written."""
    return 0
