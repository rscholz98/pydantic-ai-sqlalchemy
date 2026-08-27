"""Pure turn-boundary utilities over in-memory message lists (no database access).

Implemented by work unit U3. Signatures are frozen.

A turn starts at a ``ModelRequest`` containing a ``user-prompt`` part. Slicing must keep
tool-call/tool-return pairs whole: never cut between a response with tool calls and the
request carrying their returns.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai.messages import ModelMessage

__all__ = ['messages_to_text', 'slice_to_recent_turns']


def slice_to_recent_turns(messages: Sequence[ModelMessage], *, max_turns: int) -> list[ModelMessage]:
    """Return the suffix of ``messages`` covering at most the last ``max_turns`` turns."""
    raise NotImplementedError('implemented in unit U3')


def messages_to_text(messages: Sequence[ModelMessage]) -> str:
    """Render a ``User: ...`` / ``Assistant: ...`` plain-text transcript (text parts only)."""
    raise NotImplementedError('implemented in unit U3')
