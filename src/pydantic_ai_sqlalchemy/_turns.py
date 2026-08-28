"""Pure turn-boundary utilities over in-memory message lists (no database access).

Implemented by work unit U3. Signatures are frozen.

A turn starts at a ``ModelRequest`` that contains a ``user-prompt`` part and contains no
``tool-return`` and no ``retry-prompt`` parts: a deferred/HITL resume request can carry a
tool return alongside a fresh user prompt, and such a request continues its turn. Slicing
must keep tool-call/tool-return pairs whole: never cut between a response with tool calls
and the request carrying their returns.

Part detection uses ``part_kind`` string checks (mirroring ``_serialize.extract_denorm``)
so the module stays tolerant of part types added or renamed across pydantic-ai releases.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pydantic_ai.messages as _pai_messages
from pydantic_ai.messages import ModelMessage

__all__ = ['messages_to_text', 'slice_to_recent_turns']

_NON_BOUNDARY_PART_KINDS = ('tool-return', 'retry-prompt')

#: ``TextContent`` user-content items carry real user text; guarded for older releases.
_TEXT_CONTENT_TYPE: type | None = getattr(_pai_messages, 'TextContent', None)


def _part_kind(part: object) -> str | None:
    kind = getattr(part, 'part_kind', None)
    return kind if isinstance(kind, str) else None


def _is_turn_boundary(message: ModelMessage) -> bool:
    if message.kind != 'request':
        return False
    has_user_prompt = False
    for part in message.parts:
        part_kind = _part_kind(part)
        if part_kind == 'user-prompt':
            has_user_prompt = True
        elif part_kind in _NON_BOUNDARY_PART_KINDS:
            return False
    return has_user_prompt


def slice_to_recent_turns(messages: Sequence[ModelMessage], *, max_turns: int) -> list[ModelMessage]:
    """Return the suffix of ``messages`` covering at most the last ``max_turns`` turns."""
    if max_turns <= 0:
        return []
    boundaries = [index for index, message in enumerate(messages) if _is_turn_boundary(message)]
    if len(boundaries) < max_turns:
        return list(messages)
    start = boundaries[-max_turns]
    return list(messages[start:])


def _content_item_text(item: object) -> str | None:
    if isinstance(item, str):
        return item
    if _TEXT_CONTENT_TYPE is not None and isinstance(item, _TEXT_CONTENT_TYPE):
        content = getattr(item, 'content', None)
        return content if isinstance(content, str) else None
    return None


def _user_prompt_text(part: object) -> str:
    content = getattr(part, 'content', None)
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        items = cast('Sequence[object]', content)
        texts = [text for item in items if (text := _content_item_text(item)) is not None]
        return '\n'.join(text for text in texts if text)
    return ''


def messages_to_text(messages: Sequence[ModelMessage]) -> str:
    """Render a ``User: ...`` / ``Assistant: ...`` plain-text transcript (text parts only).

    A user prompt whose content carries no text at all (e.g. image-only) still produces a
    ``User: [non-text content]`` line so the turn stays visible in the transcript.
    """
    entries: list[str] = []
    for message in messages:
        if message.kind == 'request':
            prompt_parts = [part for part in message.parts if _part_kind(part) == 'user-prompt']
            texts = [text for part in prompt_parts if (text := _user_prompt_text(part))]
            if not texts and prompt_parts:
                texts = ['[non-text content]']
            prefix = 'User: '
        else:
            contents = (getattr(part, 'content', None) for part in message.parts if _part_kind(part) == 'text')
            texts = [content for content in contents if isinstance(content, str) and content]
            prefix = 'Assistant: '
        if texts:
            entries.append(prefix + '\n'.join(texts))
    return '\n\n'.join(entries)
