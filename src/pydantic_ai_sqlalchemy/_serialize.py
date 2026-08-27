"""Serialization core: round-trip, content hashing, denorm extraction, dialect sanitizing.

This is the single choke point for pydantic-ai message (de)serialization and for
version-compat shims across pydantic-ai releases. Every other module goes through it.

Rules:
- Always serialize through ``ModelMessagesTypeAdapter`` (its ``ser_json_bytes='base64'``
  config is load-bearing for ``BinaryContent`` round-trips). Never build a private adapter.
- Never mutate message objects in place (``MessageHistoryMutatedWarning``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from typing import cast

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse

__all__ = [
    'dump_message',
    'extract_denorm',
    'load_messages',
    'message_content_hash',
    'sanitize_payload',
    'strip_nuls',
]

from ._types import MessageDenorm

JsonSanitizer = Callable[[object], object]


def dump_message(message: ModelMessage) -> dict[str, object]:
    """Dump one message to a JSON-compatible dict via the official adapter semantics."""
    payload = ModelMessagesTypeAdapter.dump_python([message], mode='json')
    first = payload[0]
    assert isinstance(first, dict)
    return cast('dict[str, object]', first)


def load_messages(payloads: Sequence[Mapping[str, object]]) -> list[ModelMessage]:
    """Validate raw JSON payloads (one per message, in order) back into ``ModelMessage`` objects."""
    return ModelMessagesTypeAdapter.validate_python(list(payloads))


def message_content_hash(message: ModelMessage) -> str:
    """Stable content identity of one message.

    Content-based, never object identity: durable executors re-instantiate messages between
    steps and a re-fired save hook re-serializes the same history, so identity would re-append.
    """
    return hashlib.sha256(ModelMessagesTypeAdapter.dump_json([message])).hexdigest()


def strip_nuls(value: object) -> object:
    """Recursively remove NUL characters from strings; PostgreSQL jsonb rejects ``\\x00``."""
    if isinstance(value, str):
        return value.replace('\x00', '')
    if isinstance(value, list):
        return [strip_nuls(item) for item in cast('list[object]', value)]
    if isinstance(value, dict):
        return {key: strip_nuls(item) for key, item in cast('dict[object, object]', value).items()}
    return value


def sanitize_payload(
    payload: Mapping[str, object], *, dialect_name: str, custom: JsonSanitizer | None
) -> Mapping[str, object]:
    """Apply the host sanitizer, or the default NUL strip on PostgreSQL only."""
    if custom is not None:
        result = custom(payload)
    elif dialect_name == 'postgresql':
        result = strip_nuls(payload)
    else:
        return payload
    assert isinstance(result, dict)
    return cast('Mapping[str, object]', result)


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def extract_denorm(message: ModelMessage) -> MessageDenorm:
    """Extract the denormalized query columns from one message.

    Uses ``getattr`` with defaults for fields added across pydantic-ai releases so the
    package works on every supported version; this is the only compat shim location.
    """
    run_id = _str_or_none(getattr(message, 'run_id', None))
    conversation_id = _str_or_none(getattr(message, 'conversation_id', None))
    state_raw = getattr(message, 'state', None)
    state = state_raw if isinstance(state_raw, str) else None
    timestamp = getattr(message, 'timestamp', None)
    message_timestamp = timestamp if isinstance(timestamp, datetime) else None

    has_user_prompt = False
    tool_call_count = 0
    for part in message.parts:
        part_kind = getattr(part, 'part_kind', None)
        if part_kind == 'user-prompt':
            has_user_prompt = True
        elif part_kind in ('tool-call', 'builtin-tool-call'):
            tool_call_count += 1

    if isinstance(message, ModelResponse):
        usage = message.usage
        cost_raw = getattr(usage, 'cost', None)
        finish_reason = getattr(message, 'finish_reason', None)
        return MessageDenorm(
            kind='response',
            message_timestamp=message_timestamp,
            run_id=run_id,
            conversation_id=conversation_id,
            model_name=message.model_name,
            provider_name=getattr(message, 'provider_name', None),
            provider_response_id=_str_or_none(getattr(message, 'provider_response_id', None)),
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            state=state,
            input_tokens=_int_or_none(getattr(usage, 'input_tokens', None)),
            output_tokens=_int_or_none(getattr(usage, 'output_tokens', None)),
            cache_read_tokens=_int_or_none(getattr(usage, 'cache_read_tokens', None)),
            cache_write_tokens=_int_or_none(getattr(usage, 'cache_write_tokens', None)),
            cost=cost_raw if isinstance(cost_raw, Decimal) else None,
            has_user_prompt=has_user_prompt,
            tool_call_count=tool_call_count,
        )

    return MessageDenorm(
        kind='request',
        message_timestamp=message_timestamp,
        run_id=run_id,
        conversation_id=conversation_id,
        model_name=None,
        provider_name=None,
        provider_response_id=None,
        finish_reason=None,
        state=state,
        input_tokens=None,
        output_tokens=None,
        cache_read_tokens=None,
        cache_write_tokens=None,
        cost=None,
        has_user_prompt=has_user_prompt,
        tool_call_count=tool_call_count,
    )
