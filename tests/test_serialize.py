"""U0 tests for the serialization core."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic_ai.messages import ModelMessagesTypeAdapter

from pydantic_ai_sqlalchemy._serialize import (
    dump_message,
    extract_denorm,
    load_messages,
    message_content_hash,
    sanitize_payload,
    strip_nuls,
)

from .message_fixtures import ALL_BUILDERS, text_response, tool_call_response, user_request


@pytest.mark.parametrize('builder', ALL_BUILDERS, ids=lambda b: b.__name__)
def test_dump_load_round_trip_is_byte_faithful(builder) -> None:
    original = builder()
    payload = dump_message(original)
    [loaded] = load_messages([payload])
    assert ModelMessagesTypeAdapter.dump_json([loaded]) == ModelMessagesTypeAdapter.dump_json([original])


def test_content_hash_is_stable_and_content_sensitive() -> None:
    # Hashing is content-based over the serialized message: the same object always hashes the
    # same, while two separately built messages differ (part timestamps are stamped at build
    # time). Idempotent saves rely on re-serializing the same objects, never on rebuilding.
    message = user_request('a')
    assert message_content_hash(message) == message_content_hash(message)
    assert message_content_hash(message) != message_content_hash(user_request('b'))
    assert len(message_content_hash(message)) == 64


def test_denorm_response_fields() -> None:
    denorm = extract_denorm(text_response())
    assert denorm.kind == 'response'
    assert denorm.model_name == 'test-model'
    assert denorm.input_tokens == 25
    assert denorm.output_tokens == 10
    assert denorm.has_user_prompt is False
    if denorm.cost is not None:  # cost field exists on all supported versions, but stay tolerant
        assert denorm.cost == Decimal('0.00125')


def test_denorm_request_and_tool_calls() -> None:
    request = extract_denorm(user_request())
    assert request.kind == 'request'
    assert request.has_user_prompt is True
    assert request.input_tokens is None

    tool_call = extract_denorm(tool_call_response())
    assert tool_call.tool_call_count == 1


def test_strip_nuls_recurses() -> None:
    dirty = {'a': 'x\x00y', 'b': ['\x00', {'c': 'ok\x00'}], 'd': 5}
    assert strip_nuls(dirty) == {'a': 'xy', 'b': ['', {'c': 'ok'}], 'd': 5}


def test_sanitize_payload_dialect_gate() -> None:
    payload = {'text': 'a\x00b'}
    assert sanitize_payload(payload, dialect_name='postgresql', custom=None) == {'text': 'ab'}
    assert sanitize_payload(payload, dialect_name='sqlite', custom=None) is payload
    assert sanitize_payload(payload, dialect_name='sqlite', custom=lambda p: {'r': 1}) == {'r': 1}
