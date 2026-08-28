"""U3 tests for the pure turn-boundary utilities."""

from __future__ import annotations

import pytest
from pydantic_ai import messages as pai_messages
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)

from pydantic_ai_sqlalchemy._turns import messages_to_text, slice_to_recent_turns

from .message_fixtures import (
    binary_user_request,
    image_url_request,
    retry_request,
    sample_conversation,
    sample_turn,
    system_user_request,
    text_response,
    thinking_response,
    tool_call_response,
    tool_turn,
    user_request,
)


def mixed_return_request() -> ModelRequest:
    """Deferred/HITL resume request: a tool return arriving together with a fresh user prompt."""
    return ModelRequest(
        parts=[
            ToolReturnPart(tool_name='get_weather', content={'temperature': 21}, tool_call_id='call_1'),
            UserPromptPart(content='And tomorrow?'),
        ]
    )


def mixed_retry_request() -> ModelRequest:
    return ModelRequest(parts=[RetryPromptPart(content='invalid'), UserPromptPart(content='Try again please')])


# -- slice_to_recent_turns ---------------------------------------------------------------------


def test_slice_last_turn_of_sample_conversation() -> None:
    # sample_conversation: turn 1 (2 msgs), turn 2 = tool turn (4 msgs), turn 3 (2 msgs).
    messages = sample_conversation()
    result = slice_to_recent_turns(messages, max_turns=1)
    assert result == messages[6:]


def test_slice_keeps_tool_turn_whole() -> None:
    messages = sample_conversation()
    result = slice_to_recent_turns(messages, max_turns=2)
    # The window starts at the tool turn's user request; the tool-call response and the
    # tool-return request stay inside the window, never split.
    assert result == messages[2:]
    assert result[0].kind == 'request'
    assert result[1].parts[0].part_kind == 'tool-call'
    assert result[2].parts[0].part_kind == 'tool-return'


def test_slice_exact_turn_count_returns_all() -> None:
    messages = sample_conversation()
    assert slice_to_recent_turns(messages, max_turns=3) == list(messages)


def test_slice_oversized_max_turns_returns_all() -> None:
    messages = sample_conversation()
    assert slice_to_recent_turns(messages, max_turns=99) == list(messages)


def test_slice_zero_and_negative_return_empty() -> None:
    messages = sample_conversation()
    assert slice_to_recent_turns(messages, max_turns=0) == []
    assert slice_to_recent_turns(messages, max_turns=-1) == []


def test_slice_empty_input() -> None:
    assert slice_to_recent_turns([], max_turns=3) == []


def test_slice_exact_count_drops_leading_non_boundary_messages() -> None:
    # An orphan response before the first user prompt is not part of any of the last N turns.
    orphan = text_response('dangling')
    messages: list[ModelMessage] = [orphan, *sample_turn(), *tool_turn()]
    assert slice_to_recent_turns(messages, max_turns=2) == messages[1:]
    assert slice_to_recent_turns(messages, max_turns=3) == list(messages)


def test_slice_ignores_non_user_prompt_requests_as_boundaries() -> None:
    # Retry and tool-return requests never start a turn.
    messages: list[ModelMessage] = [user_request('a'), retry_request(), text_response('b')]
    assert slice_to_recent_turns(messages, max_turns=1) == list(messages)


def test_slice_mixed_tool_return_request_is_not_a_boundary() -> None:
    # A request carrying a tool return alongside a user prompt continues its turn; treating
    # it as a boundary would split the tool-call/tool-return pair.
    messages: list[ModelMessage] = [
        user_request('Weather?'),
        tool_call_response(),
        mixed_return_request(),
        text_response('21 C'),
    ]
    assert slice_to_recent_turns(messages, max_turns=1) == list(messages)


def test_slice_mixed_retry_request_is_not_a_boundary() -> None:
    messages: list[ModelMessage] = [user_request('a'), mixed_retry_request(), text_response('b')]
    assert slice_to_recent_turns(messages, max_turns=1) == list(messages)


def test_slice_is_pure_and_returns_a_new_list() -> None:
    messages = sample_conversation()
    snapshot = list(messages)
    result = slice_to_recent_turns(messages, max_turns=99)
    assert result is not messages
    assert messages == snapshot
    result_windowed = slice_to_recent_turns(messages, max_turns=1)
    assert result_windowed is not messages
    assert messages == snapshot


# -- messages_to_text --------------------------------------------------------------------------


def test_transcript_simple_turn() -> None:
    assert messages_to_text(sample_turn()) == 'User: Hello there\n\nAssistant: Hi!'


def test_transcript_skips_tool_parts() -> None:
    assert messages_to_text(tool_turn()) == 'User: Weather in Berlin?\n\nAssistant: 21 C'


def test_transcript_skips_system_and_thinking_parts() -> None:
    messages: list[ModelMessage] = [system_user_request(), thinking_response()]
    assert messages_to_text(messages) == 'User: Hi\n\nAssistant: Done.'


def test_transcript_skips_retry_requests_entirely() -> None:
    assert messages_to_text([retry_request()]) == ''


def test_transcript_sequence_content_keeps_text_items_only() -> None:
    assert messages_to_text([binary_user_request()]) == 'User: What is in this image?'


def test_transcript_image_only_prompt_gets_placeholder() -> None:
    assert messages_to_text([image_url_request()]) == 'User: [non-text content]'


def test_transcript_text_content_items_count_as_text() -> None:
    text_content_type = getattr(pai_messages, 'TextContent', None)
    if text_content_type is None:
        pytest.skip('TextContent is not available in this pydantic-ai version')
    request = ModelRequest(parts=[UserPromptPart(content=[text_content_type('typed text')])])
    assert messages_to_text([request]) == 'User: typed text'


def test_transcript_joins_multiple_text_items_in_one_prompt() -> None:
    request = ModelRequest(parts=[UserPromptPart(content=['first line', 'second line'])])
    assert messages_to_text([request]) == 'User: first line\nsecond line'


def test_transcript_joins_multiple_text_parts_in_one_response() -> None:
    response = ModelResponse(parts=[TextPart(content='part one'), TextPart(content='part two')], model_name='m')
    assert messages_to_text([response]) == 'Assistant: part one\npart two'


def test_transcript_empty_input() -> None:
    assert messages_to_text([]) == ''
