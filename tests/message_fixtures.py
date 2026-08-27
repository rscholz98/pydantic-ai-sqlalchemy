"""Canonical ``ModelMessage`` builders covering the part kinds the store must round-trip.

Builders construct fresh objects on every call (stored messages must never share identity
with inputs) and stay compatible with every supported pydantic-ai version by only using
long-stable part types.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

PNG_BYTES = b'\x89PNG\r\n\x1a\n' + bytes(range(32))


def make_usage(*, input_tokens: int = 25, output_tokens: int = 10) -> RequestUsage:
    try:
        return RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens, cost=Decimal('0.00125'))
    except TypeError:  # older releases without the cost field
        return RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def user_request(text: str = 'Hello there') -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def system_user_request() -> ModelRequest:
    return ModelRequest(parts=[SystemPromptPart(content='You are terse.'), UserPromptPart(content='Hi')])


def binary_user_request() -> ModelRequest:
    return ModelRequest(
        parts=[
            UserPromptPart(content=['What is in this image?', BinaryContent(data=PNG_BYTES, media_type='image/png')])
        ]
    )


def image_url_request() -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=[ImageUrl(url='https://example.com/cat.png')])])


def text_response(content: str = 'Hi!', *, model_name: str = 'test-model') -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=content)], model_name=model_name, usage=make_usage())


def thinking_response() -> ModelResponse:
    return ModelResponse(
        parts=[ThinkingPart(content='pondering...'), TextPart(content='Done.')],
        model_name='test-model',
        usage=make_usage(),
    )


def tool_call_response(*, tool_call_id: str = 'call_1') -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(tool_name='get_weather', args={'city': 'Berlin'}, tool_call_id=tool_call_id)],
        model_name='test-model',
        usage=make_usage(),
    )


def tool_return_request(*, tool_call_id: str = 'call_1') -> ModelRequest:
    return ModelRequest(
        parts=[ToolReturnPart(tool_name='get_weather', content={'temperature': 21}, tool_call_id=tool_call_id)]
    )


def retry_request() -> ModelRequest:
    return ModelRequest(parts=[RetryPromptPart(content='Validation failed, try again')])


ALL_BUILDERS = [
    user_request,
    system_user_request,
    binary_user_request,
    image_url_request,
    text_response,
    thinking_response,
    tool_call_response,
    tool_return_request,
    retry_request,
]


def sample_turn() -> list[ModelMessage]:
    return [user_request(), text_response()]


def tool_turn() -> list[ModelMessage]:
    return [user_request('Weather in Berlin?'), tool_call_response(), tool_return_request(), text_response('21 C')]


def sample_conversation() -> list[ModelMessage]:
    return [*sample_turn(), *tool_turn(), user_request('Thanks'), text_response('Anytime.')]
