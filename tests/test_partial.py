"""U4 tests for the partial-run utilities (interrupted turns, unanswered tool calls)."""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import pytest
from pydantic_ai import MessageHistoryMutatedWarning
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from pydantic_ai_sqlalchemy import DefaultMessage, SQLAlchemyChatStore
from pydantic_ai_sqlalchemy._partial import (
    INTERRUPTED_MODEL_NAME,
    INTERRUPTED_TEXT,
    UNANSWERED_TOOL_NOTE,
    close_unanswered_tool_calls,
    save_partial_run,
    synthesize_interrupted_response,
)

from .message_fixtures import text_response, tool_call_response, tool_return_request, tool_turn, user_request

try:
    from pydantic_ai.messages import NativeToolCallPart as BuiltinCallPart
except ImportError:  # older pydantic-ai releases use the previous class name
    from pydantic_ai.messages import BuiltinToolCallPart as BuiltinCallPart  # type: ignore[attr-defined,no-redef]


@pytest.fixture(autouse=True)
def _error_on_history_mutation() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter('error', MessageHistoryMutatedWarning)
        yield


def two_call_response(*, first_id: str, second_id: str) -> ModelResponse:
    return ModelResponse(
        parts=[
            ToolCallPart(tool_name='get_weather', args={'city': 'Berlin'}, tool_call_id=first_id),
            ToolCallPart(tool_name='get_time', args={'tz': 'Europe/Berlin'}, tool_call_id=second_id),
        ],
        model_name='test-model',
    )


def retry_for(tool_call_id: str) -> ModelRequest:
    return ModelRequest(
        parts=[
            RetryPromptPart(content='Validation failed, try again', tool_call_id=tool_call_id, tool_name='get_weather')
        ]
    )


# -- synthesize_interrupted_response --------------------------------------------------------


def test_synthesize_interrupted_response_defaults() -> None:
    response = synthesize_interrupted_response()
    assert isinstance(response, ModelResponse)
    assert response.model_name == INTERRUPTED_MODEL_NAME == 'interrupted'
    assert len(response.parts) == 1
    part = response.parts[0]
    assert part.part_kind == 'text'
    assert isinstance(part, TextPart)
    assert part.content == INTERRUPTED_TEXT


def test_synthesize_interrupted_response_configured() -> None:
    response = synthesize_interrupted_response(model_name='cancelled', text='Stopped early')
    assert response.model_name == 'cancelled'
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == 'Stopped early'


# -- close_unanswered_tool_calls ------------------------------------------------------------


def test_unanswered_call_gets_one_synthetic_return() -> None:
    messages: list[ModelMessage] = [user_request('Weather in Berlin?'), tool_call_response(tool_call_id='call_9')]
    before = ModelMessagesTypeAdapter.dump_json(messages)

    closed = close_unanswered_tool_calls(messages)

    assert ModelMessagesTypeAdapter.dump_json(messages) == before  # inputs unchanged byte-wise
    assert len(closed) == 3
    assert closed[0] is messages[0] and closed[1] is messages[1]  # same objects carried over
    closing = closed[2]
    assert isinstance(closing, ModelRequest)
    assert len(closing.parts) == 1
    part = closing.parts[0]
    assert isinstance(part, ToolReturnPart)
    assert part.tool_call_id == 'call_9'
    assert part.tool_name == 'get_weather'
    assert part.content == UNANSWERED_TOOL_NOTE


def test_custom_note_is_used() -> None:
    closed = close_unanswered_tool_calls([tool_call_response()], note='cancelled by user')
    closing = closed[-1]
    assert isinstance(closing, ModelRequest)
    assert isinstance(closing.parts[0], ToolReturnPart)
    assert closing.parts[0].content == 'cancelled by user'


def test_fully_answered_history_returned_unchanged() -> None:
    messages = tool_turn()
    before = ModelMessagesTypeAdapter.dump_json(messages)

    closed = close_unanswered_tool_calls(messages)

    assert ModelMessagesTypeAdapter.dump_json(messages) == before
    assert closed is not messages  # a new list
    assert closed == messages  # holding the very same objects
    assert all(result is original for result, original in zip(closed, messages, strict=True))


def test_history_without_tool_calls_returned_unchanged() -> None:
    messages: list[ModelMessage] = [
        user_request(),
        text_response(),
        user_request('Thanks'),
        text_response('Anytime.'),
    ]
    closed = close_unanswered_tool_calls(messages)
    assert closed == messages
    assert closed is not messages


def test_reused_tool_call_id_across_turns_closes_latest_call() -> None:
    messages: list[ModelMessage] = [
        user_request('Weather?'),
        tool_call_response(tool_call_id='call_1'),
        tool_return_request(tool_call_id='call_1'),
        text_response('21 C'),
        user_request('And tomorrow?'),
        tool_call_response(tool_call_id='call_1'),  # provider reused the id; still unanswered
    ]
    before = ModelMessagesTypeAdapter.dump_json(messages)

    closed = close_unanswered_tool_calls(messages)

    assert ModelMessagesTypeAdapter.dump_json(messages) == before
    assert len(closed) == 7
    closing = closed[-1]
    assert isinstance(closing, ModelRequest)
    assert len(closing.parts) == 1
    assert isinstance(closing.parts[0], ToolReturnPart)
    assert closing.parts[0].tool_call_id == 'call_1'
    assert closing.parts[0].content == UNANSWERED_TOOL_NOTE


def test_builtin_tool_calls_are_skipped() -> None:
    builtin_only: list[ModelMessage] = [
        user_request('Search the web'),
        ModelResponse(
            parts=[BuiltinCallPart(tool_name='web_search', args={'query': 'grills'}, tool_call_id='builtin_1')],
            model_name='test-model',
        ),
    ]
    closed = close_unanswered_tool_calls(builtin_only)
    assert closed == builtin_only  # never closed with a function ToolReturnPart
    assert closed is not builtin_only

    mixed: list[ModelMessage] = [
        user_request('Search and check weather'),
        ModelResponse(
            parts=[
                BuiltinCallPart(tool_name='web_search', args={'query': 'grills'}, tool_call_id='builtin_2'),
                ToolCallPart(tool_name='get_weather', args={'city': 'Berlin'}, tool_call_id='call_f'),
            ],
            model_name='test-model',
        ),
    ]
    closed = close_unanswered_tool_calls(mixed)
    assert len(closed) == 3
    closing = closed[-1]
    assert isinstance(closing, ModelRequest)
    assert [part.tool_call_id for part in closing.parts if isinstance(part, ToolReturnPart)] == ['call_f']


def test_multiple_unanswered_calls_across_messages_all_closed() -> None:
    messages: list[ModelMessage] = [
        user_request('Weather and time please'),
        two_call_response(first_id='call_a', second_id='call_b'),
        ModelRequest(
            parts=[ToolReturnPart(tool_name='get_weather', content={'temperature': 21}, tool_call_id='call_a')]
        ),
        tool_call_response(tool_call_id='call_c'),
    ]
    before = ModelMessagesTypeAdapter.dump_json(messages)

    closed = close_unanswered_tool_calls(messages)

    assert ModelMessagesTypeAdapter.dump_json(messages) == before
    assert len(closed) == 5
    closing = closed[-1]
    assert isinstance(closing, ModelRequest)
    returns = [part for part in closing.parts if isinstance(part, ToolReturnPart)]
    assert [(part.tool_call_id, part.tool_name) for part in returns] == [
        ('call_b', 'get_time'),
        ('call_c', 'get_weather'),
    ]
    assert all(part.content == UNANSWERED_TOOL_NOTE for part in returns)


def test_retry_prompt_counts_as_answered() -> None:
    messages: list[ModelMessage] = [
        user_request('Weather?'),
        tool_call_response(tool_call_id='call_retry'),
        retry_for('call_retry'),
    ]
    before = ModelMessagesTypeAdapter.dump_json(messages)

    closed = close_unanswered_tool_calls(messages)

    assert ModelMessagesTypeAdapter.dump_json(messages) == before
    assert closed == messages
    assert closed is not messages


# -- save_partial_run -----------------------------------------------------------------------


async def test_save_partial_run_persists_closed_and_marked_history(
    store: SQLAlchemyChatStore, engine: AsyncEngine
) -> None:
    messages: list[ModelMessage] = [user_request('Weather in Berlin?'), tool_call_response(tool_call_id='call_p')]
    before = ModelMessagesTypeAdapter.dump_json(messages)

    try:
        result = await save_partial_run(store, messages=messages, conversation_key='conv-partial', run_id='run-p')
    except NotImplementedError:
        pytest.xfail('depends on unit U2')

    assert ModelMessagesTypeAdapter.dump_json(messages) == before  # inputs unchanged
    assert result.conversation_key == 'conv-partial'
    assert result.run_id == 'run-p'
    # two originals + one closing request + one interrupted response
    assert len(result.message_ids) == 4

    async with AsyncSession(engine) as session:
        rows = (await session.execute(select(DefaultMessage).order_by(DefaultMessage.seq))).scalars().all()
    assert len(rows) == 4
    persisted = ModelMessagesTypeAdapter.validate_python([row.message for row in rows])

    closing = persisted[2]
    assert isinstance(closing, ModelRequest)
    assert isinstance(closing.parts[0], ToolReturnPart)
    assert closing.parts[0].tool_call_id == 'call_p'
    assert closing.parts[0].content == UNANSWERED_TOOL_NOTE

    interrupted = persisted[3]
    assert isinstance(interrupted, ModelResponse)
    assert interrupted.model_name == INTERRUPTED_MODEL_NAME
    assert isinstance(interrupted.parts[0], TextPart)
    assert interrupted.parts[0].content == INTERRUPTED_TEXT
    assert rows[3].model_name == INTERRUPTED_MODEL_NAME  # denormalized column lets analytics filter


async def test_save_partial_run_flags_disable_transforms(store: SQLAlchemyChatStore, engine: AsyncEngine) -> None:
    messages: list[ModelMessage] = [user_request('Hi'), tool_call_response(tool_call_id='call_q')]

    try:
        result = await save_partial_run(
            store,
            messages=messages,
            conversation_key='conv-raw',
            mark_interrupted=False,
            close_deferred_tools=False,
        )
    except NotImplementedError:
        pytest.xfail('depends on unit U2')

    assert len(result.message_ids) == 2
    async with AsyncSession(engine) as session:
        rows = (await session.execute(select(DefaultMessage).order_by(DefaultMessage.seq))).scalars().all()
    assert len(rows) == 2
    persisted = ModelMessagesTypeAdapter.validate_python([row.message for row in rows])
    assert ModelMessagesTypeAdapter.dump_json(persisted) == ModelMessagesTypeAdapter.dump_json(messages)


async def test_save_partial_run_empty_messages_persists_nothing(
    store: SQLAlchemyChatStore, engine: AsyncEngine
) -> None:
    try:
        result = await save_partial_run(store, messages=[], conversation_key='conv-empty')
    except NotImplementedError:
        pytest.xfail('depends on unit U2')

    assert result.message_ids == ()  # no lone synthetic response for an empty run
    async with AsyncSession(engine) as session:
        rows = (await session.execute(select(DefaultMessage))).scalars().all()
    assert rows == []
