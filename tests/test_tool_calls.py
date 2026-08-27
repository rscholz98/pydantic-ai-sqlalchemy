"""Tests for tool-call extraction (U5): message rows are seeded directly (save paths are
other units) and ``extract_and_store`` is called with the aligned row/message lists."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from pydantic_ai_sqlalchemy import SQLAlchemyChatStore
from pydantic_ai_sqlalchemy._models import DefaultConversation, DefaultMessage, DefaultToolCall
from pydantic_ai_sqlalchemy._partial import UNANSWERED_TOOL_NOTE
from pydantic_ai_sqlalchemy._serialize import dump_message, extract_denorm, message_content_hash
from pydantic_ai_sqlalchemy._tool_calls import extract_and_store

from .message_fixtures import (
    make_usage,
    text_response,
    tool_call_response,
    tool_return_request,
    tool_turn,
    user_request,
)

# -- local builders (fixture module is frozen; extras live here) ----------------------------


def tool_error_retry_request(*, tool_call_id: str = 'call_1') -> ModelRequest:
    return ModelRequest(
        parts=[RetryPromptPart(content='Tool raised ModelRetry', tool_name='get_weather', tool_call_id=tool_call_id)]
    )


def unanswered_note_request(*, tool_call_id: str = 'call_1') -> ModelRequest:
    return ModelRequest(
        parts=[ToolReturnPart(tool_name='get_weather', content=UNANSWERED_TOOL_NOTE, tool_call_id=tool_call_id)]
    )


def response_with_args(args: object, *, tool_call_id: str = 'call_args') -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(tool_name='get_weather', args=args, tool_call_id=tool_call_id)],  # type: ignore[arg-type]
        model_name='test-model',
        usage=make_usage(),
    )


def native_tool_call_response(*, tool_call_id: str = 'native_1') -> ModelResponse:
    return ModelResponse(
        parts=[NativeToolCallPart(tool_name='web_search', args={'query': 'grills'}, tool_call_id=tool_call_id)],
        model_name='test-model',
        usage=make_usage(),
    )


# -- seeding helpers ------------------------------------------------------------------------


async def seed_conversation(session: AsyncSession) -> DefaultConversation:
    conversation = DefaultConversation(conversation_key=f'conv-{uuid.uuid4()}')
    session.add(conversation)
    await session.flush()
    return conversation


async def seed_message_rows(
    session: AsyncSession,
    conversation_pk: uuid.UUID,
    messages: Sequence[ModelMessage],
    *,
    run_id: str | None = 'run-1',
    start_seq: int = 0,
) -> list[DefaultMessage]:
    rows: list[DefaultMessage] = []
    for offset, message in enumerate(messages):
        denorm = extract_denorm(message)
        rows.append(
            DefaultMessage(
                conversation_pk=conversation_pk,
                seq=start_seq + offset,
                kind=denorm.kind,
                message=dump_message(message),
                content_hash=message_content_hash(message),
                run_id=run_id,
                message_timestamp=denorm.message_timestamp,
                model_name=denorm.model_name,
            )
        )
    session.add_all(rows)
    await session.flush()
    return rows


async def fetch_tool_calls(session: AsyncSession) -> list[DefaultToolCall]:
    result = await session.execute(sa.select(DefaultToolCall).order_by(DefaultToolCall.tool_call_id))
    return list(result.scalars())


def make_store(engine: AsyncEngine, *, extract_tool_calls: bool = True) -> SQLAlchemyChatStore:
    return SQLAlchemyChatStore(engine, extract_tool_calls=extract_tool_calls)


# -- tests ----------------------------------------------------------------------------------


async def test_in_batch_call_and_return_marked_returned(engine: AsyncEngine) -> None:
    store = make_store(engine)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session, session.begin():
        conversation = await seed_conversation(session)
        messages = tool_turn()
        rows = await seed_message_rows(session, conversation.id, messages)

        inserted = await extract_and_store(store, session, rows=rows, messages=messages)
        assert inserted == 1

        (record,) = await fetch_tool_calls(session)
        assert record.status == 'returned'
        assert record.tool_name == 'get_weather'
        assert record.tool_call_id == 'call_1'
        assert record.args == {'city': 'Berlin'}
        assert record.conversation_pk == conversation.id
        assert record.run_id == 'run-1'
        assert record.message_pk == rows[1].id
        assert record.called_at is not None
        called_at = record.called_at.replace(tzinfo=None)
        expected = rows[1].message_timestamp
        assert expected is not None
        assert called_at == expected.replace(tzinfo=None)


async def test_call_without_return_stays_called(engine: AsyncEngine) -> None:
    store = make_store(engine)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session, session.begin():
        conversation = await seed_conversation(session)
        messages: list[ModelMessage] = [user_request('Weather?'), tool_call_response()]
        rows = await seed_message_rows(session, conversation.id, messages)

        inserted = await extract_and_store(store, session, rows=rows, messages=messages)
        assert inserted == 1

        (record,) = await fetch_tool_calls(session)
        assert record.status == 'called'


async def test_cross_batch_return_flips_earlier_call(engine: AsyncEngine) -> None:
    store = make_store(engine)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session, session.begin():
        conversation = await seed_conversation(session)

        first_batch: list[ModelMessage] = [tool_call_response()]
        first_rows = await seed_message_rows(session, conversation.id, first_batch)
        assert await extract_and_store(store, session, rows=first_rows, messages=first_batch) == 1

        second_batch: list[ModelMessage] = [tool_return_request(), text_response('21 C')]
        second_rows = await seed_message_rows(session, conversation.id, second_batch, start_seq=1)
        assert await extract_and_store(store, session, rows=second_rows, messages=second_batch) == 0

        (record,) = await fetch_tool_calls(session)
        assert record.status == 'returned'
        assert record.message_pk == first_rows[0].id


async def test_retry_prompt_marks_error(engine: AsyncEngine) -> None:
    store = make_store(engine)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session, session.begin():
        conversation = await seed_conversation(session)
        messages: list[ModelMessage] = [tool_call_response(), tool_error_retry_request()]
        rows = await seed_message_rows(session, conversation.id, messages)

        assert await extract_and_store(store, session, rows=rows, messages=messages) == 1

        (record,) = await fetch_tool_calls(session)
        assert record.status == 'error'


async def test_plain_retry_prompt_does_not_answer(engine: AsyncEngine) -> None:
    store = make_store(engine)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session, session.begin():
        conversation = await seed_conversation(session)
        messages: list[ModelMessage] = [
            tool_call_response(),
            ModelRequest(parts=[RetryPromptPart(content='Validation failed, try again')]),
        ]
        rows = await seed_message_rows(session, conversation.id, messages)

        assert await extract_and_store(store, session, rows=rows, messages=messages) == 1

        (record,) = await fetch_tool_calls(session)
        assert record.status == 'called'


async def test_synthetic_note_marks_unanswered(engine: AsyncEngine) -> None:
    store = make_store(engine)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session, session.begin():
        conversation = await seed_conversation(session)
        messages: list[ModelMessage] = [tool_call_response(), unanswered_note_request()]
        rows = await seed_message_rows(session, conversation.id, messages)

        assert await extract_and_store(store, session, rows=rows, messages=messages) == 1

        (record,) = await fetch_tool_calls(session)
        assert record.status == 'unanswered'


async def test_args_variants(engine: AsyncEngine) -> None:
    store = make_store(engine)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session, session.begin():
        conversation = await seed_conversation(session)
        messages: list[ModelMessage] = [
            response_with_args({'city': 'Berlin'}, tool_call_id='args_dict'),
            response_with_args('{"city": "Berlin"}', tool_call_id='args_json'),
            response_with_args('not {valid json', tool_call_id='args_raw'),
            response_with_args(None, tool_call_id='args_none'),
        ]
        rows = await seed_message_rows(session, conversation.id, messages)

        assert await extract_and_store(store, session, rows=rows, messages=messages) == 4

        by_id = {record.tool_call_id: record for record in await fetch_tool_calls(session)}
        assert by_id['args_dict'].args == {'city': 'Berlin'}
        assert by_id['args_json'].args == {'city': 'Berlin'}
        assert by_id['args_raw'].args == {'raw': 'not {valid json'}
        assert by_id['args_none'].args is None


async def test_native_tool_call_extracted(engine: AsyncEngine) -> None:
    store = make_store(engine)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session, session.begin():
        conversation = await seed_conversation(session)
        messages: list[ModelMessage] = [native_tool_call_response()]
        rows = await seed_message_rows(session, conversation.id, messages)

        assert await extract_and_store(store, session, rows=rows, messages=messages) == 1

        (record,) = await fetch_tool_calls(session)
        assert record.tool_name == 'web_search'
        assert record.args == {'query': 'grills'}
        assert record.status == 'called'


async def test_disabled_store_writes_nothing(engine: AsyncEngine) -> None:
    store = make_store(engine, extract_tool_calls=False)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session, session.begin():
        conversation = await seed_conversation(session)
        messages = tool_turn()
        rows = await seed_message_rows(session, conversation.id, messages)

        assert await extract_and_store(store, session, rows=rows, messages=messages) == 0
        assert await fetch_tool_calls(session) == []
