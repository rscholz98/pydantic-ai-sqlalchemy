"""U2 tests for the write path: saving batches and runs, idempotency, rollups."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any, cast

import pytest
import sqlalchemy as sa
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.run import AgentRunResult
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from pydantic_ai_sqlalchemy import (
    DefaultConversation,
    DefaultMessage,
    DefaultRun,
    SQLAlchemyChatStore,
    StoreError,
)

from .message_fixtures import sample_conversation, sample_turn, tool_turn, user_request


async def _fetch_rows(store: SQLAlchemyChatStore, conversation_key: str) -> list[DefaultMessage]:
    async with store.sessions.scope() as (session, _owned):
        rows = await session.scalars(
            sa.select(DefaultMessage)
            .join(DefaultConversation, DefaultMessage.conversation_pk == DefaultConversation.id)
            .where(DefaultConversation.conversation_key == conversation_key)
            .order_by(DefaultMessage.seq)
        )
        return list(rows.all())


async def _fetch_conversation(store: SQLAlchemyChatStore, conversation_key: str) -> DefaultConversation:
    async with store.sessions.scope() as (session, _owned):
        conversation = await session.scalar(
            sa.select(DefaultConversation).where(DefaultConversation.conversation_key == conversation_key)
        )
        assert conversation is not None
        return conversation


async def test_save_messages_persists_rows_denorm_and_rollups(store: SQLAlchemyChatStore) -> None:
    messages = sample_conversation()
    result = await store.save_messages(messages, conversation_key='conv-basic')

    assert result.conversation_key == 'conv-basic'
    assert result.run_id is None
    assert result.skipped == 0
    assert len(result.message_ids) == len(messages) == 8

    rows = await _fetch_rows(store, 'conv-basic')
    assert [row.seq for row in rows] == list(range(1, 9))
    assert [row.id for row in rows] == list(result.message_ids)
    assert all(row.conversation_pk == result.conversation_pk for row in rows)
    assert [row.kind for row in rows] == [
        'request',
        'response',
        'request',
        'response',
        'request',
        'response',
        'request',
        'response',
    ]

    first = rows[0]
    assert first.has_user_prompt is True
    assert first.model_name is None
    assert first.content_hash and len(first.content_hash) == 64
    assert isinstance(first.message, dict) and first.message['kind'] == 'request'

    response = rows[1]
    assert response.model_name == 'test-model'
    assert response.input_tokens == 25
    assert response.output_tokens == 10
    assert response.cost == Decimal('0.00125')
    assert response.message_timestamp is not None

    tool_call_row = rows[3]
    assert tool_call_row.tool_call_count == 1

    conversation = await _fetch_conversation(store, 'conv-basic')
    assert conversation.message_count == 8
    assert conversation.total_input_tokens == 4 * 25
    assert conversation.total_output_tokens == 4 * 10
    assert conversation.total_cost == Decimal('0.005')
    assert conversation.first_activity_at is not None
    assert conversation.last_activity_at is not None


async def test_double_save_with_same_run_id_writes_nothing(store: SQLAlchemyChatStore) -> None:
    messages = sample_turn()
    first = await store.save_messages(messages, conversation_key='conv-idem', run_id='run-1')
    assert first.skipped == 0 and len(first.message_ids) == 2

    second = await store.save_messages(messages, conversation_key='conv-idem', run_id='run-1')
    assert second.skipped == 2
    assert second.message_ids == ()
    assert second.conversation_pk == first.conversation_pk

    rows = await _fetch_rows(store, 'conv-idem')
    assert len(rows) == 2
    conversation = await _fetch_conversation(store, 'conv-idem')
    assert conversation.message_count == 2


async def test_growing_history_skips_only_the_persisted_prefix(store: SQLAlchemyChatStore) -> None:
    turn_one = sample_turn()
    await store.save_messages(turn_one, conversation_key='conv-grow', run_id='run-g')

    extended = [*turn_one, *tool_turn()]
    result = await store.save_messages(extended, conversation_key='conv-grow', run_id='run-g')
    assert result.skipped == 2
    assert len(result.message_ids) == 4

    rows = await _fetch_rows(store, 'conv-grow')
    assert [row.seq for row in rows] == list(range(1, 7))


async def test_final_message_id_forces_last_response_primary_key(store: SQLAlchemyChatStore) -> None:
    forced = uuid.uuid4()
    messages = [*sample_turn(), user_request('and thanks')]
    result = await store.save_messages(messages, conversation_key='conv-final', final_message_id=forced)

    rows = await _fetch_rows(store, 'conv-final')
    response_rows = [row for row in rows if row.kind == 'response']
    assert response_rows[-1].id == forced
    assert forced in result.message_ids
    # the trailing request keeps its own generated id
    assert rows[-1].kind == 'request' and rows[-1].id != forced


def _clock_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart(tool_name='get_time', args={}, tool_call_id='call_time_1')])
    return ModelResponse(parts=[TextPart(content='It is 12:00.')])


def _make_clock_agent() -> Agent[None, str]:
    agent: Agent[None, str] = Agent(FunctionModel(_clock_model))

    @agent.tool_plain
    def get_time() -> str:
        return '12:00'

    return agent


async def test_save_run_with_function_model_and_tool(store: SQLAlchemyChatStore) -> None:
    agent = _make_clock_agent()
    run_result = await agent.run('What time is it?')

    saved = await store.save_run(run_result, conversation_key='conv-run', agent_name='clock-agent')
    assert saved.run_id == run_result.run_id
    assert saved.skipped == 0
    # request, tool-call response, tool-return request, final response
    assert len(saved.message_ids) == 4

    rows = await _fetch_rows(store, 'conv-run')
    assert [row.kind for row in rows] == ['request', 'response', 'request', 'response']
    assert all(row.run_id == run_result.run_id for row in rows)

    async with store.sessions.scope() as (session, _owned):
        run = await session.get(DefaultRun, run_result.run_id)
        assert run is not None
        assert run.conversation_pk == saved.conversation_pk
        assert run.agent_name == 'clock-agent'
        assert run.message_count == 4
        assert run.model_request_count == 2
        assert run.tool_call_count == 1
        assert run.model_name is not None
        assert run.input_tokens > 0
        assert run.finished_at is not None
        if run.started_at is not None:
            assert run.duration_ms is not None and run.duration_ms >= 0

    # re-firing the identical save writes nothing
    again = await store.save_run(run_result, conversation_key='conv-run')
    assert again.skipped == 4
    assert again.message_ids == ()
    assert len(await _fetch_rows(store, 'conv-run')) == 4


async def test_save_run_defaults_conversation_key_to_conversation_id(store: SQLAlchemyChatStore) -> None:
    agent = _make_clock_agent()
    run_result = await agent.run('What time is it?')

    saved = await store.save_run(run_result)
    assert saved.conversation_key == run_result.conversation_id
    conversation = await _fetch_conversation(store, run_result.conversation_id)
    assert conversation.conversation_id == run_result.conversation_id
    assert conversation.message_count == 4


async def test_save_run_without_resolvable_key_raises(store: SQLAlchemyChatStore) -> None:
    class _KeylessResult:
        def new_messages(self) -> list[ModelMessage]:
            return sample_turn()

        def all_messages(self) -> list[ModelMessage]:
            return sample_turn()

    with pytest.raises(StoreError):
        await store.save_run(cast('AgentRunResult[Any]', _KeylessResult()))


async def test_caller_session_is_flushed_but_never_committed(store: SQLAlchemyChatStore, engine: AsyncEngine) -> None:
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as caller:
        result = await store.save_messages(sample_turn(), conversation_key='conv-caller', session=caller)
        assert len(result.message_ids) == 2
        # the store flushed: rows are visible inside the caller's transaction
        visible = await caller.scalar(sa.select(sa.func.count()).select_from(DefaultMessage))
        assert visible == 2
        await caller.rollback()

    async with maker() as check:
        message_count = await check.scalar(sa.select(sa.func.count()).select_from(DefaultMessage))
        conversation_count = await check.scalar(sa.select(sa.func.count()).select_from(DefaultConversation))
    assert message_count == 0
    assert conversation_count == 0
