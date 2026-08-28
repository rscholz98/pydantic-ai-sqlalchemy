"""Analytics query tests: tables are seeded directly via the ORM and every aggregate
is asserted with exact numbers on each backend the ``engine`` fixture provides."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from pydantic_ai_sqlalchemy import SQLAlchemyChatStore
from pydantic_ai_sqlalchemy._models import DefaultConversation, DefaultMessage, DefaultRun, DefaultToolCall
from pydantic_ai_sqlalchemy._types import ConversationUsage, CostRow, DailyUsage, ModelUsage, RunStats, ToolUsage

UTC = timezone.utc

D1 = datetime(2026, 8, 24, tzinfo=UTC)
D2 = datetime(2026, 8, 25, tzinfo=UTC)
D3 = datetime(2026, 8, 26, tzinfo=UTC)

CONV_A_FIRST = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
CONV_A_LAST = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
CONV_B_FIRST = datetime(2026, 8, 25, 9, 0, tzinfo=UTC)
CONV_B_LAST = datetime(2026, 8, 25, 10, 0, tzinfo=UTC)

MSG_A2_TS = datetime(2026, 8, 24, 10, 0, 5, tzinfo=UTC)


def _utc(value: datetime | None) -> datetime | None:
    """SQLite hands back naive datetimes for timezone-aware columns; normalize for equality."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _message(
    conversation_pk: uuid.UUID,
    seq: int,
    *,
    kind: str = 'response',
    timestamp: datetime | None = None,
    model_name: str | None = None,
    provider_name: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    cost: Decimal | None = None,
    run_id: str | None = None,
    has_user_prompt: bool = False,
    tool_call_count: int = 0,
) -> DefaultMessage:
    return DefaultMessage(
        conversation_pk=conversation_pk,
        seq=seq,
        kind=kind,
        message={},
        content_hash=f'hash-{conversation_pk}-{seq}',
        run_id=run_id,
        message_timestamp=timestamp,
        model_name=model_name,
        provider_name=provider_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cost=cost,
        has_user_prompt=has_user_prompt,
        tool_call_count=tool_call_count,
    )


def _tool_call(
    message: DefaultMessage,
    *,
    tool_name: str,
    status: str,
    called_at: datetime | None,
    tool_call_id: str,
) -> DefaultToolCall:
    return DefaultToolCall(
        message_pk=message.id,
        conversation_pk=message.conversation_pk,
        run_id=message.run_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        called_at=called_at,
        status=status,
    )


@pytest.fixture
async def seeded(engine: AsyncEngine, store: SQLAlchemyChatStore) -> SQLAlchemyChatStore:
    """Two conversations across three days, five runs, five extracted tool calls.

    conv-a: request+response on day 1, request + three responses on day 2 (one of them the
    synthetic 'interrupted' marker), one response with a NULL timestamp, a request on day 3.
    conv-b: request + two responses on day 2 (one with NULL model_name).
    """
    maker = async_sessionmaker(engine, expire_on_commit=False)
    conv_a = DefaultConversation(
        conversation_key='conv-a',
        message_count=8,
        first_activity_at=CONV_A_FIRST,
        last_activity_at=CONV_A_LAST,
    )
    conv_b = DefaultConversation(
        conversation_key='conv-b',
        message_count=3,
        first_activity_at=CONV_B_FIRST,
        last_activity_at=CONV_B_LAST,
    )
    async with maker() as session:
        session.add_all([conv_a, conv_b])
        await session.flush()
        a2 = _message(
            conv_a.id,
            2,
            timestamp=MSG_A2_TS,
            model_name='gpt-5',
            provider_name='openai',
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=10,
            cost=Decimal('0.001'),
            run_id='run-1',
            tool_call_count=1,
        )
        a4 = _message(
            conv_a.id,
            4,
            timestamp=datetime(2026, 8, 25, 11, 0, 5, tzinfo=UTC),
            model_name='gpt-5',
            provider_name='openai',
            input_tokens=200,
            output_tokens=80,
            cost=Decimal('0.002'),
            run_id='run-2',
            tool_call_count=3,
        )
        b2 = _message(
            conv_b.id,
            2,
            timestamp=datetime(2026, 8, 25, 9, 0, 10, tzinfo=UTC),
            model_name='gpt-5',
            provider_name='openai',
            input_tokens=50,
            output_tokens=25,
            cache_read_tokens=5,
            cost=Decimal('0.0007'),
            run_id='run-3',
            tool_call_count=1,
        )
        messages = [
            _message(
                conv_a.id, 1, kind='request', timestamp=datetime(2026, 8, 24, 10, 0, tzinfo=UTC), has_user_prompt=True
            ),
            a2,
            _message(
                conv_a.id, 3, kind='request', timestamp=datetime(2026, 8, 25, 11, 0, tzinfo=UTC), has_user_prompt=True
            ),
            a4,
            _message(
                conv_a.id,
                5,
                timestamp=datetime(2026, 8, 25, 11, 30, tzinfo=UTC),
                model_name='claude-fable-5',
                provider_name='anthropic',
                input_tokens=300,
                output_tokens=120,
                cache_read_tokens=30,
                run_id='run-2',
            ),
            _message(
                conv_a.id,
                6,
                timestamp=datetime(2026, 8, 25, 11, 45, tzinfo=UTC),
                model_name='interrupted',
                input_tokens=999,
                output_tokens=999,
                cache_read_tokens=999,
                cost=Decimal('9.9'),
            ),
            _message(
                conv_a.id,
                7,
                timestamp=None,
                model_name='gpt-5',
                provider_name='openai',
                input_tokens=5,
                output_tokens=5,
                cost=Decimal('0.0005'),
            ),
            _message(
                conv_a.id, 8, kind='request', timestamp=datetime(2026, 8, 26, 12, 0, tzinfo=UTC), has_user_prompt=True
            ),
            _message(
                conv_b.id, 1, kind='request', timestamp=datetime(2026, 8, 25, 9, 0, tzinfo=UTC), has_user_prompt=True
            ),
            b2,
            _message(
                conv_b.id,
                3,
                timestamp=datetime(2026, 8, 25, 10, 0, tzinfo=UTC),
                input_tokens=10,
                output_tokens=5,
            ),
        ]
        session.add_all(messages)
        await session.flush()
        session.add_all(
            [
                _tool_call(
                    a2,
                    tool_name='search',
                    status='returned',
                    called_at=datetime(2026, 8, 24, 10, 0, 2, tzinfo=UTC),
                    tool_call_id='tc-1',
                ),
                _tool_call(
                    a4,
                    tool_name='search',
                    status='returned',
                    called_at=datetime(2026, 8, 25, 11, 0, 2, tzinfo=UTC),
                    tool_call_id='tc-2',
                ),
                _tool_call(
                    a4,
                    tool_name='search',
                    status='error',
                    called_at=datetime(2026, 8, 25, 11, 0, 3, tzinfo=UTC),
                    tool_call_id='tc-3',
                ),
                _tool_call(
                    a4,
                    tool_name='search',
                    status='called',
                    called_at=datetime(2026, 8, 25, 11, 0, 4, tzinfo=UTC),
                    tool_call_id='tc-4',
                ),
                _tool_call(b2, tool_name='calculator', status='unanswered', called_at=None, tool_call_id='tc-5'),
            ]
        )
        session.add_all(
            [
                DefaultRun(
                    run_id='run-1',
                    conversation_pk=conv_a.id,
                    state='completed',
                    started_at=datetime(2026, 8, 24, 10, 0, tzinfo=UTC),
                    duration_ms=1200,
                    input_tokens=100,
                    output_tokens=50,
                    cost=Decimal('0.001'),
                ),
                DefaultRun(
                    run_id='run-2',
                    conversation_pk=conv_a.id,
                    state='completed',
                    started_at=datetime(2026, 8, 25, 11, 0, tzinfo=UTC),
                    duration_ms=1800,
                    input_tokens=500,
                    output_tokens=200,
                    cost=Decimal('0.002'),
                ),
                DefaultRun(
                    run_id='run-3',
                    conversation_pk=conv_b.id,
                    state='error',
                    started_at=datetime(2026, 8, 25, 9, 0, tzinfo=UTC),
                    duration_ms=300,
                    input_tokens=50,
                    output_tokens=25,
                    cost=Decimal('0.0007'),
                ),
                DefaultRun(
                    run_id='run-4',
                    conversation_pk=conv_a.id,
                    state='running',
                    started_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
                ),
                DefaultRun(
                    run_id='run-5',
                    conversation_pk=conv_b.id,
                    state='error',
                    started_at=None,
                    input_tokens=10,
                    output_tokens=5,
                ),
            ]
        )
        await session.commit()
    return store


async def test_usage_by_day(seeded: SQLAlchemyChatStore) -> None:
    result = await seeded.usage_by_day()
    assert result == [
        DailyUsage(
            day=date(2026, 8, 24),
            model_requests=1,
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=10,
            cost=Decimal('0.001'),
            active_conversations=1,
        ),
        DailyUsage(
            day=date(2026, 8, 25),
            model_requests=4,
            input_tokens=560,
            output_tokens=230,
            cache_read_tokens=35,
            cost=Decimal('0.0027'),
            active_conversations=2,
        ),
        DailyUsage(
            day=date(2026, 8, 26),
            model_requests=0,
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cost=None,
            active_conversations=1,
        ),
    ]
    assert all(type(row.day) is date for row in result)


async def test_usage_by_day_bounds(seeded: SQLAlchemyChatStore) -> None:
    result = await seeded.usage_by_day(since=D2, until=D3)
    assert len(result) == 1
    assert result[0].day == date(2026, 8, 25)
    assert result[0].model_requests == 4
    assert await seeded.usage_by_day(since=datetime(2026, 8, 27, tzinfo=UTC)) == []


async def test_usage_by_model(seeded: SQLAlchemyChatStore) -> None:
    result = await seeded.usage_by_model()
    assert result[0] == ModelUsage(
        model_name='gpt-5',
        provider_name='openai',
        model_requests=4,
        input_tokens=355,
        output_tokens=160,
        cost=Decimal('0.0042'),
    )
    by_key = {(row.model_name, row.provider_name): row for row in result}
    assert by_key[('claude-fable-5', 'anthropic')] == ModelUsage(
        model_name='claude-fable-5',
        provider_name='anthropic',
        model_requests=1,
        input_tokens=300,
        output_tokens=120,
        cost=None,
    )
    assert by_key[(None, None)] == ModelUsage(
        model_name=None, provider_name=None, model_requests=1, input_tokens=10, output_tokens=5, cost=None
    )
    assert len(result) == 3


async def test_usage_by_model_boundaries(seeded: SQLAlchemyChatStore) -> None:
    inclusive = await seeded.usage_by_model(since=MSG_A2_TS, until=MSG_A2_TS + timedelta(seconds=1))
    assert inclusive == [
        ModelUsage(
            model_name='gpt-5',
            provider_name='openai',
            model_requests=1,
            input_tokens=100,
            output_tokens=50,
            cost=Decimal('0.001'),
        )
    ]
    assert await seeded.usage_by_model(since=MSG_A2_TS, until=MSG_A2_TS) == []

    # The NULL-timestamp response is counted unbounded (4 gpt-5 requests) but drops out of any window.
    bounded = await seeded.usage_by_model(since=D1)
    assert bounded[0] == ModelUsage(
        model_name='gpt-5',
        provider_name='openai',
        model_requests=3,
        input_tokens=350,
        output_tokens=155,
        cost=Decimal('0.0037'),
    )


async def test_usage_by_conversation(seeded: SQLAlchemyChatStore) -> None:
    result = await seeded.usage_by_conversation()
    normalized = [
        ConversationUsage(
            conversation_key=row.conversation_key,
            message_count=row.message_count,
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            cost=row.cost,
            first_activity_at=_utc(row.first_activity_at),
            last_activity_at=_utc(row.last_activity_at),
        )
        for row in result
    ]
    assert normalized == [
        ConversationUsage(
            conversation_key='conv-a',
            message_count=8,
            input_tokens=605,
            output_tokens=255,
            cost=Decimal('0.0035'),
            first_activity_at=CONV_A_FIRST,
            last_activity_at=CONV_A_LAST,
        ),
        ConversationUsage(
            conversation_key='conv-b',
            message_count=3,
            input_tokens=60,
            output_tokens=30,
            cost=Decimal('0.0007'),
            first_activity_at=CONV_B_FIRST,
            last_activity_at=CONV_B_LAST,
        ),
    ]


async def test_usage_by_conversation_limit_and_since(seeded: SQLAlchemyChatStore) -> None:
    top = await seeded.usage_by_conversation(limit=1)
    assert [row.conversation_key for row in top] == ['conv-a']

    result = await seeded.usage_by_conversation(since=D2)
    assert [row.conversation_key for row in result] == ['conv-a', 'conv-b']
    conv_a = result[0]
    assert conv_a.message_count == 5
    assert conv_a.input_tokens == 500
    assert conv_a.output_tokens == 200
    assert conv_a.cost == Decimal('0.002')


async def test_run_stats(seeded: SQLAlchemyChatStore) -> None:
    result = await seeded.run_stats()
    by_state = {row.state: row for row in result}
    assert by_state['completed'] == RunStats(
        state='completed',
        run_count=2,
        avg_duration_ms=1500.0,
        input_tokens=600,
        output_tokens=250,
        cost=Decimal('0.003'),
    )
    # run-5 has a NULL started_at: included in the unbounded call, its NULL duration ignored by AVG.
    assert by_state['error'] == RunStats(
        state='error', run_count=2, avg_duration_ms=300.0, input_tokens=60, output_tokens=30, cost=Decimal('0.0007')
    )
    assert by_state['running'] == RunStats(
        state='running', run_count=1, avg_duration_ms=None, input_tokens=0, output_tokens=0, cost=None
    )
    assert len(result) == 3

    since_d2 = {row.state: row for row in await seeded.run_stats(since=D2)}
    assert since_d2['completed'] == RunStats(
        state='completed',
        run_count=1,
        avg_duration_ms=1800.0,
        input_tokens=500,
        output_tokens=200,
        cost=Decimal('0.002'),
    )
    # The NULL started_at run is excluded as soon as a time bound is passed.
    assert since_d2['error'] == RunStats(
        state='error', run_count=1, avg_duration_ms=300.0, input_tokens=50, output_tokens=25, cost=Decimal('0.0007')
    )
    assert len(since_d2) == 3


async def test_tool_usage_grouped(seeded: SQLAlchemyChatStore) -> None:
    result = await seeded.tool_usage()
    assert result == [
        ToolUsage(tool_name='search', call_count=4, returned_count=2, error_count=1, unanswered_count=0),
        ToolUsage(tool_name='calculator', call_count=1, returned_count=0, error_count=0, unanswered_count=1),
    ]

    # NULL called_at rows (calculator) are excluded as soon as a time bound is passed.
    since_d2 = await seeded.tool_usage(since=D2)
    assert since_d2 == [
        ToolUsage(tool_name='search', call_count=3, returned_count=1, error_count=1, unanswered_count=0)
    ]

    # A populated tool_call table with an empty window returns [] and never falls back.
    assert await seeded.tool_usage(since=datetime(2026, 8, 27, tzinfo=UTC)) == []


async def test_tool_usage_fallback(engine: AsyncEngine, store: SQLAlchemyChatStore) -> None:
    maker = async_sessionmaker(engine, expire_on_commit=False)
    conv = DefaultConversation(conversation_key='conv-fallback', message_count=2)
    async with maker() as session:
        session.add(conv)
        await session.flush()
        session.add_all(
            [
                _message(
                    conv.id,
                    1,
                    timestamp=datetime(2026, 8, 24, 10, 0, tzinfo=UTC),
                    model_name='gpt-5',
                    tool_call_count=1,
                ),
                _message(
                    conv.id,
                    2,
                    timestamp=datetime(2026, 8, 25, 10, 0, tzinfo=UTC),
                    model_name='gpt-5',
                    tool_call_count=2,
                ),
            ]
        )
        await session.commit()

    assert await store.tool_usage() == [
        ToolUsage(tool_name='*', call_count=3, returned_count=0, error_count=0, unanswered_count=0)
    ]
    assert await store.tool_usage(since=D2) == [
        ToolUsage(tool_name='*', call_count=2, returned_count=0, error_count=0, unanswered_count=0)
    ]
    assert await store.tool_usage(since=datetime(2026, 8, 27, tzinfo=UTC)) == []


async def test_tool_usage_empty_store(store: SQLAlchemyChatStore) -> None:
    assert await store.tool_usage() == []


async def test_cost_report_by_day(seeded: SQLAlchemyChatStore) -> None:
    result = await seeded.cost_report(group_by='day')
    assert result == [
        CostRow(group='2026-08-25', model_requests=4, input_tokens=560, output_tokens=230, cost=Decimal('0.0027')),
        CostRow(group='2026-08-24', model_requests=1, input_tokens=100, output_tokens=50, cost=Decimal('0.001')),
    ]

    bounded = await seeded.cost_report(group_by='day', since=D2, until=D3)
    assert bounded == [
        CostRow(group='2026-08-25', model_requests=4, input_tokens=560, output_tokens=230, cost=Decimal('0.0027'))
    ]


async def test_cost_report_by_model(seeded: SQLAlchemyChatStore) -> None:
    result = await seeded.cost_report(group_by='model')
    assert result[0] == CostRow(
        group='gpt-5', model_requests=4, input_tokens=355, output_tokens=160, cost=Decimal('0.0042')
    )
    # None costs sort last; among them the order is unspecified.
    assert {row.group for row in result[1:]} == {'(unknown)', 'claude-fable-5'}
    by_group = {row.group: row for row in result}
    assert by_group['(unknown)'] == CostRow(
        group='(unknown)', model_requests=1, input_tokens=10, output_tokens=5, cost=None
    )
    assert by_group['claude-fable-5'] == CostRow(
        group='claude-fable-5', model_requests=1, input_tokens=300, output_tokens=120, cost=None
    )


async def test_cost_report_by_conversation(seeded: SQLAlchemyChatStore) -> None:
    result = await seeded.cost_report(group_by='conversation')
    assert result == [
        CostRow(group='conv-a', model_requests=4, input_tokens=605, output_tokens=255, cost=Decimal('0.0035')),
        CostRow(group='conv-b', model_requests=2, input_tokens=60, output_tokens=30, cost=Decimal('0.0007')),
    ]

    since_d2 = await seeded.cost_report(group_by='conversation', since=D2)
    assert since_d2 == [
        CostRow(group='conv-a', model_requests=2, input_tokens=500, output_tokens=200, cost=Decimal('0.002')),
        CostRow(group='conv-b', model_requests=2, input_tokens=60, output_tokens=30, cost=Decimal('0.0007')),
    ]
