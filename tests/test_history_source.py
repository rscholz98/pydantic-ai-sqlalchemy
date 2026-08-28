"""U9 tests for the harness ``HistorySource`` seam: run listing and per-run history."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from pydantic_ai_sqlalchemy import SQLAlchemyChatStore
from pydantic_ai_sqlalchemy._history_source import list_runs, run_history
from pydantic_ai_sqlalchemy._models import DefaultConversation, DefaultMessage, DefaultRun
from pydantic_ai_sqlalchemy._serialize import dump_message, extract_denorm, message_content_hash

from .message_fixtures import binary_user_request, sample_turn, text_response, tool_turn, user_request

T0 = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 8, 1, 13, 0, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 8, 1, 14, 0, 0, tzinfo=timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """Normalize backend return values: SQLite yields naive UTC, PostgreSQL aware datetimes."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


async def _seed_conversation(
    session: AsyncSession, *, conversation_key: str, conversation_id: str | None = None
) -> uuid.UUID:
    conversation = DefaultConversation(conversation_key=conversation_key, conversation_id=conversation_id)
    session.add(conversation)
    await session.flush()
    return conversation.id


def _run_row(conversation_pk: uuid.UUID, run_id: str, **extra: object) -> DefaultRun:
    return DefaultRun(conversation_pk=conversation_pk, run_id=run_id, **extra)


def _message_row(conversation_pk: uuid.UUID, seq: int, message: ModelMessage, *, run_id: str | None) -> DefaultMessage:
    denorm = extract_denorm(message)
    return DefaultMessage(
        conversation_pk=conversation_pk,
        seq=seq,
        kind=denorm.kind,
        message=dump_message(message),
        content_hash=message_content_hash(message),
        run_id=run_id,
        message_timestamp=denorm.message_timestamp,
        model_name=denorm.model_name,
        input_tokens=denorm.input_tokens,
        output_tokens=denorm.output_tokens,
    )


def _maker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def test_list_runs_sorted_started_at_ascending_none_last(engine: AsyncEngine, store: SQLAlchemyChatStore) -> None:
    async with _maker(engine)() as session:
        conversation_pk = await _seed_conversation(session, conversation_key='conv-order')
        session.add(_run_row(conversation_pk, 'run-late', started_at=T1))
        session.add(_run_row(conversation_pk, 'run-none'))
        session.add(_run_row(conversation_pk, 'run-early', started_at=T0))
        session.add(_run_row(conversation_pk, 'run-latest', started_at=T2))
        await session.commit()

    records = await list_runs(store)
    assert [record.run_id for record in records] == ['run-early', 'run-late', 'run-latest', 'run-none']
    assert records[-1].started_at is None


async def test_list_runs_maps_all_fields(engine: AsyncEngine, store: SQLAlchemyChatStore) -> None:
    async with _maker(engine)() as session:
        conversation_pk = await _seed_conversation(session, conversation_key='conv-fields', conversation_id='conv-id-7')
        session.add(
            _run_row(
                conversation_pk,
                'run-full',
                parent_run_id='run-parent',
                agent_name='support_agent',
                started_at=T0,
                finished_at=T1,
                state='complete',
                model_name='gpt-test',
                input_tokens=120,
                output_tokens=34,
                cost=Decimal('0.005'),
                run_metadata={'channel': 'web'},
            )
        )
        await session.commit()

    (record,) = await list_runs(store)
    assert record.run_id == 'run-full'
    assert record.conversation_id == 'conv-id-7'
    assert record.parent_run_id == 'run-parent'
    assert record.agent_name == 'support_agent'
    assert record.metadata == {'channel': 'web'}
    assert record.started_at is not None and _as_utc(record.started_at) == T0
    assert record.finished_at is not None and _as_utc(record.finished_at) == T1
    assert record.state == 'complete'
    assert record.model_name == 'gpt-test'
    assert record.input_tokens == 120
    assert record.output_tokens == 34
    assert record.cost == Decimal('0.005')


async def test_list_runs_conversation_key_filter_and_id_fallback(
    engine: AsyncEngine, store: SQLAlchemyChatStore
) -> None:
    async with _maker(engine)() as session:
        with_id_pk = await _seed_conversation(session, conversation_key='conv-a', conversation_id='provider-conv-a')
        without_id_pk = await _seed_conversation(session, conversation_key='conv-b')
        session.add(_run_row(with_id_pk, 'run-a', started_at=T0))
        session.add(_run_row(without_id_pk, 'run-b', started_at=T1))
        await session.commit()

    all_records = await list_runs(store)
    assert [record.run_id for record in all_records] == ['run-a', 'run-b']
    assert all_records[0].conversation_id == 'provider-conv-a'
    assert all_records[1].conversation_id == 'conv-b'

    filtered = await list_runs(store, conversation_key='conv-b')
    assert [record.run_id for record in filtered] == ['run-b']

    assert await list_runs(store, conversation_key='conv-missing') == []


async def test_list_runs_metadata_coercion(engine: AsyncEngine, store: SQLAlchemyChatStore) -> None:
    async with _maker(engine)() as session:
        conversation_pk = await _seed_conversation(session, conversation_key='conv-meta')
        session.add(
            _run_row(
                conversation_pk,
                'run-mixed',
                started_at=T0,
                run_metadata={'retries': 2, 'cached': True, 'label': 'beta', 'ratio': 0.5},
            )
        )
        session.add(_run_row(conversation_pk, 'run-list-meta', started_at=T1, run_metadata=['not', 'a', 'dict']))
        session.add(_run_row(conversation_pk, 'run-null-meta', started_at=T2, run_metadata=None))
        await session.commit()

    records = {record.run_id: record for record in await list_runs(store)}
    assert records['run-mixed'].metadata == {'retries': '2', 'cached': 'True', 'label': 'beta', 'ratio': '0.5'}
    assert records['run-list-meta'].metadata == {}
    assert records['run-null-meta'].metadata == {}


async def test_run_history_returns_messages_in_seq_order(engine: AsyncEngine, store: SQLAlchemyChatStore) -> None:
    messages = [*tool_turn(), binary_user_request(), text_response('A cat.')]
    async with _maker(engine)() as session:
        conversation_pk = await _seed_conversation(session, conversation_key='conv-history')
        # Insert out of seq order to prove ordering comes from the query, not insertion.
        for seq in (3, 1, 5, 2, 4, 6):
            session.add(_message_row(conversation_pk, seq, messages[seq - 1], run_id='run-h'))
        session.add(_message_row(conversation_pk, 7, user_request('other run'), run_id='run-other'))
        session.add(_message_row(conversation_pk, 8, user_request('no run'), run_id=None))
        await session.commit()

    loaded = await run_history(store, run_id='run-h')
    assert len(loaded) == len(messages)
    assert ModelMessagesTypeAdapter.dump_json(loaded) == ModelMessagesTypeAdapter.dump_json(messages)


async def test_run_history_unknown_run_id_returns_empty(store: SQLAlchemyChatStore) -> None:
    assert await run_history(store, run_id='run-unknown') == []


async def test_store_satisfies_harness_history_source_protocol(engine: AsyncEngine, store: SQLAlchemyChatStore) -> None:
    harness_cs = pytest.importorskip('pydantic_ai_harness.conversation_search')
    assert isinstance(store, harness_cs.HistorySource)

    messages = sample_turn()
    async with _maker(engine)() as session:
        conversation_pk = await _seed_conversation(session, conversation_key='conv-facade')
        session.add(_run_row(conversation_pk, 'run-f', started_at=T0, run_metadata={'origin': 'test'}))
        for seq, message in enumerate(messages, start=1):
            session.add(_message_row(conversation_pk, seq, message, run_id='run-f'))
        await session.commit()

    records = await store.list_runs()
    assert [record.run_id for record in records] == ['run-f']
    assert records[0].metadata == {'origin': 'test'}

    loaded = await store.run_history(run_id='run-f')
    assert ModelMessagesTypeAdapter.dump_json(loaded) == ModelMessagesTypeAdapter.dump_json(messages)
    assert await store.run_history(run_id='nope') == []
