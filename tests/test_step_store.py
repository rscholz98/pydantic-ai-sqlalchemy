"""U13 tests for the pydantic-ai-harness ``StepStore`` backend."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import sqlalchemy as sa
from pydantic_ai.messages import ModelMessagesTypeAdapter
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from pydantic_ai_sqlalchemy import SQLAlchemyStepStore
from pydantic_ai_sqlalchemy._step_models import StepToolEffect

from .message_fixtures import binary_user_request, sample_conversation, text_response, user_request

sp = pytest.importorskip('pydantic_ai_harness.step_persistence')


def _as_utc(value: datetime) -> datetime:
    """SQLite drops tzinfo on round trip; stored values are UTC wall-clock either way."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _at(minute: int) -> datetime:
    return datetime(2026, 1, 1, 12, minute, 0, tzinfo=timezone.utc)


def test_protocol_conformance(engine: AsyncEngine) -> None:
    assert isinstance(SQLAlchemyStepStore(engine), sp.StepStore)


async def test_register_and_get_run(engine: AsyncEngine) -> None:
    store = SQLAlchemyStepStore(engine)
    record = sp.RunRecord(
        run_id='run-1',
        conversation_id='conv-1',
        parent_run_id='parent-1',
        agent_name='librarian',
        metadata={'attempt': '1'},
        started_at=_at(0),
    )
    await store.register_run(record)

    loaded = await store.get_run(run_id='run-1')
    assert loaded is not None
    assert loaded.run_id == 'run-1'
    assert loaded.conversation_id == 'conv-1'
    assert loaded.parent_run_id == 'parent-1'
    assert loaded.agent_name == 'librarian'
    assert loaded.metadata == {'attempt': '1'}
    assert _as_utc(loaded.started_at) == _at(0)

    assert await store.get_run(run_id='missing') is None


async def test_register_duplicate_run_id_raises(engine: AsyncEngine) -> None:
    store = SQLAlchemyStepStore(engine)
    await store.register_run(sp.RunRecord(run_id='run-dup'))
    with pytest.raises(IntegrityError):
        await store.register_run(sp.RunRecord(run_id='run-dup'))
    # The failed insert must not poison later calls on the same store.
    assert await store.get_run(run_id='run-dup') is not None


async def test_list_runs_sorting_and_filters(engine: AsyncEngine) -> None:
    store = SQLAlchemyStepStore(engine)
    # Registered out of chronological order to prove started_at ascending sorting.
    await store.register_run(
        sp.RunRecord(run_id='child-b', conversation_id='conv-2', parent_run_id='root', started_at=_at(3))
    )
    await store.register_run(sp.RunRecord(run_id='root', conversation_id='conv-1', started_at=_at(1)))
    await store.register_run(
        sp.RunRecord(run_id='child-a', conversation_id='conv-1', parent_run_id='root', started_at=_at(2))
    )

    assert [run.run_id for run in await store.list_runs()] == ['root', 'child-a', 'child-b']
    assert [run.run_id for run in await store.list_runs(parent_run_id='root')] == ['child-a', 'child-b']
    assert [run.run_id for run in await store.list_runs(conversation_id='conv-1')] == ['root', 'child-a']
    assert [run.run_id for run in await store.list_runs(parent_run_id='root', conversation_id='conv-1')] == ['child-a']
    assert await store.list_runs(parent_run_id='nope') == []


async def test_event_append_and_ordering_across_interleaved_runs(engine: AsyncEngine) -> None:
    store = SQLAlchemyStepStore(engine)
    # step_index resets across runs; ordering must come from the insert sequence.
    await store.append_event(sp.StepEvent(run_id='run-a', kind='run_started', step_index=0, metadata={'m': 'a0'}))
    await store.append_event(sp.StepEvent(run_id='run-b', kind='run_started', step_index=0))
    await store.append_event(
        sp.StepEvent(
            run_id='run-a',
            kind='tool_call_started',
            step_index=1,
            conversation_id='conv-1',
            parent_run_id='root',
            agent_name='librarian',
            tool_call_id='call_1',
            tool_name='get_weather',
        )
    )
    await store.append_event(sp.StepEvent(run_id='run-b', kind='run_failed', step_index=0, error='boom'))
    await store.append_event(sp.StepEvent(run_id='run-a', kind='run_completed', step_index=0))

    events_a = await store.list_events(run_id='run-a')
    assert [event.kind for event in events_a] == ['run_started', 'tool_call_started', 'run_completed']
    assert [event.step_index for event in events_a] == [0, 1, 0]
    assert events_a[0].metadata == {'m': 'a0'}
    assert events_a[1].conversation_id == 'conv-1'
    assert events_a[1].parent_run_id == 'root'
    assert events_a[1].agent_name == 'librarian'
    assert events_a[1].tool_call_id == 'call_1'
    assert events_a[1].tool_name == 'get_weather'

    events_b = await store.list_events(run_id='run-b')
    assert [event.kind for event in events_b] == ['run_started', 'run_failed']
    assert events_b[1].error == 'boom'
    assert await store.list_events(run_id='run-c') == []


async def test_snapshot_round_trip(engine: AsyncEngine) -> None:
    store = SQLAlchemyStepStore(engine)
    messages = [*sample_conversation(), binary_user_request()]
    await store.save_snapshot(
        sp.ContinuableSnapshot(
            run_id='run-1',
            step_index=2,
            messages=messages,
            conversation_id='conv-1',
            parent_run_id='root',
            agent_name='librarian',
            timestamp=_at(5),
        )
    )

    loaded = await store.latest_snapshot(run_id='run-1')
    assert loaded is not None
    assert loaded.run_id == 'run-1'
    assert loaded.step_index == 2
    assert loaded.conversation_id == 'conv-1'
    assert loaded.parent_run_id == 'root'
    assert loaded.agent_name == 'librarian'
    assert loaded.state == 'complete'
    assert _as_utc(loaded.timestamp) == _at(5)
    assert ModelMessagesTypeAdapter.dump_json(loaded.messages) == ModelMessagesTypeAdapter.dump_json(messages)


async def test_latest_snapshot_state_gating(engine: AsyncEngine) -> None:
    store = SQLAlchemyStepStore(engine)
    assert await store.latest_snapshot(run_id='run-1') is None

    await store.save_snapshot(sp.ContinuableSnapshot(run_id='run-1', step_index=1, messages=[user_request()]))
    await store.save_snapshot(
        sp.ContinuableSnapshot(
            run_id='run-1', step_index=2, messages=[user_request(), text_response()], state='interrupted'
        )
    )
    await store.save_snapshot(sp.ContinuableSnapshot(run_id='run-2', step_index=9, messages=[user_request()]))

    latest = await store.latest_snapshot(run_id='run-1')
    assert latest is not None
    assert latest.step_index == 1
    assert latest.state == 'complete'

    frontier = await store.latest_snapshot(run_id='run-1', include_interrupted=True)
    assert frontier is not None
    assert frontier.step_index == 2
    assert frontier.state == 'interrupted'

    interrupted_only = SQLAlchemyStepStore(engine)
    await interrupted_only.save_snapshot(
        sp.ContinuableSnapshot(run_id='run-3', step_index=1, messages=[user_request()], state='interrupted')
    )
    assert await interrupted_only.latest_snapshot(run_id='run-3') is None
    assert await interrupted_only.latest_snapshot(run_id='run-3', include_interrupted=True) is not None


async def test_tool_effect_upsert_lifecycle(engine: AsyncEngine) -> None:
    store = SQLAlchemyStepStore(engine)
    assert await store.get_tool_effect(run_id='run-1', tool_call_id='call_1') is None

    await store.record_tool_effect(
        sp.ToolEffectRecord(
            tool_call_id='call_1', tool_name='get_weather', run_id='run-1', status='started', started_at=_at(0)
        )
    )
    await store.record_tool_effect(
        sp.ToolEffectRecord(
            tool_call_id='call_2', tool_name='get_weather', run_id='run-1', status='started', started_at=_at(1)
        )
    )

    started = await store.get_tool_effect(run_id='run-1', tool_call_id='call_1')
    assert started is not None
    assert started.status == 'started'
    assert started.ended_at is None
    unresolved = await store.list_unresolved_tool_effects(run_id='run-1')
    assert [record.tool_call_id for record in unresolved] == ['call_1', 'call_2']

    await store.record_tool_effect(
        sp.ToolEffectRecord(
            tool_call_id='call_1',
            tool_name='get_weather',
            run_id='run-1',
            status='completed',
            started_at=_at(0),
            ended_at=_at(2),
            idempotency_key='key-1',
            effect_summary='wrote 1 row',
        )
    )

    completed = await store.get_tool_effect(run_id='run-1', tool_call_id='call_1')
    assert completed is not None
    assert completed.status == 'completed'
    assert completed.ended_at is not None
    assert _as_utc(completed.ended_at) == _at(2)
    assert completed.idempotency_key == 'key-1'
    assert completed.effect_summary == 'wrote 1 row'
    assert [record.tool_call_id for record in await store.list_unresolved_tool_effects(run_id='run-1')] == ['call_2']
    assert await store.list_unresolved_tool_effects(run_id='run-2') == []

    # The upsert updated in place; the composite key still maps to exactly one row.
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        count = await session.scalar(
            sa.select(sa.func.count())
            .select_from(StepToolEffect)
            .where(StepToolEffect.run_id == 'run-1', StepToolEffect.tool_call_id == 'call_1')
        )
    assert count == 1
