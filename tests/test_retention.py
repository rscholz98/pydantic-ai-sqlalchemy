"""U7 tests for retention: per-conversation deletion and cutoff-based purge."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from pydantic_ai_sqlalchemy import (
    ConversationNotFoundError,
    DefaultConversation,
    DefaultMessage,
    DefaultRun,
    DefaultToolCall,
    SQLAlchemyChatStore,
)
from pydantic_ai_sqlalchemy._serialize import dump_message, extract_denorm, message_content_hash

from .message_fixtures import text_response, user_request

OLD_ACTIVITY = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
NEW_ACTIVITY = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
CUTOFF = datetime(2026, 3, 1, tzinfo=timezone.utc)

TABLES = (DefaultConversation, DefaultMessage, DefaultRun, DefaultToolCall)


def _build_conversation_rows(
    key: str, last_activity_at: datetime | None
) -> tuple[DefaultConversation, list[DefaultMessage], DefaultRun, DefaultToolCall]:
    """Build one conversation with two messages, one run and one tool call, seeded directly via ORM."""
    conversation = DefaultConversation(
        id=uuid.uuid4(), conversation_key=key, message_count=2, last_activity_at=last_activity_at
    )
    run_id = f'{key}-run-1'
    messages: list[DefaultMessage] = []
    for seq, model_message in enumerate([user_request(f'Hello from {key}'), text_response(f'Hi, {key}!')]):
        denorm = extract_denorm(model_message)
        messages.append(
            DefaultMessage(
                id=uuid.uuid4(),
                conversation_pk=conversation.id,
                seq=seq,
                kind=denorm.kind,
                message=dump_message(model_message),
                content_hash=message_content_hash(model_message),
                run_id=run_id,
                message_timestamp=denorm.message_timestamp,
                model_name=denorm.model_name,
                input_tokens=denorm.input_tokens,
                output_tokens=denorm.output_tokens,
                has_user_prompt=denorm.has_user_prompt,
                tool_call_count=denorm.tool_call_count,
            )
        )
    run = DefaultRun(run_id=run_id, conversation_pk=conversation.id, message_count=2)
    tool_call = DefaultToolCall(
        id=uuid.uuid4(),
        message_pk=messages[1].id,
        conversation_pk=conversation.id,
        run_id=run_id,
        tool_call_id=f'{key}-call-1',
        tool_name='get_weather',
        status='returned',
    )
    return conversation, messages, run, tool_call


async def _seed(engine: AsyncEngine) -> dict[str, uuid.UUID]:
    """Seed three conversations: one old, one recent, one without any activity timestamp."""
    ids: dict[str, uuid.UUID] = {}
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        for key, last_activity_at in [('old', OLD_ACTIVITY), ('new', NEW_ACTIVITY), ('no-activity', None)]:
            conversation, messages, run, tool_call = _build_conversation_rows(key, last_activity_at)
            ids[key] = conversation.id
            session.add(conversation)
            await session.flush()
            session.add_all(messages)
            await session.flush()
            session.add_all([run, tool_call])
        await session.commit()
    return ids


async def _rows_for_conversation(engine: AsyncEngine, conversation_pk: uuid.UUID) -> dict[str, int]:
    maker = async_sessionmaker(engine)
    counts: dict[str, int] = {}
    async with maker() as session:
        counts['conversations'] = await _count(session, DefaultConversation, DefaultConversation.id == conversation_pk)
        counts['messages'] = await _count(session, DefaultMessage, DefaultMessage.conversation_pk == conversation_pk)
        counts['runs'] = await _count(session, DefaultRun, DefaultRun.conversation_pk == conversation_pk)
        counts['tool_calls'] = await _count(
            session, DefaultToolCall, DefaultToolCall.conversation_pk == conversation_pk
        )
    return counts


async def _count(session: AsyncSession, entity: type[object], criterion: sa.ColumnElement[bool]) -> int:
    value = await session.scalar(sa.select(sa.func.count()).select_from(entity).where(criterion))
    return value or 0


async def _total_rows(engine: AsyncEngine) -> dict[str, int]:
    maker = async_sessionmaker(engine)
    async with maker() as session:
        return {entity.__tablename__: await _count(session, entity, sa.true()) for entity in TABLES}


async def test_delete_conversation_removes_only_that_conversation(
    engine: AsyncEngine, store: SQLAlchemyChatStore
) -> None:
    ids = await _seed(engine)

    deleted_messages = await store.delete_conversation(conversation_key='old')

    assert deleted_messages == 2
    gone = await _rows_for_conversation(engine, ids['old'])
    assert gone == {'conversations': 0, 'messages': 0, 'runs': 0, 'tool_calls': 0}
    for survivor in ('new', 'no-activity'):
        kept = await _rows_for_conversation(engine, ids[survivor])
        assert kept == {'conversations': 1, 'messages': 2, 'runs': 1, 'tool_calls': 1}


async def test_delete_conversation_unknown_key_raises(engine: AsyncEngine, store: SQLAlchemyChatStore) -> None:
    await _seed(engine)
    with pytest.raises(ConversationNotFoundError):
        await store.delete_conversation(conversation_key='missing')
    totals = await _total_rows(engine)
    assert totals == {'pai_conversations': 3, 'pai_messages': 6, 'pai_runs': 3, 'pai_tool_calls': 3}


async def test_purge_dry_run_reports_counts_without_deleting(engine: AsyncEngine, store: SQLAlchemyChatStore) -> None:
    await _seed(engine)

    report = await store.purge_older_than(CUTOFF, dry_run=True)

    assert report.dry_run is True
    assert (report.conversations, report.messages, report.runs, report.tool_calls) == (1, 2, 1, 1)
    totals = await _total_rows(engine)
    assert totals == {'pai_conversations': 3, 'pai_messages': 6, 'pai_runs': 3, 'pai_tool_calls': 3}


async def test_purge_deletes_only_conversations_older_than_cutoff(
    engine: AsyncEngine, store: SQLAlchemyChatStore
) -> None:
    ids = await _seed(engine)

    report = await store.purge_older_than(CUTOFF)

    assert report.dry_run is False
    assert (report.conversations, report.messages, report.runs, report.tool_calls) == (1, 2, 1, 1)
    gone = await _rows_for_conversation(engine, ids['old'])
    assert gone == {'conversations': 0, 'messages': 0, 'runs': 0, 'tool_calls': 0}
    for survivor in ('new', 'no-activity'):
        kept = await _rows_for_conversation(engine, ids[survivor])
        assert kept == {'conversations': 1, 'messages': 2, 'runs': 1, 'tool_calls': 1}


async def test_purge_with_early_cutoff_matches_nothing(engine: AsyncEngine, store: SQLAlchemyChatStore) -> None:
    await _seed(engine)

    report = await store.purge_older_than(datetime(2020, 1, 1, tzinfo=timezone.utc))

    assert (report.conversations, report.messages, report.runs, report.tool_calls) == (0, 0, 0, 0)
    totals = await _total_rows(engine)
    assert totals == {'pai_conversations': 3, 'pai_messages': 6, 'pai_runs': 3, 'pai_tool_calls': 3}


async def test_caller_session_keeps_transaction_ownership(engine: AsyncEngine, store: SQLAlchemyChatStore) -> None:
    ids = await _seed(engine)

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as caller_session:
        deleted_messages = await store.delete_conversation(conversation_key='old', session=caller_session)
        assert deleted_messages == 2
        await caller_session.rollback()

    kept = await _rows_for_conversation(engine, ids['old'])
    assert kept == {'conversations': 1, 'messages': 2, 'runs': 1, 'tool_calls': 1}
