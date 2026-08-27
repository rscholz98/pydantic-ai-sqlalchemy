"""U0 tests for the ORM schema and constraints."""

from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from pydantic_ai_sqlalchemy import DefaultConversation, DefaultMessage, SQLAlchemyChatStore


def _message(conversation_pk: uuid.UUID, seq: int) -> DefaultMessage:
    return DefaultMessage(
        conversation_pk=conversation_pk, seq=seq, kind='request', message={'kind': 'request'}, content_hash='0' * 64
    )


async def test_create_and_drop_tables_roundtrip() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import StaticPool

    engine = create_async_engine('sqlite+aiosqlite://', poolclass=StaticPool)
    store = SQLAlchemyChatStore(engine)
    await store.create_tables()
    async with engine.connect() as connection:
        names = await connection.run_sync(lambda c: sa.inspect(c).get_table_names())
    assert {'pai_conversations', 'pai_messages', 'pai_runs', 'pai_tool_calls'} <= set(names)
    await store.drop_tables()
    await engine.dispose()


async def test_unique_seq_per_conversation(engine: AsyncEngine) -> None:
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        conversation = DefaultConversation(conversation_key='c1')
        session.add(conversation)
        await session.flush()
        session.add(_message(conversation.id, 1))
        await session.flush()
        session.add(_message(conversation.id, 1))
        with pytest.raises(IntegrityError):
            await session.flush()


async def test_cascade_delete_removes_messages(engine: AsyncEngine) -> None:
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        conversation = DefaultConversation(conversation_key='c2')
        session.add(conversation)
        await session.flush()
        session.add(_message(conversation.id, 1))
        session.add(_message(conversation.id, 2))
        await session.commit()

    async with maker() as session:
        await session.execute(sa.delete(DefaultConversation).where(DefaultConversation.conversation_key == 'c2'))
        await session.commit()
        remaining = await session.scalar(sa.select(sa.func.count()).select_from(DefaultMessage))
        assert remaining == 0
