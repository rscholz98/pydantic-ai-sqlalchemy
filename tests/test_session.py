"""U0 tests for session provisioning and commit ownership."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from pydantic_ai_sqlalchemy import DefaultConversation, StoreError
from pydantic_ai_sqlalchemy._session import SessionProvider


async def test_managed_session_commits(engine: AsyncEngine) -> None:
    provider = SessionProvider(engine)
    async with provider.scope() as (session, owned):
        assert owned is True
        session.add(DefaultConversation(conversation_key='managed'))

    maker = async_sessionmaker(engine)
    async with maker() as check:
        count = await check.scalar(
            sa.select(sa.func.count())
            .select_from(DefaultConversation)
            .where(DefaultConversation.conversation_key == 'managed')
        )
    assert count == 1


async def test_caller_session_is_never_committed(engine: AsyncEngine) -> None:
    provider = SessionProvider(None)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as caller_session:
        async with provider.scope(caller_session) as (session, owned):
            assert session is caller_session
            assert owned is False
            session.add(DefaultConversation(conversation_key='caller'))
            await session.flush()
        await caller_session.rollback()

    async with maker() as check:
        count = await check.scalar(
            sa.select(sa.func.count())
            .select_from(DefaultConversation)
            .where(DefaultConversation.conversation_key == 'caller')
        )
    assert count == 0


async def test_no_bind_and_no_session_raises() -> None:
    provider = SessionProvider(None)
    with pytest.raises(StoreError):
        async with provider.scope():
            pass


async def test_sessionmaker_bind_accepted(engine: AsyncEngine) -> None:
    maker = async_sessionmaker(engine, expire_on_commit=False)
    provider = SessionProvider(maker)
    assert provider.has_bind
    async with provider.scope() as (_session, owned):
        assert owned is True


def test_invalid_bind_rejected() -> None:
    with pytest.raises(StoreError):
        SessionProvider(object())  # type: ignore[arg-type]
