"""Shared fixtures: every store test runs on SQLite always, and on PostgreSQL when
``PAI_SQLA_TEST_PG_DSN`` is set (CI provides a postgres:16 service container)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import StaticPool

from pydantic_ai_sqlalchemy import SQLAlchemyChatStore
from pydantic_ai_sqlalchemy._base import Base

PG_DSN = os.environ.get('PAI_SQLA_TEST_PG_DSN')
BACKENDS = ['sqlite', *(['postgres'] if PG_DSN else [])]


def _make_engine(backend: str) -> AsyncEngine:
    if backend == 'sqlite':
        engine = create_async_engine('sqlite+aiosqlite://', poolclass=StaticPool)

        @event.listens_for(engine.sync_engine, 'connect')
        def _enable_sqlite_fks(dbapi_connection: object, _record: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            cursor.execute('PRAGMA foreign_keys=ON')
            cursor.close()

        return engine
    assert PG_DSN is not None
    return create_async_engine(PG_DSN)


@pytest.fixture(params=BACKENDS)
async def engine(request: pytest.FixtureRequest) -> AsyncIterator[AsyncEngine]:
    engine = _make_engine(request.param)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
def store(engine: AsyncEngine) -> SQLAlchemyChatStore:
    return SQLAlchemyChatStore(engine)
