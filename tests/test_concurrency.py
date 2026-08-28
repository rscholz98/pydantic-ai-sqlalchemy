"""U2 tests for concurrent sequence allocation and collision retries.

Uses its own engine fixture: a file-backed SQLite database (BEGIN IMMEDIATE so concurrent
writers serialize cleanly instead of deadlocking) plus PostgreSQL when the CI DSN is set;
the shared in-memory fixture cannot host truly concurrent sessions.
"""

from __future__ import annotations

import asyncio
import pathlib
from collections.abc import AsyncIterator

import pytest
import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from pydantic_ai_sqlalchemy import Base, DefaultMessage, SQLAlchemyChatStore
from pydantic_ai_sqlalchemy import _write as write_module
from pydantic_ai_sqlalchemy._models import BaseMessage

from .conftest import PG_DSN
from .message_fixtures import sample_turn

CONCURRENCY_BACKENDS = ['sqlite', *(['postgres'] if PG_DSN else [])]


def _make_sqlite_file_engine(tmp_path: pathlib.Path) -> AsyncEngine:
    engine = create_async_engine(
        f'sqlite+aiosqlite:///{tmp_path}/concurrency.db', poolclass=NullPool, connect_args={'timeout': 30}
    )

    @event.listens_for(engine.sync_engine, 'connect')
    def _configure_sqlite(dbapi_connection: object, _record: object) -> None:
        # Disable the driver's implicit transaction handling so SQLAlchemy controls BEGIN;
        # this is the documented recipe that makes SAVEPOINTs reliable on (aio)sqlite.
        dbapi_connection.isolation_level = None  # type: ignore[attr-defined]
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute('PRAGMA foreign_keys=ON')
        cursor.close()

    @event.listens_for(engine.sync_engine, 'begin')
    def _begin_immediate(connection: sa.Connection) -> None:
        # Take the write lock up front: concurrent writers queue on it (honoring the busy
        # timeout) instead of hitting SQLITE_BUSY deadlocks on a read-to-write upgrade.
        connection.exec_driver_sql('BEGIN IMMEDIATE')

    return engine


@pytest.fixture(params=CONCURRENCY_BACKENDS)
async def concurrent_engine(request: pytest.FixtureRequest, tmp_path: pathlib.Path) -> AsyncIterator[AsyncEngine]:
    if request.param == 'sqlite':
        engine = _make_sqlite_file_engine(tmp_path)
    else:
        assert PG_DSN is not None
        engine = create_async_engine(PG_DSN)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def test_concurrent_saves_allocate_dense_unique_sequences(concurrent_engine: AsyncEngine) -> None:
    store = SQLAlchemyChatStore(concurrent_engine)
    writers = 20

    results = await asyncio.gather(
        *[
            store.save_messages(sample_turn(), conversation_key='conc', run_id=f'run-{index:02d}')
            for index in range(writers)
        ]
    )

    assert all(result.skipped == 0 and len(result.message_ids) == 2 for result in results)
    assert len({result.conversation_pk for result in results}) == 1

    maker = async_sessionmaker(concurrent_engine, expire_on_commit=False)
    async with maker() as session:
        seqs = (await session.scalars(sa.select(DefaultMessage.seq).order_by(DefaultMessage.seq))).all()
    assert list(seqs) == list(range(1, writers * 2 + 1))


async def test_seq_collision_retries_and_keeps_outer_transaction_alive(
    concurrent_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SQLAlchemyChatStore(concurrent_engine)
    maker = async_sessionmaker(concurrent_engine, expire_on_commit=False)

    async with maker() as caller:
        first = await store.save_messages(sample_turn(), conversation_key='collide', session=caller)
        assert len(first.message_ids) == 2  # seq 1 and 2

        # pre-insert a row occupying the next seq (3) inside the same caller transaction
        blocker = DefaultMessage(
            conversation_pk=first.conversation_pk,
            seq=3,
            kind='request',
            message={'kind': 'request', 'parts': []},
            content_hash='0' * 64,
        )
        caller.add(blocker)
        await caller.flush()

        real_next_seq = write_module._next_seq
        observed: list[int] = []

        async def stale_next_seq(db: AsyncSession, message_cls: type[BaseMessage], conversation_pk: object) -> int:
            value = await real_next_seq(db, message_cls, conversation_pk)  # type: ignore[arg-type]
            observed.append(value)
            if len(observed) == 1:
                return value - 1  # stale read: lands on the seq the blocker row occupies
            return value

        monkeypatch.setattr(write_module, '_next_seq', stale_next_seq)
        second = await store.save_messages(sample_turn(), conversation_key='collide', session=caller)

        assert len(observed) >= 2, 'the insert must have retried after the seq collision'
        assert second.skipped == 0 and len(second.message_ids) == 2

        # the retry moved past the occupied seq, and the savepoint kept the outer transaction alive
        seqs = (
            await caller.scalars(sa.select(DefaultMessage.seq).where(DefaultMessage.id.in_(second.message_ids)))
        ).all()
        assert sorted(seqs) == [4, 5]
        total = await caller.scalar(sa.select(sa.func.count()).select_from(DefaultMessage))
        assert total == 5  # blocker row survived the savepoint rollback
        await caller.commit()

    async with maker() as check:
        seqs = (await check.scalars(sa.select(DefaultMessage.seq).order_by(DefaultMessage.seq))).all()
    assert list(seqs) == [1, 2, 3, 4, 5]
