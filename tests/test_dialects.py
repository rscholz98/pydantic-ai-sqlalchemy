"""U11 tests for cross-dialect behavior: DDL variants, the sanitize gate and value round-trips.

The JSON columns must compile to JSONB on PostgreSQL and plain JSON elsewhere, the
autoincrement PK variant must degrade from BIGINT to INTEGER on SQLite, and the NUL
sanitize gate must fire only for PostgreSQL (jsonb rejects ``\\x00``; SQLite does not).
"""

from __future__ import annotations

import os
import uuid
from decimal import Decimal
from typing import cast

import pytest
import sqlalchemy as sa
from pydantic_ai.messages import ModelMessagesTypeAdapter
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlalchemy.schema import CreateTable

from pydantic_ai_sqlalchemy._models import DefaultConversation, DefaultMessage
from pydantic_ai_sqlalchemy._serialize import dump_message, load_messages, message_content_hash, sanitize_payload
from pydantic_ai_sqlalchemy._step_models import StepEvent, StepSnapshot

from .message_fixtures import text_response


def _create_table_sql(table: object, dialect: sa.Dialect) -> str:
    return str(CreateTable(cast('sa.Table', table)).compile(dialect=dialect))


# --- DDL compilation (no server required) ---


def test_message_table_compiles_to_jsonb_and_numeric_on_postgres() -> None:
    ddl = _create_table_sql(DefaultMessage.__table__, postgresql.dialect())
    assert 'JSONB' in ddl
    assert 'NUMERIC(18, 8)' in ddl


def test_message_table_compiles_to_plain_json_on_sqlite() -> None:
    ddl = _create_table_sql(DefaultMessage.__table__, sqlite.dialect())
    assert 'JSONB' not in ddl
    assert 'JSON' in ddl


def test_step_snapshot_table_compiles_to_jsonb_on_postgres() -> None:
    ddl = _create_table_sql(StepSnapshot.__table__, postgresql.dialect())
    assert 'JSONB' in ddl


def test_step_event_seq_autoincrement_variant() -> None:
    seq_type = StepEvent.__table__.c.seq.type
    assert seq_type.compile(dialect=postgresql.dialect()) == 'BIGINT'
    assert seq_type.compile(dialect=sqlite.dialect()) == 'INTEGER'


# --- sanitize gate ---


def test_sanitize_strips_nul_for_postgresql() -> None:
    payload = {'text': 'a\x00b', 'nested': {'items': ['x\x00', 'y']}}
    result = sanitize_payload(payload, dialect_name='postgresql', custom=None)
    assert result == {'text': 'ab', 'nested': {'items': ['x', 'y']}}


def test_sanitize_passes_through_unchanged_for_sqlite() -> None:
    payload = {'text': 'a\x00b'}
    result = sanitize_payload(payload, dialect_name='sqlite', custom=None)
    assert result is payload


def test_custom_sanitizer_wins_for_any_dialect() -> None:
    def custom(payload: object) -> object:
        return {'replaced': True}

    for dialect_name in ('postgresql', 'sqlite', 'mysql'):
        assert sanitize_payload({'text': 'a\x00b'}, dialect_name=dialect_name, custom=custom) == {'replaced': True}


# --- value round-trips on the parametrized engine (SQLite always, PostgreSQL when DSN set) ---


async def test_message_row_roundtrips_cost_uuid_and_payload(engine: AsyncEngine) -> None:
    original = text_response()
    payload = dict(dump_message(original))
    conversation_id = uuid.uuid4()
    message_id = uuid.uuid4()

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        session.add(DefaultConversation(id=conversation_id, conversation_key='dialect-roundtrip'))
        session.add(
            DefaultMessage(
                id=message_id,
                conversation_pk=conversation_id,
                seq=0,
                kind='response',
                message=payload,
                content_hash=message_content_hash(original),
                cost=Decimal('0.00125'),
            )
        )
        await session.commit()

    async with maker() as session:
        row = await session.scalar(sa.select(DefaultMessage).where(DefaultMessage.conversation_pk == conversation_id))
        assert row is not None
        assert isinstance(row.cost, Decimal)
        assert row.cost == Decimal('0.00125')
        assert isinstance(row.id, uuid.UUID)
        assert row.id == message_id
        (reloaded,) = load_messages([row.message])
        assert ModelMessagesTypeAdapter.dump_json([reloaded]) == ModelMessagesTypeAdapter.dump_json([original])


# --- NUL handling per backend (documents why the sanitize gate is dialect-scoped) ---


def _nul_message(conversation_pk: uuid.UUID, payload: dict[str, object]) -> DefaultMessage:
    return DefaultMessage(
        conversation_pk=conversation_pk, seq=0, kind='request', message=payload, content_hash='0' * 64
    )


@pytest.mark.skipif(not os.environ.get('PAI_SQLA_TEST_PG_DSN'), reason='postgres only')
async def test_sanitized_nul_payload_stores_on_postgres(engine: AsyncEngine) -> None:
    if engine.dialect.name != 'postgresql':
        pytest.skip('postgres only')
    payload = {'text': 'before\x00after'}
    sanitized = dict(sanitize_payload(payload, dialect_name=engine.dialect.name, custom=None))
    conversation_id = uuid.uuid4()

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        session.add(DefaultConversation(id=conversation_id, conversation_key='nul-postgres'))
        session.add(_nul_message(conversation_id, sanitized))
        await session.commit()

    async with maker() as session:
        row = await session.scalar(sa.select(DefaultMessage).where(DefaultMessage.conversation_pk == conversation_id))
        assert row is not None
        assert row.message['text'] == 'beforeafter'


async def test_raw_nul_survives_unsanitized_on_sqlite(engine: AsyncEngine) -> None:
    if engine.dialect.name != 'sqlite':
        pytest.skip('sqlite only')
    # The JSON serializer escapes NUL to the backslash-u0000 JSON escape, which SQLite stores and round-trips fine,
    # while PostgreSQL jsonb rejects that escape. That is why the default sanitize gate only
    # fires for PostgreSQL: stripping on SQLite would lose data for no reason.
    payload: dict[str, object] = {'text': 'before\x00after'}
    conversation_id = uuid.uuid4()

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        session.add(DefaultConversation(id=conversation_id, conversation_key='nul-sqlite'))
        session.add(_nul_message(conversation_id, payload))
        await session.commit()

    async with maker() as session:
        row = await session.scalar(sa.select(DefaultMessage).where(DefaultMessage.conversation_pk == conversation_id))
        assert row is not None
        assert row.message['text'] == 'before\x00after'
