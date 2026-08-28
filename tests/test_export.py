"""U8 tests for JSONL export and import."""

from __future__ import annotations

import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
import sqlalchemy as sa
from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from pydantic_ai_sqlalchemy import (
    DefaultConversation,
    DefaultMessage,
    DefaultRun,
    DefaultToolCall,
    SQLAlchemyChatStore,
)
from pydantic_ai_sqlalchemy._serialize import dump_message, extract_denorm, load_messages, message_content_hash

from .message_fixtures import binary_user_request, text_response, tool_turn, user_request

LINE_KEYS = {'conversation_key', 'seq', 'run_id', 'message'}


def _message_row(
    conversation_pk: uuid.UUID,
    seq: int,
    message: ModelMessage,
    *,
    run_id: str | None = None,
    created_at: datetime | None = None,
) -> DefaultMessage:
    denorm = extract_denorm(message)
    row = DefaultMessage(
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
        has_user_prompt=denorm.has_user_prompt,
        tool_call_count=denorm.tool_call_count,
    )
    if created_at is not None:
        row.created_at = created_at
    return row


async def _seed_conversation(
    session: AsyncSession,
    conversation_key: str,
    messages: list[ModelMessage],
    *,
    run_id: str | None = None,
    created_at: datetime | None = None,
) -> None:
    conversation = DefaultConversation(conversation_key=conversation_key, message_count=len(messages))
    session.add(conversation)
    await session.flush()
    for index, message in enumerate(messages, start=1):
        session.add(_message_row(conversation.id, index, message, run_id=run_id, created_at=created_at))


async def _seed_two_conversations(engine: AsyncEngine) -> dict[str, list[ModelMessage]]:
    """Two conversations: a tool turn on 'alpha' and a binary-content user turn on 'beta'."""
    histories = {
        'alpha': tool_turn(),
        'beta': [binary_user_request(), text_response('An image.')],
    }
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        await _seed_conversation(session, 'alpha', histories['alpha'], run_id='run-alpha')
        await _seed_conversation(session, 'beta', histories['beta'], run_id='run-beta')
        await session.commit()
    return histories


def _parse_lines(data: bytes) -> list[dict[str, object]]:
    return [json.loads(line) for line in data.splitlines() if line.strip()]


async def _reload_history(engine: AsyncEngine, conversation_key: str) -> list[ModelMessage]:
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        payloads = (
            await session.scalars(
                sa.select(DefaultMessage.message)
                .join_from(
                    DefaultMessage, DefaultConversation, DefaultMessage.conversation_pk == DefaultConversation.id
                )
                .where(DefaultConversation.conversation_key == conversation_key)
                .order_by(DefaultMessage.seq)
            )
        ).all()
    return load_messages(payloads)


async def test_export_to_bytesio_schema_and_order(engine: AsyncEngine, store: SQLAlchemyChatStore) -> None:
    histories = await _seed_two_conversations(engine)
    buffer = io.BytesIO()
    count = await store.export_jsonl(buffer)
    assert count == 6
    assert not buffer.closed

    records = _parse_lines(buffer.getvalue())
    assert len(records) == 6
    for record in records:
        assert set(record) == LINE_KEYS
        assert isinstance(record['message'], dict)
    assert [(r['conversation_key'], r['seq']) for r in records] == [
        ('alpha', 1),
        ('alpha', 2),
        ('alpha', 3),
        ('alpha', 4),
        ('beta', 1),
        ('beta', 2),
    ]
    assert {r['run_id'] for r in records} == {'run-alpha', 'run-beta'}
    exported_alpha = [r['message'] for r in records if r['conversation_key'] == 'alpha']
    assert exported_alpha == [dump_message(message) for message in histories['alpha']]


async def test_export_to_path(engine: AsyncEngine, store: SQLAlchemyChatStore, tmp_path: Path) -> None:
    await _seed_two_conversations(engine)
    target = tmp_path / 'export.jsonl'
    count = await store.export_jsonl(target)
    assert count == 6
    records = _parse_lines(target.read_bytes())
    assert len(records) == 6
    assert all(set(record) == LINE_KEYS for record in records)


async def test_export_filters_by_conversation_keys(engine: AsyncEngine, store: SQLAlchemyChatStore) -> None:
    await _seed_two_conversations(engine)
    buffer = io.BytesIO()
    count = await store.export_jsonl(buffer, conversation_keys=['beta'])
    assert count == 2
    records = _parse_lines(buffer.getvalue())
    assert {r['conversation_key'] for r in records} == {'beta'}


async def test_export_filters_by_since(engine: AsyncEngine, store: SQLAlchemyChatStore) -> None:
    old = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    new = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        await _seed_conversation(session, 'old-conv', [user_request('old')], created_at=old)
        await _seed_conversation(session, 'new-conv', [user_request('new'), text_response()], created_at=new)
        await session.commit()

    buffer = io.BytesIO()
    count = await store.export_jsonl(buffer, since=datetime(2025, 6, 1, tzinfo=timezone.utc))
    assert count == 2
    records = _parse_lines(buffer.getvalue())
    assert {r['conversation_key'] for r in records} == {'new-conv'}


async def test_export_import_round_trip(engine: AsyncEngine, store: SQLAlchemyChatStore, tmp_path: Path) -> None:
    histories = await _seed_two_conversations(engine)
    target = tmp_path / 'roundtrip.jsonl'
    exported = await store.export_jsonl(target)
    assert exported == 6

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        for model in (DefaultToolCall, DefaultRun, DefaultMessage, DefaultConversation):
            await session.execute(sa.delete(model))
        await session.commit()

    imported = await store.import_jsonl(target)
    assert imported == 6

    for conversation_key, original in histories.items():
        reloaded = await _reload_history(engine, conversation_key)
        assert ModelMessagesTypeAdapter.dump_json(reloaded) == ModelMessagesTypeAdapter.dump_json(original)

    async with maker() as session:
        conversations = (await session.scalars(sa.select(DefaultConversation))).all()
        counts = {c.conversation_key: c.message_count for c in conversations}
        assert counts == {'alpha': 4, 'beta': 2}
        assert all(c.last_activity_at is not None for c in conversations)


async def test_import_appends_after_existing_max_seq(engine: AsyncEngine, store: SQLAlchemyChatStore) -> None:
    existing = [user_request('first'), text_response('second')]
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        await _seed_conversation(session, 'alpha', existing, run_id='run-old')
        await session.commit()

    appended = text_response('appended later')
    line = json.dumps(
        {'conversation_key': 'alpha', 'seq': 1, 'run_id': 'run-new', 'message': dump_message(appended)}
    ).encode('utf-8')
    source = io.BytesIO(b'\n' + line + b'\n\n')  # blank lines are skipped

    imported = await store.import_jsonl(source)
    assert imported == 1

    async with maker() as session:
        rows = (
            await session.execute(sa.select(DefaultMessage.seq, DefaultMessage.run_id).order_by(DefaultMessage.seq))
        ).all()
        assert [tuple(row) for row in rows] == [(1, 'run-old'), (2, 'run-old'), (3, 'run-new')]
        conversation = (await session.scalars(sa.select(DefaultConversation))).one()
        assert conversation.message_count == 3

    reloaded = await _reload_history(engine, 'alpha')
    assert ModelMessagesTypeAdapter.dump_json(reloaded) == ModelMessagesTypeAdapter.dump_json([*existing, appended])


async def test_import_corrupt_line_raises(store: SQLAlchemyChatStore) -> None:
    source = io.BytesIO(
        json.dumps({'conversation_key': 'bad', 'seq': 1, 'run_id': None, 'message': {'kind': 'bogus'}}).encode('utf-8')
    )
    with pytest.raises(ValidationError):
        await store.import_jsonl(source)


async def test_import_non_object_line_raises(store: SQLAlchemyChatStore) -> None:
    with pytest.raises(ValueError):
        await store.import_jsonl(io.BytesIO(b'[1, 2, 3]\n'))
