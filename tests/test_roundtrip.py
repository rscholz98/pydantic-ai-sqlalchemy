"""U1 tests: byte-fidelity round trips through the database.

Every case serializes messages with the U0 core, writes rows directly via the ORM,
reads them back ordered by ``seq`` and asserts both the raw stored payloads and the
adapter-level JSON bytes are identical to the originals. Running on both backends
(via the ``engine`` fixture) proves BinaryContent base64 bytes, Decimal cost and
tz-aware datetimes survive both SQLite JSON and PostgreSQL JSONB storage.
"""

from __future__ import annotations

import itertools
import uuid
from collections.abc import Callable, Sequence

import pytest
import sqlalchemy as sa
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from pydantic_ai_sqlalchemy import DefaultConversation, DefaultMessage, SQLAlchemyChatStore
from pydantic_ai_sqlalchemy._serialize import dump_message, extract_denorm, load_messages, message_content_hash

from .message_fixtures import ALL_BUILDERS, sample_conversation

MessageBuilder = Callable[[], ModelMessage]

#: Deterministic concatenations/orderings of the canonical builders (no randomness).
#: Full forward, full reversed and a doubled run prove seq ordering and repeats; the
#: length-2 permutations of a builder subset prove adjacent-kind independence.
BUILDER_COMBINATIONS: list[tuple[MessageBuilder, ...]] = [
    tuple(ALL_BUILDERS),
    tuple(reversed(ALL_BUILDERS)),
    tuple(ALL_BUILDERS) * 2,
    *itertools.permutations(ALL_BUILDERS[:4], 2),
]


def _dump_json(messages: Sequence[ModelMessage]) -> bytes:
    return ModelMessagesTypeAdapter.dump_json(list(messages))


def _message_row(conversation_pk: uuid.UUID, seq: int, message: ModelMessage) -> DefaultMessage:
    """Build one ORM row exactly the way the write path is specified to: payload via
    ``dump_message``, identity via ``message_content_hash``, columns via ``extract_denorm``."""
    denorm = extract_denorm(message)
    return DefaultMessage(
        conversation_pk=conversation_pk,
        seq=seq,
        kind=denorm.kind,
        message=dump_message(message),
        content_hash=message_content_hash(message),
        run_id=denorm.run_id,
        message_timestamp=denorm.message_timestamp,
        model_name=denorm.model_name,
        provider_name=denorm.provider_name,
        provider_response_id=denorm.provider_response_id,
        finish_reason=denorm.finish_reason,
        state=denorm.state,
        input_tokens=denorm.input_tokens,
        output_tokens=denorm.output_tokens,
        cache_read_tokens=denorm.cache_read_tokens,
        cache_write_tokens=denorm.cache_write_tokens,
        cost=denorm.cost,
        has_user_prompt=denorm.has_user_prompt,
        tool_call_count=denorm.tool_call_count,
    )


async def _insert_and_reload(engine: AsyncEngine, messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    """Insert one conversation with the given messages directly via the ORM and read it back.

    Also asserts the raw stored payloads still equal ``dump_message`` output, so a storage
    layer mutation that pydantic validation would silently repair still fails the test.
    """
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        conversation = DefaultConversation(conversation_key=uuid.uuid4().hex)
        session.add(conversation)
        await session.flush()
        for index, message in enumerate(messages):
            session.add(_message_row(conversation.id, index + 1, message))
        await session.commit()
        conversation_pk = conversation.id

    async with maker() as session:
        rows = await session.scalars(
            sa.select(DefaultMessage)
            .where(DefaultMessage.conversation_pk == conversation_pk)
            .order_by(DefaultMessage.seq)
        )
        payloads = [row.message for row in rows]

    for payload, message in zip(payloads, messages, strict=True):
        assert payload == dump_message(message)  # dict equality is key-order-insensitive, so JSONB-safe
    return load_messages(payloads)


@pytest.mark.parametrize('builder', ALL_BUILDERS, ids=lambda b: b.__name__)
async def test_single_message_roundtrip_is_byte_faithful(engine: AsyncEngine, builder: MessageBuilder) -> None:
    original = [builder()]
    loaded = await _insert_and_reload(engine, original)
    assert _dump_json(loaded) == _dump_json(original)


async def test_sample_conversation_roundtrip_is_byte_faithful(engine: AsyncEngine) -> None:
    original = sample_conversation()
    loaded = await _insert_and_reload(engine, original)
    assert _dump_json(loaded) == _dump_json(original)


async def test_deterministic_builder_combinations_roundtrip(engine: AsyncEngine) -> None:
    # Each combination becomes its own conversation on the same engine. Divergent
    # combinations are collected (not asserted one by one) so a regression reports
    # every affected ordering in a single run.
    diverged: list[str] = []
    for combination in BUILDER_COMBINATIONS:
        original = [builder() for builder in combination]
        loaded = await _insert_and_reload(engine, original)
        if _dump_json(loaded) != _dump_json(original):
            diverged.append('-'.join(builder.__name__ for builder in combination))
    assert not diverged, f'round trip diverged for combinations: {diverged}'


async def test_store_save_and_load_roundtrip(store: SQLAlchemyChatStore) -> None:
    original = sample_conversation()
    try:
        await store.save_messages(original, conversation_key='roundtrip-store')
        loaded = await store.load_history(conversation_key='roundtrip-store')
    except NotImplementedError:
        pytest.xfail('depends on units U2/U3')
    assert _dump_json(loaded) == _dump_json(original)
