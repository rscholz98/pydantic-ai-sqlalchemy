"""U3 tests for the read path: history loading, transcripts, conversation listing.

Rows are seeded directly through the ORM (dump_message + extract_denorm + explicit seq)
so these tests stay independent of the write-path unit.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart, UserPromptPart
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from pydantic_ai_sqlalchemy import SQLAlchemyChatStore, _read
from pydantic_ai_sqlalchemy._exceptions import ConversationNotFoundError
from pydantic_ai_sqlalchemy._models import DefaultConversation, DefaultMessage
from pydantic_ai_sqlalchemy._serialize import dump_message, extract_denorm, message_content_hash
from pydantic_ai_sqlalchemy._turns import slice_to_recent_turns

from .message_fixtures import (
    sample_conversation,
    sample_turn,
    system_user_request,
    text_response,
    tool_call_response,
    tool_turn,
    user_request,
)

UTC = timezone.utc


async def seed_conversation(
    engine: AsyncEngine,
    conversation_key: str,
    messages: list[ModelMessage],
    *,
    last_activity_at: datetime | None = None,
) -> uuid.UUID:
    """Insert one conversation and its message rows directly, bypassing the write path."""
    async with AsyncSession(engine) as db:
        conversation = DefaultConversation(
            conversation_key=conversation_key,
            message_count=len(messages),
            last_activity_at=last_activity_at,
        )
        db.add(conversation)
        await db.flush()
        for seq, message in enumerate(messages, start=1):
            denorm = extract_denorm(message)
            db.add(
                DefaultMessage(
                    conversation_pk=conversation.id,
                    seq=seq,
                    kind=denorm.kind,
                    message=dump_message(message),
                    content_hash=message_content_hash(message),
                    run_id=denorm.run_id,
                    message_timestamp=denorm.message_timestamp,
                    model_name=denorm.model_name,
                    provider_name=denorm.provider_name,
                    input_tokens=denorm.input_tokens,
                    output_tokens=denorm.output_tokens,
                    has_user_prompt=denorm.has_user_prompt,
                    tool_call_count=denorm.tool_call_count,
                )
            )
        conversation_pk = conversation.id
        await db.commit()
    return conversation_pk


# -- load_history ------------------------------------------------------------------------------


async def test_load_history_full(store: SQLAlchemyChatStore, engine: AsyncEngine) -> None:
    messages = sample_conversation()
    await seed_conversation(engine, 'conv-full', messages)
    loaded = await store.load_history(conversation_key='conv-full')
    assert loaded == messages


async def test_load_history_windowed_keeps_tool_turn_whole(store: SQLAlchemyChatStore, engine: AsyncEngine) -> None:
    messages = sample_conversation()
    await seed_conversation(engine, 'conv-window', messages)
    loaded = await store.load_history(conversation_key='conv-window', max_turns=2)
    # The window opens at the tool turn's user request and keeps its call/return pair intact.
    assert loaded == messages[2:]


async def test_load_history_windowed_matches_in_memory_slicing(store: SQLAlchemyChatStore, engine: AsyncEngine) -> None:
    messages = sample_conversation()
    await seed_conversation(engine, 'conv-equiv', messages)
    for max_turns in (1, 2, 3, 4, 99):
        loaded = await store.load_history(conversation_key='conv-equiv', max_turns=max_turns)
        assert loaded == slice_to_recent_turns(messages, max_turns=max_turns), f'max_turns={max_turns}'


async def test_load_history_window_treats_mixed_resume_request_as_same_turn(
    store: SQLAlchemyChatStore, engine: AsyncEngine
) -> None:
    # A deferred/HITL resume request carries a tool return next to a fresh user prompt; it
    # continues the turn, so max_turns=1 must keep the whole exchange.
    resume_request = ModelRequest(
        parts=[
            ToolReturnPart(tool_name='get_weather', content={'temperature': 21}, tool_call_id='call_1'),
            UserPromptPart(content='And tomorrow?'),
        ]
    )
    messages: list[ModelMessage] = [
        user_request('Weather?'),
        tool_call_response(),
        resume_request,
        text_response('21 C'),
    ]
    await seed_conversation(engine, 'conv-mixed', messages)
    loaded = await store.load_history(conversation_key='conv-mixed', max_turns=1)
    assert loaded == messages
    assert loaded == slice_to_recent_turns(messages, max_turns=1)


async def test_load_history_window_scans_across_chunks(
    store: SQLAlchemyChatStore, engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_read, '_WINDOW_CHUNK_SIZE', 2)
    messages = sample_conversation()
    await seed_conversation(engine, 'conv-chunks', messages)
    for max_turns in (1, 2, 3, 4):
        loaded = await store.load_history(conversation_key='conv-chunks', max_turns=max_turns)
        assert loaded == slice_to_recent_turns(messages, max_turns=max_turns), f'max_turns={max_turns}'


async def test_load_history_oversized_window_returns_all(store: SQLAlchemyChatStore, engine: AsyncEngine) -> None:
    messages = sample_conversation()
    await seed_conversation(engine, 'conv-oversized', messages)
    loaded = await store.load_history(conversation_key='conv-oversized', max_turns=50)
    assert loaded == messages


async def test_load_history_zero_and_negative_max_turns(store: SQLAlchemyChatStore, engine: AsyncEngine) -> None:
    await seed_conversation(engine, 'conv-zero', sample_conversation())
    assert await store.load_history(conversation_key='conv-zero', max_turns=0) == []
    assert await store.load_history(conversation_key='conv-zero', max_turns=-3) == []


async def test_load_history_unknown_key_returns_empty(store: SQLAlchemyChatStore) -> None:
    assert await store.load_history(conversation_key='no-such-conversation') == []
    assert await store.load_history(conversation_key='no-such-conversation', max_turns=2) == []


async def test_load_history_with_caller_session(store: SQLAlchemyChatStore, engine: AsyncEngine) -> None:
    messages = sample_turn()
    await seed_conversation(engine, 'conv-caller-session', messages)
    async with AsyncSession(engine) as db:
        loaded = await store.load_history(conversation_key='conv-caller-session', session=db)
    assert loaded == messages


# -- sanitize gate -----------------------------------------------------------------------------


def _part_kinds(messages: list[ModelMessage]) -> list[list[str]]:
    return [[part.part_kind for part in message.parts] for message in messages]


@pytest.mark.filterwarnings('ignore::UserWarning')
async def test_load_history_sanitize_true_strips_system_prompt(store: SQLAlchemyChatStore, engine: AsyncEngine) -> None:
    await seed_conversation(engine, 'conv-sanitize', [system_user_request(), text_response()])
    loaded = await store.load_history(conversation_key='conv-sanitize', sanitize=True)
    assert _part_kinds(loaded) == [['user-prompt'], ['text']]


async def test_load_history_trusted_store_keeps_system_prompt_by_default(
    store: SQLAlchemyChatStore, engine: AsyncEngine
) -> None:
    await seed_conversation(engine, 'conv-trusted', [system_user_request(), text_response()])
    loaded = await store.load_history(conversation_key='conv-trusted')
    assert _part_kinds(loaded) == [['system-prompt', 'user-prompt'], ['text']]


@pytest.mark.filterwarnings('ignore::UserWarning')
async def test_untrusted_store_sanitizes_by_default(engine: AsyncEngine) -> None:
    untrusted_store = SQLAlchemyChatStore(engine, trusted_history=False)
    await seed_conversation(engine, 'conv-untrusted', [system_user_request(), text_response()])
    loaded = await untrusted_store.load_history(conversation_key='conv-untrusted')
    assert _part_kinds(loaded) == [['user-prompt'], ['text']]
    # An explicit sanitize=False overrides the store default.
    raw = await untrusted_store.load_history(conversation_key='conv-untrusted', sanitize=False)
    assert _part_kinds(raw) == [['system-prompt', 'user-prompt'], ['text']]


# -- get_transcript ----------------------------------------------------------------------------


async def test_get_transcript(store: SQLAlchemyChatStore, engine: AsyncEngine) -> None:
    await seed_conversation(engine, 'conv-transcript', tool_turn())
    transcript = await store.get_transcript(conversation_key='conv-transcript')
    assert transcript == 'User: Weather in Berlin?\n\nAssistant: 21 C'


async def test_get_transcript_empty_conversation(store: SQLAlchemyChatStore, engine: AsyncEngine) -> None:
    await seed_conversation(engine, 'conv-empty', [])
    assert await store.get_transcript(conversation_key='conv-empty') == ''


async def test_get_transcript_unknown_key_raises(store: SQLAlchemyChatStore) -> None:
    with pytest.raises(ConversationNotFoundError):
        await store.get_transcript(conversation_key='no-such-conversation')


@pytest.mark.filterwarnings('ignore::UserWarning')
async def test_get_transcript_applies_untrusted_sanitize_gate(engine: AsyncEngine) -> None:
    untrusted_store = SQLAlchemyChatStore(engine, trusted_history=False)
    await seed_conversation(engine, 'conv-transcript-gate', [system_user_request(), text_response()])
    # Sanitizing strips the system prompt before rendering; the visible lines are unchanged.
    transcript = await untrusted_store.get_transcript(conversation_key='conv-transcript-gate')
    assert transcript == 'User: Hi\n\nAssistant: Hi!'


# -- list_conversations ------------------------------------------------------------------------


async def test_list_conversations_orders_by_last_activity_desc_nulls_last(
    store: SQLAlchemyChatStore, engine: AsyncEngine
) -> None:
    base = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    await seed_conversation(engine, 'conv-old', sample_turn(), last_activity_at=base - timedelta(days=2))
    await seed_conversation(engine, 'conv-new', sample_turn(), last_activity_at=base)
    await seed_conversation(engine, 'conv-mid', sample_turn(), last_activity_at=base - timedelta(days=1))
    await seed_conversation(engine, 'conv-null', sample_turn(), last_activity_at=None)
    records = await store.list_conversations()
    assert [record.conversation_key for record in records] == ['conv-new', 'conv-mid', 'conv-old', 'conv-null']
    newest = records[0]
    assert newest.message_count == 2
    assert newest.last_activity_at is not None


async def test_list_conversations_since_filter(store: SQLAlchemyChatStore, engine: AsyncEngine) -> None:
    base = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    await seed_conversation(engine, 'conv-old', sample_turn(), last_activity_at=base - timedelta(days=2))
    await seed_conversation(engine, 'conv-new', sample_turn(), last_activity_at=base)
    await seed_conversation(engine, 'conv-null', sample_turn(), last_activity_at=None)
    records = await store.list_conversations(since=base - timedelta(days=1))
    assert [record.conversation_key for record in records] == ['conv-new']


async def test_list_conversations_limit_and_offset(store: SQLAlchemyChatStore, engine: AsyncEngine) -> None:
    base = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    for index in range(5):
        await seed_conversation(engine, f'conv-{index}', sample_turn(), last_activity_at=base - timedelta(hours=index))
    page_one = await store.list_conversations(limit=2)
    page_two = await store.list_conversations(limit=2, offset=2)
    assert [record.conversation_key for record in page_one] == ['conv-0', 'conv-1']
    assert [record.conversation_key for record in page_two] == ['conv-2', 'conv-3']


async def test_list_conversations_empty(store: SQLAlchemyChatStore) -> None:
    assert await store.list_conversations() == []


async def test_list_conversations_timezone_normalization(store: SQLAlchemyChatStore, engine: AsyncEngine) -> None:
    # Stored values are UTC; an aware `since` in another zone must filter by instant, and
    # returned datetimes must come back timezone-aware on both backends.
    instant = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    await seed_conversation(engine, 'conv-tz', sample_turn(), last_activity_at=instant)
    cest = timezone(timedelta(hours=2))
    included = await store.list_conversations(since=(instant - timedelta(hours=1)).astimezone(cest))
    assert [record.conversation_key for record in included] == ['conv-tz']
    record = included[0]
    assert record.last_activity_at == instant
    assert record.last_activity_at is not None and record.last_activity_at.tzinfo is not None
    assert record.created_at.tzinfo is not None
    assert record.updated_at.tzinfo is not None
    excluded = await store.list_conversations(since=(instant + timedelta(hours=1)).astimezone(cest))
    assert excluded == []
