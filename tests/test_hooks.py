"""U10 tests for ``build_on_complete``: kwarg plumbing with a recorder double, key
resolution (str, sync/async callable, non-str rejection), session factory scoping and
commit, error policy, and idempotent re-firing against the real store."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, cast

import pytest
import sqlalchemy as sa
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.run import AgentRunResult
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from pydantic_ai_sqlalchemy import (
    DefaultConversation,
    DefaultMessage,
    SaveResult,
    SQLAlchemyChatStore,
    StoreError,
    build_on_complete,
)

from .message_fixtures import text_response


def _respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    return text_response('hello from the function model')


@pytest.fixture
async def run_result() -> AgentRunResult[str]:
    agent = Agent(FunctionModel(_respond))
    return await agent.run('hi there')


class _Recorder:
    """Stands in for the store: ``build_on_complete`` only ever calls ``save_run``.

    The signature mirrors ``SQLAlchemyChatStore.save_run`` so drift there breaks these
    tests instead of being hidden by a loose fake.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.events: list[str] = []
        self.raise_on_save: Exception | None = None

    async def save_run(
        self,
        result: AgentRunResult[Any],
        *,
        conversation_key: str | None = None,
        agent_name: str | None = None,
        only_new: bool = True,
        final_message_id: uuid.UUID | None = None,
        session: AsyncSession | None = None,
    ) -> SaveResult:
        self.events.append('save')
        self.calls.append(
            {
                'result': result,
                'conversation_key': conversation_key,
                'agent_name': agent_name,
                'only_new': only_new,
                'final_message_id': final_message_id,
                'session': session,
            }
        )
        if self.raise_on_save is not None:
            raise self.raise_on_save
        return SaveResult(
            conversation_key=conversation_key or 'default',
            conversation_pk=uuid.uuid4(),
            run_id=None,
            message_ids=(),
        )


def _as_store(recorder: _Recorder) -> SQLAlchemyChatStore:
    return cast(SQLAlchemyChatStore, recorder)


class _FakeSession:
    """Minimal session double: only ``commit`` is touched by the hook itself."""

    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def commit(self) -> None:
        self._events.append('commit')


# -- key resolution and kwarg plumbing (recorder double) ----------------------------------


async def test_str_key_and_agent_name_forwarded(run_result: AgentRunResult[str]) -> None:
    recorder = _Recorder()
    callback = build_on_complete(_as_store(recorder), conversation_key='conv-42', agent_name='support-bot')
    assert await callback(run_result) is None
    assert recorder.calls == [
        {
            'result': run_result,
            'conversation_key': 'conv-42',
            'agent_name': 'support-bot',
            'only_new': True,
            'final_message_id': None,
            'session': None,
        }
    ]


async def test_callable_key_invoked_with_result(run_result: AgentRunResult[str]) -> None:
    recorder = _Recorder()
    seen: list[AgentRunResult[Any]] = []

    def derive_key(result: AgentRunResult[Any]) -> str:
        seen.append(result)
        return 'derived-key'

    callback = build_on_complete(_as_store(recorder), conversation_key=derive_key)
    await callback(run_result)
    assert seen == [run_result]
    assert recorder.calls[0]['conversation_key'] == 'derived-key'


async def test_async_callable_key_awaited(run_result: AgentRunResult[str]) -> None:
    recorder = _Recorder()

    async def derive_key(result: AgentRunResult[Any]) -> str:
        assert result is run_result
        return 'async-key'

    callback = build_on_complete(_as_store(recorder), conversation_key=derive_key)
    await callback(run_result)
    assert recorder.calls[0]['conversation_key'] == 'async-key'


async def test_none_key_passed_as_none(run_result: AgentRunResult[str]) -> None:
    recorder = _Recorder()
    callback = build_on_complete(_as_store(recorder))
    await callback(run_result)
    assert recorder.calls[0]['conversation_key'] is None
    assert recorder.calls[0]['agent_name'] is None
    assert recorder.calls[0]['session'] is None


async def test_only_new_forwarded(run_result: AgentRunResult[str]) -> None:
    recorder = _Recorder()
    callback = build_on_complete(_as_store(recorder), conversation_key='conv-all', only_new=False)
    await callback(run_result)
    assert recorder.calls[0]['only_new'] is False


async def test_callable_key_must_resolve_to_str(run_result: AgentRunResult[str]) -> None:
    recorder = _Recorder()
    bad = cast('Callable[[AgentRunResult[Any]], str]', lambda _result: 123)
    callback = build_on_complete(_as_store(recorder), conversation_key=bad)
    with pytest.raises(StoreError, match='expected str'):
        await callback(run_result)
    assert recorder.calls == []


# -- session factory ----------------------------------------------------------------------


async def test_session_factory_session_forwarded_scoped_and_committed(run_result: AgentRunResult[str]) -> None:
    recorder = _Recorder()
    fake_session = _FakeSession(recorder.events)

    @asynccontextmanager
    async def session_factory() -> AsyncIterator[AsyncSession]:
        recorder.events.append('enter')
        try:
            yield cast(AsyncSession, fake_session)
        finally:
            recorder.events.append('exit')

    callback = build_on_complete(_as_store(recorder), conversation_key='conv-scoped', session_factory=session_factory)
    await callback(run_result)
    assert recorder.calls[0]['session'] is fake_session
    assert recorder.events == ['enter', 'save', 'commit', 'exit']


# -- error policy -------------------------------------------------------------------------


async def test_save_error_propagates_by_default(run_result: AgentRunResult[str]) -> None:
    recorder = _Recorder()
    recorder.raise_on_save = RuntimeError('db down')
    callback = build_on_complete(_as_store(recorder), conversation_key='conv-err')
    with pytest.raises(RuntimeError, match='db down'):
        await callback(run_result)


async def test_on_error_suppresses_and_receives_exception(run_result: AgentRunResult[str]) -> None:
    recorder = _Recorder()
    failure = RuntimeError('db down')
    recorder.raise_on_save = failure
    captured: list[tuple[AgentRunResult[Any], BaseException]] = []

    async def on_error(result: AgentRunResult[Any], exc: BaseException) -> None:
        captured.append((result, exc))

    callback = build_on_complete(_as_store(recorder), conversation_key='conv-err', on_error=on_error)
    assert await callback(run_result) is None
    assert captured == [(run_result, failure)]


async def test_sync_on_error_supported(run_result: AgentRunResult[str]) -> None:
    recorder = _Recorder()
    recorder.raise_on_save = RuntimeError('db down')
    captured: list[BaseException] = []
    callback = build_on_complete(
        _as_store(recorder),
        conversation_key='conv-err',
        on_error=lambda _result, exc: captured.append(exc),
    )
    await callback(run_result)
    assert len(captured) == 1


# -- integration against the real store ---------------------------------------------------


async def _message_count(maker: async_sessionmaker[AsyncSession], conversation_key: str) -> int:
    async with maker() as session:
        count = await session.scalar(
            sa.select(sa.func.count())
            .select_from(DefaultMessage)
            .join(DefaultConversation, DefaultMessage.conversation_pk == DefaultConversation.id)
            .where(DefaultConversation.conversation_key == conversation_key)
        )
    return count or 0


async def test_refire_adds_no_rows(
    store: SQLAlchemyChatStore, engine: AsyncEngine, run_result: AgentRunResult[str]
) -> None:
    maker = async_sessionmaker(engine)
    callback = build_on_complete(store, conversation_key='conv-hooks')
    try:
        await callback(run_result)
    except NotImplementedError:
        pytest.xfail('depends on unit U2')
    count_after_first = await _message_count(maker, 'conv-hooks')
    assert count_after_first > 0
    try:
        await callback(run_result)
    except NotImplementedError:
        pytest.xfail('depends on unit U2')
    assert await _message_count(maker, 'conv-hooks') == count_after_first


async def test_session_factory_rows_persist_after_exit(
    store: SQLAlchemyChatStore, engine: AsyncEngine, run_result: AgentRunResult[str]
) -> None:
    maker = async_sessionmaker(engine, expire_on_commit=False)
    callback = build_on_complete(store, conversation_key='conv-factory', session_factory=maker)
    try:
        await callback(run_result)
    except NotImplementedError:
        pytest.xfail('depends on unit U2')
    assert await _message_count(maker, 'conv-factory') > 0
