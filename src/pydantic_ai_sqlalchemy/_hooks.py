"""Integration hooks: the ``on_complete`` save-callback factory.

Implemented by work unit U10. Signatures are frozen.

``build_on_complete`` returns an async callable suitable for
``UIAdapter.dispatch_request(on_complete=...)`` (and for calling manually after
``agent.run``). The callback saves ``result.new_messages()`` via ``store.save_run``;
re-firing it must write nothing thanks to content-hash idempotency.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from ._exceptions import StoreError

if TYPE_CHECKING:
    from pydantic_ai.run import AgentRunResult

    from ._store import SQLAlchemyChatStore

__all__ = ['build_on_complete']


def build_on_complete(
    store: SQLAlchemyChatStore,
    *,
    conversation_key: str | Callable[[AgentRunResult[Any]], str | Awaitable[str]] | None = None,
    agent_name: str | None = None,
    session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]] | None = None,
    on_error: Callable[[AgentRunResult[Any], BaseException], Awaitable[None] | None] | None = None,
    only_new: bool = True,
) -> Callable[[AgentRunResult[Any]], Awaitable[None]]:
    """Build a save callback; ``session_factory`` lets hosts supply scoped (e.g. tenant) sessions.

    The returned coroutine function takes exactly one argument, the finished
    ``AgentRunResult``. That is the shape pydantic-ai's
    ``UIAdapter.dispatch_request(on_complete=...)`` expects (not the zero-argument
    ``StreamedRunResult._on_complete``), and it works just as well called manually after
    ``agent.run(...)``. Each invocation persists the run via
    :meth:`SQLAlchemyChatStore.save_run` and returns ``None``.

    ``conversation_key`` may be a fixed string, a callable deriving the key from the
    result (sync, or async in which case its awaitable is awaited), or ``None`` to let
    ``save_run`` fall back to the run's ``conversation_id``. A callable resolving to
    anything other than ``str`` raises :class:`StoreError`. ``agent_name`` and
    ``only_new`` are forwarded verbatim to ``save_run``.

    When ``session_factory`` is given, each invocation opens one session from it,
    passes that session to ``save_run``, and commits it after a successful save: the
    hook owns that session's lifecycle (the store itself never commits sessions a host
    passes into its methods). Without a factory the store manages its own session.

    By default persistence failures propagate to the caller, so in a UI adapter they
    surface in the request stream. Hosts that want best-effort archiving pass
    ``on_error``: it is called with the result and the exception (awaited when it
    returns an awaitable) and the exception is not re-raised. Key-resolution errors
    always propagate.

    The hook keeps no state of its own, so re-firing it for the same result is safe as
    long as the resolved conversation key is deterministic for that result: idempotency
    comes from ``save_run``'s content-hash deduplication.
    """

    async def on_complete(result: AgentRunResult[Any]) -> None:
        if callable(conversation_key):
            raw: object = conversation_key(result)
            resolved: object = await raw if inspect.isawaitable(raw) else raw
        else:
            resolved = conversation_key
        # Runtime defense against untyped derivers (e.g. one returning an int or a coroutine
        # object it forgot to type as async): the declared types make this look redundant.
        if resolved is not None and not isinstance(resolved, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise StoreError(
                f'conversation_key callable resolved to {type(resolved).__name__}; expected str '
                '(async derivers are awaited, anything else is a bug in the deriver)'
            )
        try:
            if session_factory is not None:
                async with session_factory() as session:
                    await store.save_run(
                        result, conversation_key=resolved, agent_name=agent_name, only_new=only_new, session=session
                    )
                    await session.commit()
            else:
                await store.save_run(result, conversation_key=resolved, agent_name=agent_name, only_new=only_new)
        except Exception as exc:
            if on_error is None:
                raise
            outcome = on_error(result, exc)
            if inspect.isawaitable(outcome):
                await outcome

    return on_complete
