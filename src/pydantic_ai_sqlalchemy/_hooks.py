"""Integration hooks: the ``on_complete`` save-callback factory.

Implemented by work unit U10. Signatures are frozen.

``build_on_complete`` returns an async callable suitable for
``UIAdapter.dispatch_request(on_complete=...)`` (and for calling manually after
``agent.run``). The callback saves ``result.new_messages()`` via ``store.save_run``;
re-firing it must write nothing thanks to content-hash idempotency.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

if TYPE_CHECKING:
    from pydantic_ai.run import AgentRunResult

    from ._store import SQLAlchemyChatStore

__all__ = ['build_on_complete']


def build_on_complete(
    store: SQLAlchemyChatStore,
    *,
    conversation_key: str | Callable[[AgentRunResult[Any]], str] | None = None,
    agent_name: str | None = None,
    session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]] | None = None,
) -> Callable[[AgentRunResult[Any]], Awaitable[None]]:
    """Build a save callback; ``session_factory`` lets hosts supply scoped (e.g. tenant) sessions."""
    raise NotImplementedError('implemented in unit U10')
