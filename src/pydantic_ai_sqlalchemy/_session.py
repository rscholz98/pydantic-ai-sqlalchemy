"""Session provisioning and commit-ownership rules.

Contract:
- A caller-provided ``AsyncSession`` is never committed, rolled back or closed by the store;
  the store only flushes. This keeps host transaction management (e.g. row-level-security
  session variables armed on ``session.info``) fully in the caller's hands.
- Without a caller session, the store opens one from its bind and commits on success.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import cast

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from ._exceptions import StoreError

__all__ = ['SessionProvider']


class SessionProvider:
    """Resolves the session for each store call from an optional engine or sessionmaker bind."""

    __slots__ = ('_maker',)

    def __init__(self, bind: AsyncEngine | async_sessionmaker[AsyncSession] | None) -> None:
        self._maker = _resolve_maker(bind)

    @property
    def has_bind(self) -> bool:
        return self._maker is not None

    @asynccontextmanager
    async def scope(self, session: AsyncSession | None = None) -> AsyncGenerator[tuple[AsyncSession, bool]]:
        """Yield ``(session, owned)``; commit/rollback only when the store owns the session."""
        if session is not None:
            yield session, False
            return
        if self._maker is None:
            raise StoreError('no session passed and the store was constructed without a bind')
        async with self._maker() as owned_session:
            try:
                yield owned_session, True
            except BaseException:
                await owned_session.rollback()
                raise
            else:
                await owned_session.commit()


def _resolve_maker(bind: object) -> async_sessionmaker[AsyncSession] | None:
    if bind is None:
        return None
    if isinstance(bind, async_sessionmaker):
        return cast('async_sessionmaker[AsyncSession]', bind)
    if isinstance(bind, AsyncEngine):
        return async_sessionmaker(bind, expire_on_commit=False, autoflush=False)
    raise StoreError(f'bind must be an AsyncEngine or async_sessionmaker, got {type(bind).__name__}')
