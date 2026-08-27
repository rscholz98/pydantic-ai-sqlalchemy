"""Retention: purge and per-conversation deletion.

Implemented by work unit U7. Signatures are frozen; see ``_store.SQLAlchemyChatStore``.

Implementation contract:
- ``purge_older_than`` keys off ``last_activity_at``; count affected rows per table first
  (that is the ``dry_run`` report), then delete conversations and rely on FK CASCADE.
- ``delete_conversation`` returns the number of deleted message rows; raise
  ``ConversationNotFoundError`` for unknown keys.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from ._types import PurgeReport

if TYPE_CHECKING:
    from ._store import SQLAlchemyChatStore

__all__ = ['delete_conversation', 'purge_older_than']


async def delete_conversation(
    store: SQLAlchemyChatStore,
    *,
    conversation_key: str,
    session: AsyncSession | None = None,
) -> int:
    raise NotImplementedError('implemented in unit U7')


async def purge_older_than(
    store: SQLAlchemyChatStore,
    cutoff: datetime,
    *,
    dry_run: bool = False,
    session: AsyncSession | None = None,
) -> PurgeReport:
    raise NotImplementedError('implemented in unit U7')
