"""Read path: history loading with turn windowing, conversation listing, transcripts.

Implemented by work unit U3. Signatures are frozen; see ``_store.SQLAlchemyChatStore``.

Implementation contract:
- ``load_history`` orders by ``seq``; ``max_turns`` selects the last N turns using the
  ``has_user_prompt`` column as the turn boundary in SQL (a turn starts at a request row
  containing a user prompt), so tool-call/tool-return pairs are never split.
- When ``sanitize`` is true (or the store was built with ``trusted_history=False`` and
  ``sanitize`` is None), pass the loaded history through
  ``pydantic_ai.messages.sanitize_messages`` before returning.
- Raise ``ConversationNotFoundError`` for unknown keys on ``get_transcript``; return an
  empty list from ``load_history`` (loading an empty history must be cheap and non-fatal).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelMessage
from sqlalchemy.ext.asyncio import AsyncSession

from ._types import ConversationRecord

if TYPE_CHECKING:
    from ._store import SQLAlchemyChatStore

__all__ = ['get_transcript', 'list_conversations', 'load_history']


async def load_history(
    store: SQLAlchemyChatStore,
    *,
    conversation_key: str,
    max_turns: int | None = None,
    sanitize: bool | None = None,
    session: AsyncSession | None = None,
) -> list[ModelMessage]:
    raise NotImplementedError('implemented in unit U3')


async def get_transcript(
    store: SQLAlchemyChatStore,
    *,
    conversation_key: str,
    session: AsyncSession | None = None,
) -> str:
    raise NotImplementedError('implemented in unit U3')


async def list_conversations(
    store: SQLAlchemyChatStore,
    *,
    since: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession | None = None,
) -> list[ConversationRecord]:
    raise NotImplementedError('implemented in unit U3')
