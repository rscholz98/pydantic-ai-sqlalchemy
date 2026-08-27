"""Harness ``HistorySource`` read seam: run listing and per-run history.

Implemented by work unit U9. Signatures are frozen; see ``_store.SQLAlchemyChatStore``.

The store's ``list_runs``/``run_history`` methods must structurally satisfy the
``pydantic_ai_harness.conversation_search.HistorySource`` runtime-checkable protocol.
The harness import stays lazy and optional; ``pydantic_ai_sqlalchemy.RunRecord`` is the
structural stand-in when the harness is absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelMessage
from sqlalchemy.ext.asyncio import AsyncSession

from ._types import RunRecord

if TYPE_CHECKING:
    from ._store import SQLAlchemyChatStore

__all__ = ['list_runs', 'run_history']


async def list_runs(
    store: SQLAlchemyChatStore,
    *,
    conversation_key: str | None = None,
    session: AsyncSession | None = None,
) -> list[RunRecord]:
    """All runs (optionally scoped to one conversation) sorted by ``started_at`` ascending."""
    raise NotImplementedError('implemented in unit U9')


async def run_history(
    store: SQLAlchemyChatStore,
    *,
    run_id: str,
    session: AsyncSession | None = None,
) -> list[ModelMessage]:
    """The messages of one run, ordered by ``seq``."""
    raise NotImplementedError('implemented in unit U9')
