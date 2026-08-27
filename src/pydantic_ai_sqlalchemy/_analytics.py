"""Analytics aggregate queries over the denormalized columns.

Implemented by work unit U6. Signatures are frozen; see ``_store.SQLAlchemyChatStore``.

Implementation contract:
- Aggregate over the denormalized message/run columns only; never parse JSON blobs in SQL.
- Filter synthetic rows (``model_name`` in ('interrupted', 'error')) out of token/cost sums.
- Group by calendar day with ``sa.func.date(...)`` so SQLite and PostgreSQL both work.
- Return raw numbers; no thresholds or judgments baked in.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal

from sqlalchemy.ext.asyncio import AsyncSession

from ._types import ConversationUsage, CostRow, DailyUsage, ModelUsage, RunStats, ToolUsage

if TYPE_CHECKING:
    from ._store import SQLAlchemyChatStore

__all__ = ['cost_report', 'run_stats', 'tool_usage', 'usage_by_conversation', 'usage_by_day', 'usage_by_model']


async def usage_by_day(
    store: SQLAlchemyChatStore,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    session: AsyncSession | None = None,
) -> list[DailyUsage]:
    raise NotImplementedError('implemented in unit U6')


async def usage_by_model(
    store: SQLAlchemyChatStore,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    session: AsyncSession | None = None,
) -> list[ModelUsage]:
    raise NotImplementedError('implemented in unit U6')


async def usage_by_conversation(
    store: SQLAlchemyChatStore,
    *,
    since: datetime | None = None,
    limit: int = 100,
    session: AsyncSession | None = None,
) -> list[ConversationUsage]:
    raise NotImplementedError('implemented in unit U6')


async def run_stats(
    store: SQLAlchemyChatStore,
    *,
    since: datetime | None = None,
    session: AsyncSession | None = None,
) -> list[RunStats]:
    raise NotImplementedError('implemented in unit U6')


async def tool_usage(
    store: SQLAlchemyChatStore,
    *,
    since: datetime | None = None,
    session: AsyncSession | None = None,
) -> list[ToolUsage]:
    """Aggregate the tool_calls table; fall back to summing ``tool_call_count`` when extraction is off."""
    raise NotImplementedError('implemented in unit U6')


async def cost_report(
    store: SQLAlchemyChatStore,
    *,
    group_by: Literal['day', 'model', 'conversation'],
    since: datetime | None = None,
    until: datetime | None = None,
    session: AsyncSession | None = None,
) -> list[CostRow]:
    raise NotImplementedError('implemented in unit U6')
