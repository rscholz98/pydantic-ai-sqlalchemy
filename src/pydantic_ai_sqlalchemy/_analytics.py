"""Analytics aggregate queries over the denormalized columns.

Implemented by work unit U6. Signatures are frozen; see ``_store.SQLAlchemyChatStore``.

Implementation contract:
- Aggregate over the denormalized message/run columns only; never parse JSON blobs in SQL.
- Filter synthetic rows (``model_name`` in ('interrupted', 'error')) out of token/cost sums.
- Group by calendar day with ``sa.func.date(...)`` so SQLite and PostgreSQL both work.
- Return raw numbers; no thresholds or judgments baked in.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from ._types import ConversationUsage, CostRow, DailyUsage, ModelUsage, RunStats, ToolUsage

if TYPE_CHECKING:
    from sqlalchemy.orm import InstrumentedAttribute

    from ._models import BaseMessage
    from ._store import SQLAlchemyChatStore

__all__ = ['cost_report', 'run_stats', 'tool_usage', 'usage_by_conversation', 'usage_by_day', 'usage_by_model']

#: Marker rows written by the partial-save path; they carry no billable usage.
_SYNTHETIC_MODEL_NAMES: tuple[str, ...] = ('interrupted', 'error')


def _as_int(value: object) -> int:
    """Coerce an aggregate to ``int``; PostgreSQL SUM over integer columns returns Decimal."""
    if value is None:
        return 0
    if isinstance(value, (int, float, Decimal)):
        return int(value)
    return int(str(value))


def _as_float(value: object) -> float | None:
    """Coerce an aggregate to ``float``; PostgreSQL AVG over integer columns returns Decimal."""
    if value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    return float(str(value))


def _as_cost(value: object) -> Decimal | None:
    """Coerce a cost aggregate to ``Decimal``; SQLite may hand back float."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _as_day(value: object) -> date:
    """Coerce ``sa.func.date(...)`` output to ``datetime.date``; SQLite returns str, PostgreSQL date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _qualifying_response(message: type[BaseMessage]) -> sa.ColumnElement[bool]:
    """Rows whose tokens/cost count: model responses that are not synthetic markers."""
    return sa.and_(
        message.kind == 'response',
        sa.or_(message.model_name.is_(None), message.model_name.not_in(_SYNTHETIC_MODEL_NAMES)),
    )


def _time_bounds(
    column: InstrumentedAttribute[datetime | None],
    since: datetime | None,
    until: datetime | None,
) -> list[sa.ColumnElement[bool]]:
    """``since`` is inclusive, ``until`` exclusive; missing bounds add no filter."""
    bounds: list[sa.ColumnElement[bool]] = []
    if since is not None:
        bounds.append(column >= since)
    if until is not None:
        bounds.append(column < until)
    return bounds


async def usage_by_day(
    store: SQLAlchemyChatStore,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    session: AsyncSession | None = None,
) -> list[DailyUsage]:
    message = store.models.message
    qualifies = _qualifying_response(message)
    day = sa.func.date(message.message_timestamp)
    statement = (
        sa.select(
            day.label('day'),
            sa.func.count(sa.case((qualifies, 1))).label('model_requests'),
            sa.func.sum(sa.case((qualifies, message.input_tokens))).label('input_tokens'),
            sa.func.sum(sa.case((qualifies, message.output_tokens))).label('output_tokens'),
            sa.func.sum(sa.case((qualifies, message.cache_read_tokens))).label('cache_read_tokens'),
            sa.func.sum(sa.case((qualifies, message.cost))).label('cost'),
            sa.func.count(sa.distinct(message.conversation_pk)).label('active_conversations'),
        )
        .where(message.message_timestamp.is_not(None), *_time_bounds(message.message_timestamp, since, until))
        .group_by(day)
    )
    async with store.sessions.scope(session) as (db, _owned):
        rows = (await db.execute(statement)).all()
    daily = [
        DailyUsage(
            day=_as_day(row.day),
            model_requests=_as_int(row.model_requests),
            input_tokens=_as_int(row.input_tokens),
            output_tokens=_as_int(row.output_tokens),
            cache_read_tokens=_as_int(row.cache_read_tokens),
            cost=_as_cost(row.cost),
            active_conversations=_as_int(row.active_conversations),
        )
        for row in rows
    ]
    daily.sort(key=lambda item: item.day)
    return daily


async def usage_by_model(
    store: SQLAlchemyChatStore,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    session: AsyncSession | None = None,
) -> list[ModelUsage]:
    message = store.models.message
    statement = (
        sa.select(
            message.model_name,
            message.provider_name,
            sa.func.count().label('model_requests'),
            sa.func.sum(message.input_tokens).label('input_tokens'),
            sa.func.sum(message.output_tokens).label('output_tokens'),
            sa.func.sum(message.cost).label('cost'),
        )
        .where(_qualifying_response(message), *_time_bounds(message.message_timestamp, since, until))
        .group_by(message.model_name, message.provider_name)
    )
    async with store.sessions.scope(session) as (db, _owned):
        rows = (await db.execute(statement)).all()
    usage = [
        ModelUsage(
            model_name=row.model_name,
            provider_name=row.provider_name,
            model_requests=_as_int(row.model_requests),
            input_tokens=_as_int(row.input_tokens),
            output_tokens=_as_int(row.output_tokens),
            cost=_as_cost(row.cost),
        )
        for row in rows
    ]
    usage.sort(key=lambda item: (-item.model_requests, item.model_name or '', item.provider_name or ''))
    return usage


async def usage_by_conversation(
    store: SQLAlchemyChatStore,
    *,
    since: datetime | None = None,
    limit: int = 100,
    session: AsyncSession | None = None,
) -> list[ConversationUsage]:
    message = store.models.message
    conversation = store.models.conversation
    qualifies = _qualifying_response(message)
    input_sum = sa.func.sum(sa.case((qualifies, message.input_tokens)))
    output_sum = sa.func.sum(sa.case((qualifies, message.output_tokens)))
    statement = (
        sa.select(
            conversation.conversation_key,
            sa.func.count().label('message_count'),
            input_sum.label('input_tokens'),
            output_sum.label('output_tokens'),
            sa.func.sum(sa.case((qualifies, message.cost))).label('cost'),
            conversation.first_activity_at,
            conversation.last_activity_at,
        )
        .select_from(message)
        .join(conversation, onclause=message.conversation_pk == conversation.id)
        .where(*_time_bounds(message.message_timestamp, since, None))
        .group_by(
            conversation.id,
            conversation.conversation_key,
            conversation.first_activity_at,
            conversation.last_activity_at,
        )
        .order_by(
            sa.desc(sa.func.coalesce(input_sum, 0) + sa.func.coalesce(output_sum, 0)),
            conversation.conversation_key,
        )
        .limit(limit)
    )
    async with store.sessions.scope(session) as (db, _owned):
        rows = (await db.execute(statement)).all()
    return [
        ConversationUsage(
            conversation_key=row.conversation_key,
            message_count=_as_int(row.message_count),
            input_tokens=_as_int(row.input_tokens),
            output_tokens=_as_int(row.output_tokens),
            cost=_as_cost(row.cost),
            first_activity_at=row.first_activity_at,
            last_activity_at=row.last_activity_at,
        )
        for row in rows
    ]


async def run_stats(
    store: SQLAlchemyChatStore,
    *,
    since: datetime | None = None,
    session: AsyncSession | None = None,
) -> list[RunStats]:
    run = store.models.run
    statement = sa.select(
        run.state,
        sa.func.count().label('run_count'),
        sa.func.avg(run.duration_ms).label('avg_duration_ms'),
        sa.func.sum(run.input_tokens).label('input_tokens'),
        sa.func.sum(run.output_tokens).label('output_tokens'),
        sa.func.sum(run.cost).label('cost'),
    ).group_by(run.state)
    if since is not None:
        statement = statement.where(run.started_at >= since)
    async with store.sessions.scope(session) as (db, _owned):
        rows = (await db.execute(statement)).all()
    stats = [
        RunStats(
            state=row.state,
            run_count=_as_int(row.run_count),
            avg_duration_ms=_as_float(row.avg_duration_ms),
            input_tokens=_as_int(row.input_tokens),
            output_tokens=_as_int(row.output_tokens),
            cost=_as_cost(row.cost),
        )
        for row in rows
    ]
    stats.sort(key=lambda item: (-item.run_count, item.state or ''))
    return stats


async def tool_usage(
    store: SQLAlchemyChatStore,
    *,
    since: datetime | None = None,
    session: AsyncSession | None = None,
) -> list[ToolUsage]:
    """Aggregate the tool_calls table; fall back to summing ``tool_call_count`` when extraction is off."""
    message = store.models.message
    tool_call = store.models.tool_call
    grouped = sa.select(
        tool_call.tool_name,
        sa.func.count().label('call_count'),
        sa.func.count(sa.case((tool_call.status == 'returned', 1))).label('returned_count'),
        sa.func.count(sa.case((tool_call.status == 'error', 1))).label('error_count'),
        sa.func.count(sa.case((tool_call.status == 'unanswered', 1))).label('unanswered_count'),
    ).group_by(tool_call.tool_name)
    if since is not None:
        grouped = grouped.where(sa.or_(tool_call.called_at >= since, tool_call.called_at.is_(None)))
    async with store.sessions.scope(session) as (db, _owned):
        rows = (await db.execute(grouped)).all()
        if rows:
            usage = [
                ToolUsage(
                    tool_name=row.tool_name,
                    call_count=_as_int(row.call_count),
                    returned_count=_as_int(row.returned_count),
                    error_count=_as_int(row.error_count),
                    unanswered_count=_as_int(row.unanswered_count),
                )
                for row in rows
            ]
            usage.sort(key=lambda item: (-item.call_count, item.tool_name))
            return usage
        fallback = sa.select(sa.func.sum(message.tool_call_count)).where(
            *_time_bounds(message.message_timestamp, since, None)
        )
        total = _as_int((await db.execute(fallback)).scalar())
    if total == 0:
        return []
    return [ToolUsage(tool_name='*', call_count=total, returned_count=0, error_count=0, unanswered_count=0)]


async def cost_report(
    store: SQLAlchemyChatStore,
    *,
    group_by: Literal['day', 'model', 'conversation'],
    since: datetime | None = None,
    until: datetime | None = None,
    session: AsyncSession | None = None,
) -> list[CostRow]:
    message = store.models.message
    conversation = store.models.conversation
    conditions: list[sa.ColumnElement[bool]] = [
        _qualifying_response(message),
        *_time_bounds(message.message_timestamp, since, until),
    ]
    if group_by == 'day':
        key = sa.func.date(message.message_timestamp).label('group_key')
        conditions.append(message.message_timestamp.is_not(None))
    elif group_by == 'model':
        key = message.model_name.label('group_key')
    else:
        key = conversation.conversation_key.label('group_key')
    statement = (
        sa.select(
            key,
            sa.func.count().label('model_requests'),
            sa.func.sum(message.input_tokens).label('input_tokens'),
            sa.func.sum(message.output_tokens).label('output_tokens'),
            sa.func.sum(message.cost).label('cost'),
        )
        .where(*conditions)
        .group_by(key)
    )
    if group_by == 'conversation':
        statement = statement.select_from(message).join(
            conversation, onclause=message.conversation_pk == conversation.id
        )
    async with store.sessions.scope(session) as (db, _owned):
        rows = (await db.execute(statement)).all()

    def _group_label(value: object) -> str:
        if group_by == 'day':
            return _as_day(value).isoformat()
        if value is None:
            return '(unknown)'
        return str(value)

    report = [
        CostRow(
            group=_group_label(row.group_key),
            model_requests=_as_int(row.model_requests),
            input_tokens=_as_int(row.input_tokens),
            output_tokens=_as_int(row.output_tokens),
            cost=_as_cost(row.cost),
        )
        for row in rows
    ]

    def _sort_key(row: CostRow) -> tuple[int, Decimal, str]:
        if row.cost is None:
            return (1, Decimal(0), row.group)
        return (0, -row.cost, row.group)

    report.sort(key=_sort_key)
    return report
