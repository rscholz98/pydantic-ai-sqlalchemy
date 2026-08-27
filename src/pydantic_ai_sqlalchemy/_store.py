"""``SQLAlchemyChatStore``: the public store facade.

Every method delegates to a per-concern private module; the signatures here are the
frozen public API that the parallel work units implement against. Do not change them.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai.messages import ModelMessage
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from . import _analytics, _export, _history_source, _partial, _read, _retention, _write
from ._models import DEFAULT_SPEC, ModelSpec
from ._serialize import JsonSanitizer
from ._session import SessionProvider
from ._types import (
    ConversationRecord,
    ConversationUsage,
    CostRow,
    DailyUsage,
    ModelUsage,
    PurgeReport,
    RunRecord,
    RunStats,
    SaveResult,
    ToolUsage,
)

if TYPE_CHECKING:
    import os
    from typing import IO

    from pydantic_ai.run import AgentRunResult

__all__ = ['SQLAlchemyChatStore']


class SQLAlchemyChatStore:
    """Async SQLAlchemy archive for pydantic-ai conversations, built for analytics.

    Stores one row per ``ModelMessage`` (full-fidelity JSON blob written with
    ``ModelMessagesTypeAdapter`` semantics) plus denormalized usage/cost/model columns,
    with run and conversation rollups.
    """

    def __init__(
        self,
        bind: AsyncEngine | async_sessionmaker[AsyncSession] | None = None,
        *,
        models: ModelSpec | None = None,
        extract_tool_calls: bool = False,
        trusted_history: bool = True,
        json_sanitizer: JsonSanitizer | None = None,
    ) -> None:
        self.models = models if models is not None else DEFAULT_SPEC
        self.sessions = SessionProvider(bind)
        self.extract_tool_calls_enabled = extract_tool_calls
        self.trusted_history = trusted_history
        self.json_sanitizer = json_sanitizer

    # -- schema management ------------------------------------------------------------------

    async def create_tables(self) -> None:
        """Create the four store tables (host apps on their own Base typically use Alembic instead)."""
        tables = list(self.models.tables())
        metadata = tables[0].metadata
        async with self.sessions.scope() as (session, _owned):
            connection = await session.connection()
            await connection.run_sync(lambda sync_connection: metadata.create_all(sync_connection, tables=tables))

    async def drop_tables(self) -> None:
        tables = list(self.models.tables())
        metadata = tables[0].metadata
        async with self.sessions.scope() as (session, _owned):
            connection = await session.connection()
            await connection.run_sync(lambda sync_connection: metadata.drop_all(sync_connection, tables=tables))

    # -- write ------------------------------------------------------------------------------

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
        """Persist a finished run; ``conversation_key`` defaults to the run's ``conversation_id``."""
        return await _write.save_run(
            self,
            result=result,
            conversation_key=conversation_key,
            agent_name=agent_name,
            only_new=only_new,
            final_message_id=final_message_id,
            session=session,
        )

    async def save_messages(
        self,
        messages: Sequence[ModelMessage],
        *,
        conversation_key: str,
        run_id: str | None = None,
        agent_name: str | None = None,
        final_message_id: uuid.UUID | None = None,
        session: AsyncSession | None = None,
    ) -> SaveResult:
        return await _write.save_messages(
            self,
            messages=messages,
            conversation_key=conversation_key,
            run_id=run_id,
            agent_name=agent_name,
            final_message_id=final_message_id,
            session=session,
        )

    async def save_partial_run(
        self,
        messages: Sequence[ModelMessage],
        *,
        conversation_key: str,
        run_id: str | None = None,
        agent_name: str | None = None,
        mark_interrupted: bool = True,
        close_deferred_tools: bool = True,
        session: AsyncSession | None = None,
    ) -> SaveResult:
        """Persist a cancelled or failed run (e.g. ``RunCancelled.new_messages()``) safely."""
        return await _partial.save_partial_run(
            self,
            messages=messages,
            conversation_key=conversation_key,
            run_id=run_id,
            agent_name=agent_name,
            mark_interrupted=mark_interrupted,
            close_deferred_tools=close_deferred_tools,
            session=session,
        )

    # -- read -------------------------------------------------------------------------------

    async def load_history(
        self,
        *,
        conversation_key: str,
        max_turns: int | None = None,
        sanitize: bool | None = None,
        session: AsyncSession | None = None,
    ) -> list[ModelMessage]:
        """Load history for ``Agent.run(message_history=...)``; ``max_turns`` never splits tool pairs."""
        return await _read.load_history(
            self, conversation_key=conversation_key, max_turns=max_turns, sanitize=sanitize, session=session
        )

    async def get_transcript(self, *, conversation_key: str, session: AsyncSession | None = None) -> str:
        return await _read.get_transcript(self, conversation_key=conversation_key, session=session)

    async def list_conversations(
        self,
        *,
        since: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
        session: AsyncSession | None = None,
    ) -> list[ConversationRecord]:
        return await _read.list_conversations(self, since=since, limit=limit, offset=offset, session=session)

    # -- harness HistorySource seam ---------------------------------------------------------

    async def list_runs(
        self, *, conversation_key: str | None = None, session: AsyncSession | None = None
    ) -> list[RunRecord]:
        return await _history_source.list_runs(self, conversation_key=conversation_key, session=session)

    async def run_history(self, *, run_id: str, session: AsyncSession | None = None) -> list[ModelMessage]:
        return await _history_source.run_history(self, run_id=run_id, session=session)

    # -- retention --------------------------------------------------------------------------

    async def delete_conversation(self, *, conversation_key: str, session: AsyncSession | None = None) -> int:
        return await _retention.delete_conversation(self, conversation_key=conversation_key, session=session)

    async def purge_older_than(
        self, cutoff: datetime, *, dry_run: bool = False, session: AsyncSession | None = None
    ) -> PurgeReport:
        return await _retention.purge_older_than(self, cutoff, dry_run=dry_run, session=session)

    # -- export -----------------------------------------------------------------------------

    async def export_jsonl(
        self,
        destination: IO[bytes] | os.PathLike[str] | str,
        *,
        conversation_keys: Sequence[str] | None = None,
        since: datetime | None = None,
        session: AsyncSession | None = None,
    ) -> int:
        return await _export.export_jsonl(
            self, destination, conversation_keys=conversation_keys, since=since, session=session
        )

    async def import_jsonl(
        self, source: IO[bytes] | os.PathLike[str] | str, *, session: AsyncSession | None = None
    ) -> int:
        return await _export.import_jsonl(self, source, session=session)

    # -- analytics --------------------------------------------------------------------------

    async def usage_by_day(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        session: AsyncSession | None = None,
    ) -> list[DailyUsage]:
        return await _analytics.usage_by_day(self, since=since, until=until, session=session)

    async def usage_by_model(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        session: AsyncSession | None = None,
    ) -> list[ModelUsage]:
        return await _analytics.usage_by_model(self, since=since, until=until, session=session)

    async def usage_by_conversation(
        self,
        *,
        since: datetime | None = None,
        limit: int = 100,
        session: AsyncSession | None = None,
    ) -> list[ConversationUsage]:
        return await _analytics.usage_by_conversation(self, since=since, limit=limit, session=session)

    async def run_stats(self, *, since: datetime | None = None, session: AsyncSession | None = None) -> list[RunStats]:
        return await _analytics.run_stats(self, since=since, session=session)

    async def tool_usage(
        self, *, since: datetime | None = None, session: AsyncSession | None = None
    ) -> list[ToolUsage]:
        return await _analytics.tool_usage(self, since=since, session=session)

    async def cost_report(
        self,
        *,
        group_by: Literal['day', 'model', 'conversation'],
        since: datetime | None = None,
        until: datetime | None = None,
        session: AsyncSession | None = None,
    ) -> list[CostRow]:
        return await _analytics.cost_report(self, group_by=group_by, since=since, until=until, session=session)
