"""Retention: purge and per-conversation deletion.

Implemented by work unit U7. Signatures are frozen; see ``_store.SQLAlchemyChatStore``.

Implementation contract:
- ``purge_older_than`` keys off ``last_activity_at``. A ``dry_run`` counts affected rows per
  table without deleting; a real purge deletes child-first and reports the exact rowcounts of
  the delete statements themselves.
- Children are deleted explicitly rather than via FK CASCADE because SQLite hosts often run
  without the ``foreign_keys`` pragma. For the same reason the self-referencing
  ``message.parent_id`` SET NULL semantics only apply on SQLite when the pragma is enabled;
  retention is unaffected because all messages of a conversation are deleted together.
- ``delete_conversation`` returns the number of deleted message rows; raise
  ``ConversationNotFoundError`` for unknown keys.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import sqlalchemy as sa
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from ._exceptions import ConversationNotFoundError
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
    conversation = store.models.conversation
    matching_ids = sa.select(conversation.id).where(conversation.conversation_key == conversation_key)

    async with store.sessions.scope(session) as (db, _owned):
        report = await _delete_matching(db, store, matching_ids)
        if report.conversations == 0:
            raise ConversationNotFoundError(f'conversation with key {conversation_key!r} does not exist')
        return report.messages


async def purge_older_than(
    store: SQLAlchemyChatStore,
    cutoff: datetime,
    *,
    dry_run: bool = False,
    session: AsyncSession | None = None,
) -> PurgeReport:
    conversation = store.models.conversation
    message = store.models.message
    run = store.models.run
    tool_call = store.models.tool_call
    matching_ids = sa.select(conversation.id).where(
        conversation.last_activity_at.is_not(None), conversation.last_activity_at < cutoff
    )

    async with store.sessions.scope(session) as (db, _owned):
        if dry_run:
            return PurgeReport(
                conversations=await _count(db, conversation, conversation.id.in_(matching_ids)),
                messages=await _count(db, message, message.conversation_pk.in_(matching_ids)),
                runs=await _count(db, run, run.conversation_pk.in_(matching_ids)),
                tool_calls=await _count(db, tool_call, tool_call.conversation_pk.in_(matching_ids)),
                dry_run=True,
            )
        return await _delete_matching(db, store, matching_ids)


async def _count(db: AsyncSession, entity: type[object], criterion: sa.ColumnElement[bool]) -> int:
    value = await db.scalar(sa.select(sa.func.count()).select_from(entity).where(criterion))
    return value or 0


async def _delete_matching(
    db: AsyncSession, store: SQLAlchemyChatStore, matching_ids: sa.Select[tuple[uuid.UUID]]
) -> PurgeReport:
    """Delete all rows of the matching conversations child-first; report the deleted rowcounts.

    Only the four store tables are covered: hosts that add their own FK-linked tables onto the
    store schema must delete those rows themselves before calling retention.
    """
    conversation = store.models.conversation
    message = store.models.message
    run = store.models.run
    tool_call = store.models.tool_call

    tool_calls_deleted = await _delete_rows(db, tool_call, tool_call.conversation_pk.in_(matching_ids))
    runs_deleted = await _delete_rows(db, run, run.conversation_pk.in_(matching_ids))
    messages_deleted = await _delete_rows(db, message, message.conversation_pk.in_(matching_ids))
    conversations_deleted = await _delete_rows(db, conversation, conversation.id.in_(matching_ids))
    return PurgeReport(
        conversations=conversations_deleted,
        messages=messages_deleted,
        runs=runs_deleted,
        tool_calls=tool_calls_deleted,
        dry_run=False,
    )


async def _delete_rows(db: AsyncSession, entity: type[object], criterion: sa.ColumnElement[bool]) -> int:
    result = cast(
        'CursorResult[Any]',
        await db.execute(sa.delete(entity).where(criterion).execution_options(synchronize_session=False)),
    )
    return result.rowcount
