"""Retention: purge and per-conversation deletion.

Implemented by work unit U7. Signatures are frozen; see ``_store.SQLAlchemyChatStore``.

Implementation contract:
- ``purge_older_than`` keys off ``last_activity_at``; count affected rows per table first
  (that is the ``dry_run`` report), then delete rows child-first (tool calls, runs,
  messages, conversations). Children are deleted explicitly rather than via FK CASCADE
  because SQLite hosts often run without the ``foreign_keys`` pragma.
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
        exists = await db.scalar(matching_ids)
        if exists is None:
            raise ConversationNotFoundError(f'conversation with key {conversation_key!r} does not exist')
        deleted_messages = await _delete_matching(db, store, matching_ids)
        await db.flush()
        return deleted_messages


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
        report = PurgeReport(
            conversations=await _count(db, conversation, conversation.id.in_(matching_ids)),
            messages=await _count(db, message, message.conversation_pk.in_(matching_ids)),
            runs=await _count(db, run, run.conversation_pk.in_(matching_ids)),
            tool_calls=await _count(db, tool_call, tool_call.conversation_pk.in_(matching_ids)),
            dry_run=dry_run,
        )
        if dry_run:
            return report
        await _delete_matching(db, store, matching_ids)
        await db.flush()
        return report


async def _count(db: AsyncSession, entity: type[object], criterion: sa.ColumnElement[bool]) -> int:
    value = await db.scalar(sa.select(sa.func.count()).select_from(entity).where(criterion))
    return value or 0


async def _delete_matching(
    db: AsyncSession, store: SQLAlchemyChatStore, matching_ids: sa.Select[tuple[uuid.UUID]]
) -> int:
    """Delete all rows of the matching conversations child-first; return the deleted message count."""
    conversation = store.models.conversation
    message = store.models.message
    run = store.models.run
    tool_call = store.models.tool_call

    await db.execute(
        sa.delete(tool_call)
        .where(tool_call.conversation_pk.in_(matching_ids))
        .execution_options(synchronize_session=False)
    )
    await db.execute(
        sa.delete(run).where(run.conversation_pk.in_(matching_ids)).execution_options(synchronize_session=False)
    )
    message_result = cast(
        'CursorResult[Any]',
        await db.execute(
            sa.delete(message)
            .where(message.conversation_pk.in_(matching_ids))
            .execution_options(synchronize_session=False)
        ),
    )
    await db.execute(
        sa.delete(conversation).where(conversation.id.in_(matching_ids)).execution_options(synchronize_session=False)
    )
    return message_result.rowcount
