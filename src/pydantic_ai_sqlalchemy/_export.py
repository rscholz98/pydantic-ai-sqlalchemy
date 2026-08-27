"""JSONL export and import.

Implemented by work unit U8. Signatures are frozen; see ``_store.SQLAlchemyChatStore``.

Line format: ``{"conversation_key": str, "seq": int, "run_id": str | null, "message": {...}}``
with ``message`` being the stored blob. Import validates every line's message through
``ModelMessagesTypeAdapter`` before inserting and allocates fresh sequence numbers per
conversation; both directions stream (no full-table loads into memory).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime
from typing import IO, TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

if TYPE_CHECKING:
    from ._store import SQLAlchemyChatStore

__all__ = ['export_jsonl', 'import_jsonl']


async def export_jsonl(
    store: SQLAlchemyChatStore,
    destination: IO[bytes] | os.PathLike[str] | str,
    *,
    conversation_keys: Sequence[str] | None = None,
    since: datetime | None = None,
    session: AsyncSession | None = None,
) -> int:
    """Write matching messages as JSONL; returns the number of lines written."""
    raise NotImplementedError('implemented in unit U8')


async def import_jsonl(
    store: SQLAlchemyChatStore,
    source: IO[bytes] | os.PathLike[str] | str,
    *,
    session: AsyncSession | None = None,
) -> int:
    """Read a JSONL export and persist it; returns the number of messages imported."""
    raise NotImplementedError('implemented in unit U8')
