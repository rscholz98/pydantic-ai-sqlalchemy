"""Async SQLAlchemy storage adapter for Pydantic AI: archive chats, runs, usage and tool calls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._base import Base
from ._exceptions import ConversationNotFoundError, SequenceAllocationError, StoreError
from ._models import (
    DEFAULT_SPEC,
    BaseConversation,
    BaseMessage,
    BaseRun,
    BaseToolCall,
    DefaultConversation,
    DefaultMessage,
    DefaultRun,
    DefaultToolCall,
    ModelSpec,
)
from ._store import SQLAlchemyChatStore
from ._types import (
    ConversationRecord,
    ConversationUsage,
    CostRow,
    DailyUsage,
    MessageDenorm,
    ModelUsage,
    PurgeReport,
    RunRecord,
    RunStats,
    SaveResult,
    ToolUsage,
)

if TYPE_CHECKING:
    from ._hooks import build_on_complete
    from ._step_store import SQLAlchemyStepStore

__all__ = [
    'DEFAULT_SPEC',
    'Base',
    'BaseConversation',
    'BaseMessage',
    'BaseRun',
    'BaseToolCall',
    'ConversationNotFoundError',
    'ConversationRecord',
    'ConversationUsage',
    'CostRow',
    'DailyUsage',
    'DefaultConversation',
    'DefaultMessage',
    'DefaultRun',
    'DefaultToolCall',
    'MessageDenorm',
    'ModelSpec',
    'ModelUsage',
    'PurgeReport',
    'RunRecord',
    'RunStats',
    'SQLAlchemyChatStore',
    'SQLAlchemyStepStore',
    'SaveResult',
    'SequenceAllocationError',
    'StoreError',
    'ToolUsage',
    'build_on_complete',
]


def __getattr__(name: str) -> object:
    # Lazy imports keep optional integration surfaces (and their dependencies) out of the
    # default import path, mirroring the pydantic-ai-harness convention.
    if name == 'SQLAlchemyStepStore':
        from ._step_store import SQLAlchemyStepStore

        return SQLAlchemyStepStore
    if name == 'build_on_complete':
        from ._hooks import build_on_complete

        return build_on_complete
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
