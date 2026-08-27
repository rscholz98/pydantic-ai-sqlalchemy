"""Package exceptions."""

from __future__ import annotations

__all__ = ['ConversationNotFoundError', 'SequenceAllocationError', 'StoreError']


class StoreError(Exception):
    """Base error for all store failures, including misconfiguration (no bind and no session)."""


class ConversationNotFoundError(StoreError):
    """Raised when an operation targets a conversation key that does not exist."""


class SequenceAllocationError(StoreError):
    """Raised when message sequence allocation keeps colliding after all retries."""
