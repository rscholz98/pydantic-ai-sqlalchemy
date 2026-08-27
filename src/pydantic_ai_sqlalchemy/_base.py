"""Declarative base, naming convention and cross-dialect type helpers."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION: dict[str, str] = {
    'ix': 'ix_%(column_0_label)s',
    'uq': 'uq_%(table_name)s_%(column_0_name)s',
    'ck': 'ck_%(table_name)s_%(constraint_name)s',
    'fk': 'fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s',
    'pk': 'pk_%(table_name)s',
}


class Base(DeclarativeBase):
    """Package-owned declarative base used by the default models.

    Host applications that bind the abstract model bases to their own ``Base`` never touch this class.
    """

    metadata = sa.MetaData(naming_convention=NAMING_CONVENTION)


def json_variant() -> sa.types.TypeEngine[object]:
    """Portable JSON column type: JSONB on PostgreSQL, JSON elsewhere."""
    return sa.JSON().with_variant(postgresql.JSONB(), 'postgresql')


def autoincrement_pk_variant() -> sa.types.TypeEngine[int]:
    """BigInteger PK that degrades to INTEGER on SQLite so rowid autoincrement works."""
    return sa.BigInteger().with_variant(sa.Integer(), 'sqlite')
