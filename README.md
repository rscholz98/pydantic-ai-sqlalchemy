# pydantic-ai-sqlalchemy

Async SQLAlchemy 2.0 archival store for [Pydantic AI](https://github.com/pydantic/pydantic-ai)
conversations, built for platform-owner analytics.

> Status: under construction. The write, read and analytics method bodies land in sibling
> work-unit PRs; until they merge, `examples/basic.py` prints pending notes for those calls.

## What it is and why

Most chat persistence helpers are short-term session memory: they hold the last few turns so the
next request has context. This package solves a different problem: the **long-term archive** a
platform owner needs once agents run in production. Every conversation, run, message and tool
call is written to relational tables you can query with plain SQL (or through the bundled
analytics methods) to answer questions like "what did model X cost us last week", "which tools
fail most often" and "which conversations burn the most tokens".

Two design rules keep it consistent with how pydantic-ai itself stores messages:

- **Full fidelity.** Each `ModelMessage` is stored as one row whose payload is written with
  `pydantic_ai.messages.ModelMessagesTypeAdapter` semantics, so everything round-trips byte for
  byte: `BinaryContent` (the adapter's base64 bytes config is preserved), timestamps, usage,
  provider details. Parts are never shredded into per-kind tables, so new pydantic-ai message
  kinds keep working without schema changes.
- **Denormalize only query keys.** Alongside the blob, each row carries real columns for the
  things you filter and aggregate on: model name, provider, token counts, cost, finish reason,
  tool-call count. Analytics never has to parse JSON.

## Features

- One row per `ModelMessage`: full-fidelity JSON blob plus denormalized usage/cost/model columns
- Run and conversation rollups (tokens, cost, duration, message counts) maintained on write
- Content-hash dedup: re-saving the same history (retried hooks, durable executors) normally writes nothing
- Per-conversation sequence numbers with a `unique(conversation_pk, seq)` guarantee
- `load_history()` feeds straight into `Agent.run(message_history=...)`; `max_turns` never splits tool pairs
- Partial-run capture for cancelled or failed runs (`save_partial_run`), with deferred-tool closing
- Analytics: usage by day/model/conversation, run stats, tool usage, cost reports
- Retention: `delete_conversation`, `purge_older_than` (with dry run); portability: JSONL export/import
- Bring your own models: subclass the abstract bases onto your app's declarative `Base` (tenancy columns, custom table names, Alembic migrations)
- Caller-session contract: pass your own `AsyncSession` and the store flushes but never commits
- Optional [pydantic-ai-harness](https://github.com/rscholz98/pydantic-ai-harness) interop: `HistorySource` seam and a full SQL `StepStore`
- Works on SQLite (`aiosqlite`) and PostgreSQL (`asyncpg`); JSON columns use `JSONB` on PostgreSQL

## Install

Not yet on PyPI; install from the repository:

```bash
# pip
pip install 'pydantic-ai-sqlalchemy[sqlite] @ git+https://github.com/rscholz98/pydantic-ai-sqlalchemy'

# uv
uv add 'pydantic-ai-sqlalchemy[sqlite] @ git+https://github.com/rscholz98/pydantic-ai-sqlalchemy'
```

Extras:

| Extra | Pulls in | Use for |
| --- | --- | --- |
| `sqlite` | `aiosqlite` | local development, single-node deployments |
| `postgres` | `asyncpg` | production PostgreSQL |
| `harness` | `pydantic-ai-harness` | step persistence and conversation search interop (compatibility note: the host must be on `pydantic-ai-slim>=2.33`; the extra does not pin slim itself) |

## Quickstart

```python
import asyncio

from pydantic_ai import Agent
from sqlalchemy.ext.asyncio import create_async_engine

from pydantic_ai_sqlalchemy import SQLAlchemyChatStore

agent = Agent('openai:gpt-5-mini')


async def main() -> None:
    engine = create_async_engine('sqlite+aiosqlite:///chats.db')
    store = SQLAlchemyChatStore(engine)
    await store.create_tables()

    # First turn: run, then archive the run.
    result = await agent.run('What is the capital of France?')
    await store.save_run(result, conversation_key='user-42', agent_name='geo-bot')

    # Later (any process): reload the archive and continue the conversation.
    history = await store.load_history(conversation_key='user-42')
    followup = await agent.run('And its population?', message_history=history)
    await store.save_run(followup, conversation_key='user-42', agent_name='geo-bot')

    print(await store.get_transcript(conversation_key='user-42'))


asyncio.run(main())
```

`conversation_key` is your stable identifier for a thread (user id, ticket id, session id); it
defaults to the run's own `conversation_id` when omitted. `save_run(..., only_new=True)` (the
default) appends `result.new_messages()`, so saving a follow-up run never duplicates the history
it was started from. On top of that, saves deduplicate per (conversation, run) on a content hash
of each message, so a re-fired save (a retried hook, a durable executor replaying a step)
normally writes nothing. This is best-effort read-then-write dedup, not a hard guarantee: two
concurrent fires of the same save can still race each other; the `unique(conversation_pk, seq)`
constraint then prevents corrupted ordering, not the duplicate itself.

For hook-style integration (for example `UIAdapter.dispatch_request(on_complete=...)`) build the
callback once:

```python
from pydantic_ai_sqlalchemy import build_on_complete

on_complete = build_on_complete(store, conversation_key='user-42', agent_name='geo-bot')
await on_complete(result)  # safe to fire more than once
```

A runnable end-to-end script that needs no API key lives in
[`examples/basic.py`](examples/basic.py): `uv run python examples/basic.py`.

## Schema overview

| Table | One row per | What it holds |
| --- | --- | --- |
| `pai_conversations` | conversation | your `conversation_key`, activity timestamps, rollups (message count, tokens, cost) |
| `pai_messages` | `ModelMessage` | the full-fidelity message blob plus denormalized query columns (model, provider, tokens, cost, finish reason, tool-call count), `unique(conversation_pk, seq)` ordering |
| `pai_runs` | `Agent.run` call | run state, duration, model, per-run token/cost/message rollups |
| `pai_tool_calls` | extracted tool call | tool name, args, status (`called`/`returned`/`error`/`unanswered`); populated when the store is built with `extract_tool_calls=True` |
| `pai_sp_runs`, `pai_sp_events`, `pai_sp_snapshots`, `pai_sp_tool_effects` | step-store record | backing tables for `SQLAlchemyStepStore` (pydantic-ai-harness step persistence); independent of the chat tables |

## Analytics

All analytics read the denormalized columns, so they stay fast even when message blobs are large:

```python
from datetime import datetime, timedelta, timezone

since = datetime.now(timezone.utc) - timedelta(days=30)

for day in await store.usage_by_day(since=since):
    print(day.day, day.model_requests, day.input_tokens, day.output_tokens, day.cost)

for row in await store.cost_report(group_by='model', since=since):
    print(row.group, row.model_requests, row.cost)
```

Also available: `usage_by_model()`, `usage_by_conversation()`, `run_stats()` (run counts and
average duration per state) and `tool_usage()` (call/returned/error/unanswered counts per tool).
Every method returns typed frozen dataclass rows (`DailyUsage`, `ModelUsage`, `CostRow`, ...),
and cost values are `Decimal`.

## Retention and JSONL export

```python
from datetime import datetime, timedelta, timezone

cutoff = datetime.now(timezone.utc) - timedelta(days=365)

report = await store.purge_older_than(cutoff, dry_run=True)   # counts only, deletes nothing
print(report.conversations, report.messages, report.runs, report.tool_calls)

await store.export_jsonl('archive.jsonl', conversation_keys=None, since=None)  # everything
await store.purge_older_than(cutoff)                          # now actually delete
await store.delete_conversation(conversation_key='user-42')   # targeted delete, returns row count
```

`export_jsonl` writes adapter-faithful payloads, so `import_jsonl` restores conversations with
byte-identical message content (useful for backups, migrations between databases, or offline
analysis).

## Caller-session contract

Every read and write method accepts `session=` (the schema helpers `create_tables` and
`drop_tables` are the exception: they take no session and need a bind, so bindless
row-level-security setups should create the schema via Alembic or a one-off engine instead).
The rule:

- **You pass a session**: the store only flushes it. It never commits, rolls back or closes your
  session; your transaction boundaries (and rollback on error) stay fully yours.
- **You pass no session**: the store opens one from its bind and commits on success.

This makes the store safe inside host transactions, and it is exactly what row-level-security
setups need: open the session, arm your RLS session variables (tenant id and friends), hand the
session to the store, commit yourself.

```python
from sqlalchemy import text

async with app_sessionmaker() as session:
    await session.execute(text("SELECT set_config('app.current_tenant', :t, true)"), {'t': tenant_id})
    await store.save_run(result, conversation_key=key, session=session)
    await session.commit()
```

A store built with `SQLAlchemyChatStore(None)` has no bind at all and requires a session on every
call, which turns "forgot to pass the tenant session" into an immediate `StoreError`. For
hook-style saving in scoped setups, `build_on_complete(store, session_factory=...)` accepts an
async context manager factory that yields the prepared session.

## Bring your own models (host integration)

The default `pai_*` tables live on the package's own declarative `Base`. Host applications can
instead subclass the abstract bases onto **their own** `Base` to add columns (tenancy, foreign
keys) and rename tables. The abstract bases carry all columns, indexes and constraints; the
`__pai_conversation_table__` / `__pai_message_table__` classvars tell dependent tables where the
renamed foreign-key targets live.

```python
import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from pydantic_ai_sqlalchemy import (
    BaseConversation,
    BaseMessage,
    BaseRun,
    BaseToolCall,
    ModelSpec,
    SQLAlchemyChatStore,
)


class AppBase(DeclarativeBase):  # your application's existing declarative base
    pass


# The store never writes host-specific columns, so let the database fill them,
# for example from an RLS session variable:
def tenant_column() -> Mapped[str]:
    return mapped_column(
        sa.String(64), nullable=False, index=True, server_default=sa.text("current_setting('app.current_tenant', true)")
    )


class ChatConversation(BaseConversation, AppBase):
    __tablename__ = 'app_chat_conversations'
    tenant_id: Mapped[str] = tenant_column()


class ChatMessage(BaseMessage, AppBase):
    __tablename__ = 'app_chat_messages'
    __pai_conversation_table__ = 'app_chat_conversations'
    tenant_id: Mapped[str] = tenant_column()


class ChatRun(BaseRun, AppBase):
    __tablename__ = 'app_chat_runs'
    __pai_conversation_table__ = 'app_chat_conversations'
    tenant_id: Mapped[str] = tenant_column()


class ChatToolCall(BaseToolCall, AppBase):
    __tablename__ = 'app_chat_tool_calls'
    __pai_conversation_table__ = 'app_chat_conversations'
    __pai_message_table__ = 'app_chat_messages'
    tenant_id: Mapped[str] = tenant_column()


store = SQLAlchemyChatStore(
    None,  # no bind: every call must receive the host's (tenant-scoped) session
    models=ModelSpec(conversation=ChatConversation, message=ChatMessage, run=ChatRun, tool_call=ChatToolCall),
)
```

Because the concrete models are registered on your `Base`, **Alembic autogenerate picks the
tables up** like any other app table; skip `create_tables()` and let your normal migration flow
own the DDL.

## Harness interop

With the `harness` extra installed, the package plugs into
[pydantic-ai-harness](https://github.com/rscholz98/pydantic-ai-harness). Compatibility note:
the harness itself requires `pydantic-ai-slim>=2.33`, and the extra does not pin slim; hosts
below that version can use the core store but not the harness surfaces.

- `SQLAlchemyChatStore` structurally satisfies the harness `HistorySource` protocol
  (`list_runs()` / `run_history()`), so archived conversations feed harness conversation search.
- `SQLAlchemyStepStore` implements the full harness `StepStore` protocol over SQL tables
  (`pai_sp_*`): run registry, step events, continuable snapshots and tool-effect records, for
  durable or resumable agent execution backed by the same database.

```python
from pydantic_ai_sqlalchemy import SQLAlchemyStepStore

step_store = SQLAlchemyStepStore(engine)
await step_store.create_tables()
```

## Supported versions

- Python 3.10 to 3.14
- `pydantic-ai-slim>=2.31` (hosts using the `harness` extra must be on `>=2.33` themselves; the extra does not pin slim)
- `SQLAlchemy>=2.0.30`
- Databases: SQLite via `aiosqlite`, PostgreSQL via `asyncpg` (JSONB, tested against PostgreSQL 16 in CI)

## Development

```bash
git clone https://github.com/rscholz98/pydantic-ai-sqlalchemy
cd pydantic-ai-sqlalchemy
uv sync --group dev
```

Quality gates (all must pass, mirrored in CI):

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

`uv run python examples/basic.py` should also run and exit 0.

Tests always run on in-memory SQLite. To also run them against PostgreSQL locally, start the
postgres service from the repo's `docker-compose.yml` and point the test suite at it:

```bash
docker compose up -d postgres
export PAI_SQLA_TEST_PG_DSN='postgresql+asyncpg://postgres:postgres@localhost:5432/pai_test'
uv run pytest
```

## Roadmap

- PyPI release
- Media externalization: store large `BinaryContent` out of row (object storage) with references in the blob
- Session-tree replay on the reserved `parent_id` column (append-only log with branch replay)
- Shipped Alembic migration scripts for the default tables

## License

[MIT](LICENSE)
