# pydantic-ai-sqlalchemy

Async SQLAlchemy 2.0 storage adapter for [Pydantic AI](https://github.com/pydantic/pydantic-ai).

Persists agent conversations as a long-term archive for platform-owner analytics: full-fidelity
message history (round-trip safe via `ModelMessagesTypeAdapter`), plus denormalized usage, cost,
model and tool-call columns you can query with plain SQL.

Status: under construction. Full documentation lands with the first complete build.

## Install

```bash
pip install 'pydantic-ai-sqlalchemy[sqlite] @ git+https://github.com/rscholz98/pydantic-ai-sqlalchemy'
```

## Quick look

```python
from sqlalchemy.ext.asyncio import create_async_engine
from pydantic_ai_sqlalchemy import SQLAlchemyChatStore

engine = create_async_engine('sqlite+aiosqlite:///chats.db')
store = SQLAlchemyChatStore(engine)
await store.create_tables()

result = await agent.run('Hello!')
await store.save_run(result)

history = await store.load_history(conversation_key=result.conversation_id)
```
