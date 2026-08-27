"""End-to-end example: run a pydantic-ai agent on a canned FunctionModel (no API key needed),
archive the conversation with SQLAlchemyChatStore, reload it and run a follow-up turn.

Run it with:

    uv run python examples/basic.py

Some store methods are implemented in parallel work units; while those are still open on this
branch the script prints a note instead of failing, and always exits 0.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from pydantic_ai_sqlalchemy import SQLAlchemyChatStore, build_on_complete

_T = TypeVar('_T')

PENDING = 'not yet implemented on this branch (pending sibling PR)'


async def call_or_note(label: str, awaitable: Awaitable[_T]) -> _T | None:
    """Await one store call; while the sibling unit implementing it is unmerged, print a note instead."""
    try:
        return await awaitable
    except NotImplementedError:
        print(f'{label}: {PENDING}')
        return None


# Keep this canned model in sync with the one in tests/test_e2e.py (same answer, same model name).
def canned_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A deterministic stand-in for a real LLM: one tool call, then a text answer."""
    seen_tool_return = any(part.part_kind == 'tool-return' for message in messages for part in message.parts)
    if info.function_tools and not seen_tool_return:
        return ModelResponse(
            parts=[ToolCallPart(tool_name='roll_dice', args={})], usage=RequestUsage(input_tokens=52, output_tokens=6)
        )
    return ModelResponse(
        parts=[TextPart('You rolled a 4. Lucky you!')], usage=RequestUsage(input_tokens=61, output_tokens=9)
    )


def build_agent() -> Agent:
    agent = Agent(FunctionModel(canned_model, model_name='canned-dice-model'))

    @agent.tool_plain
    def roll_dice() -> int:
        """Roll a perfectly fair die."""
        return 4

    return agent


async def main() -> None:
    agent = build_agent()
    engine = create_async_engine('sqlite+aiosqlite://', poolclass=StaticPool)
    store = SQLAlchemyChatStore(engine)

    try:
        await store.create_tables()
        print('created store tables (in-memory SQLite)')

        # First turn: run the agent, then archive the run via the on_complete hook.
        result = await agent.run('Roll the dice!')
        conversation_key = f'example:{result.conversation_id}'
        print(f'first answer: {result.output}')

        async def save_first_run() -> None:
            on_complete = build_on_complete(store, conversation_key=conversation_key, agent_name='dice-agent')
            await on_complete(result)
            print(f'saved run {result.run_id} into conversation {conversation_key!r}')

        await call_or_note('save (build_on_complete / save_run)', save_first_run())

        # Second turn: reload the archived history and continue the conversation from it.
        history = await call_or_note('load_history', store.load_history(conversation_key=conversation_key))
        if history is None:
            return
        print(f'loaded {len(history)} archived messages')

        followup = await agent.run('Was that a good roll?', message_history=history)
        print(f'follow-up answer: {followup.output}')
        await call_or_note(
            'save_run', store.save_run(followup, conversation_key=conversation_key, agent_name='dice-agent')
        )

        # Read the archive back: a human-readable transcript plus per-model usage rollups.
        transcript = await call_or_note('get_transcript', store.get_transcript(conversation_key=conversation_key))
        if transcript is not None:
            print(f'\ntranscript:\n{transcript}')

        rows = await call_or_note('usage_by_model', store.usage_by_model())
        if rows is not None:
            print('\nusage by model:')
            for row in rows:
                print(
                    f'  {row.model_name}: {row.model_requests} requests, '
                    f'{row.input_tokens} in / {row.output_tokens} out tokens, cost={row.cost}'
                )
    finally:
        await engine.dispose()


if __name__ == '__main__':
    asyncio.run(main())
