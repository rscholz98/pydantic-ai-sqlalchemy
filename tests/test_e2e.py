"""U12 end-to-end test: agent run -> save -> byte-identical reload -> follow-up run -> analytics.

Runs a FunctionModel agent (canned tool call plus text answer, no API key), archives the run,
proves the persisted history round-trips byte for byte through ``ModelMessagesTypeAdapter``,
continues the conversation from the loaded history and checks the analytics rollups are non-zero.

The store methods used here are implemented by parallel work units; until those merge, hitting a
``NotImplementedError`` xfails the test instead of failing the build.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import TypeVar

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from pydantic_ai_sqlalchemy import SQLAlchemyChatStore

_T = TypeVar('_T')

CONVERSATION_KEY = 'e2e:dice'


async def _store_call(awaitable: Awaitable[_T]) -> _T:
    """Await one store call, xfailing while the sibling units that implement it are unmerged."""
    try:
        return await awaitable
    except NotImplementedError:
        pytest.xfail('depends on units U2/U3/U6')


# Keep this canned model in sync with the one in examples/basic.py (same answer, same model name).
# Explicit usage keeps the analytics assertions independent of FunctionModel-synthesized token counts.
def _canned_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    seen_tool_return = any(part.part_kind == 'tool-return' for message in messages for part in message.parts)
    if info.function_tools and not seen_tool_return:
        return ModelResponse(
            parts=[ToolCallPart(tool_name='roll_dice', args={})], usage=RequestUsage(input_tokens=52, output_tokens=6)
        )
    return ModelResponse(
        parts=[TextPart('You rolled a 4. Lucky you!')], usage=RequestUsage(input_tokens=61, output_tokens=9)
    )


def _build_agent() -> Agent:
    agent = Agent(FunctionModel(_canned_model, model_name='canned-dice-model'))

    @agent.tool_plain
    def roll_dice() -> int:
        return 4

    return agent


async def test_run_save_reload_followup_analytics(store: SQLAlchemyChatStore) -> None:
    agent = _build_agent()

    # First turn: user prompt -> tool call -> tool return -> final text answer.
    result = await agent.run('Roll the dice!')
    result_messages = result.all_messages()
    assert result.output == 'You rolled a 4. Lucky you!'

    saved = await _store_call(store.save_run(result, conversation_key=CONVERSATION_KEY, agent_name='dice-agent'))
    assert saved.conversation_key == CONVERSATION_KEY
    assert len(saved.message_ids) == len(result_messages)

    # The archive must reproduce the run byte for byte at the adapter level.
    history = await _store_call(store.load_history(conversation_key=CONVERSATION_KEY))
    assert ModelMessagesTypeAdapter.dump_json(history) == ModelMessagesTypeAdapter.dump_json(result_messages)

    # Second turn continues from the loaded history; the model sees the tool return and answers directly.
    followup = await agent.run('Was that a good roll?', message_history=history)
    followup_messages = followup.all_messages()
    assert followup.output == 'You rolled a 4. Lucky you!'
    assert len(followup_messages) == len(history) + 2

    saved_followup = await _store_call(
        store.save_run(followup, conversation_key=CONVERSATION_KEY, agent_name='dice-agent')
    )
    assert saved_followup.conversation_pk == saved.conversation_pk

    # With only_new=True the second save appends exactly the new messages, never the loaded history.
    full_history = await _store_call(store.load_history(conversation_key=CONVERSATION_KEY))
    assert ModelMessagesTypeAdapter.dump_json(full_history) == ModelMessagesTypeAdapter.dump_json(followup_messages)

    # Analytics rollups must reflect the archived runs.
    by_model = await _store_call(store.usage_by_model())
    assert sum(row.model_requests for row in by_model) > 0
    assert sum(row.input_tokens for row in by_model) > 0
    assert sum(row.output_tokens for row in by_model) > 0
    assert any(row.model_name == 'canned-dice-model' for row in by_model)

    by_day = await _store_call(store.usage_by_day())
    assert sum(row.model_requests for row in by_day) > 0
    assert sum(row.input_tokens for row in by_day) > 0

    cost_rows = await _store_call(store.cost_report(group_by='conversation'))
    assert [row.group for row in cost_rows] == [CONVERSATION_KEY]
    assert cost_rows[0].input_tokens > 0
