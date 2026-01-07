import os
from typing import Literal, Optional

import dotenv
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, SecretStr

from idoagents.agents.ido_agent import create_ido_agent
from idoagents.agents.tool_runner import ToolCallerSettings
from idoagents.models.openai import OpenAIModelConfig, build_chat_model
from idoagents.utils.console import console_print


class QuestStateUpdate(BaseModel):
    quest_id: str = Field(
        description="Unique quest identifier",
        examples=["quest_lost_relic"],
    )
    state: Literal["not_started", "in_progress", "completed", "failed"] = Field(
        description="Current quest progression state",
        examples=["in_progress"],
    )
    objective: Optional[str] = Field(
        description="Current active objective shown to the player",
        examples=["Search the ruined chapel"],
    )
    world_effects: list[str] = Field(
        description="World changes caused by this update",
        examples=["unlock_chapel_door", "spawn_bandits"],
    )


@pytest.mark.anyio
async def test_ido_agent():
    pytest.importorskip("langchain_openai")
    dotenv.load_dotenv(".env.test")

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model_name = os.getenv("LLM_MODEL")
    if not (api_key and base_url and model_name):
        pytest.skip("LLM env vars not set")

    result = await _run_tool_runner_live_call(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        prompt="He said, 'He was coming for me'. I need to protect myself.",
        settings=ToolCallerSettings(max_tool_calls=2),
    )
    console_print(result)
    assert isinstance(result, QuestStateUpdate)
    assert result.state


async def _run_tool_runner_live_call(
    api_key: str,
    base_url: str,
    model_name: str,
    prompt: str,
    settings: ToolCallerSettings,
) -> QuestStateUpdate:
    cfg = OpenAIModelConfig(
        model=model_name,
        api_key=SecretStr(api_key),
        base_url=base_url,
        temperature=0.0,
        reasoning_effort="low",
    )
    model = build_chat_model(cfg)

    @tool
    def generate_quest(name: str) -> str:
        """Generate a quest in the system."""
        result = "The quest has been created: " + name
        console_print(result)
        return result

    agent = create_ido_agent(
        model=model,
        tools=[generate_quest],
    )

    result = await (
        agent.with_structured_output(QuestStateUpdate)
        .with_tool_caller(settings)
        .with_retry(max_retries=20)
        .ainvoke([HumanMessage(content=prompt)])
    )

    return result
