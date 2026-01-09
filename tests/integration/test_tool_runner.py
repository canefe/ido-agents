import os

import dotenv
import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import SecretStr

from ido_agents.agents.tool_runner import (
    ToolCallerResult,
    ToolCallerSettings,
    tool_caller,
)
from ido_agents.models.openai import OpenAIModelConfig, build_chat_model
from ido_agents.utils.console import console_print


@pytest.mark.anyio
async def test_tool_runner_live_call_if_env_set():
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
        prompt="Use the ping tool 5 times and confirm the count of tool calls.",
        settings=ToolCallerSettings(),
    )
    assert result.text and result.text != "[]"
    assert result.tool_calls > 0


@pytest.mark.anyio
async def test_tool_runner_live_call_with_tool_limit_if_env_set():
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
        prompt="Use the ping tool 5 times and confirm the count of tool calls.",
        settings=ToolCallerSettings(max_tool_calls=2),
    )
    console_print(result.text)
    assert result.text and result.text != "[]"
    assert 0 < result.tool_calls <= 2


@pytest.mark.anyio
async def test_tool_runner_live_call_with_max_iterations_if_env_set():
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
        prompt="Use the ping tool 5 times and confirm the count of tool calls.",
        settings=ToolCallerSettings(max_iterations=1),
    )
    assert result.error
    assert "recursion limit" in result.error.lower()


async def _run_tool_runner_live_call(
    api_key: str,
    base_url: str,
    model_name: str,
    prompt: str,
    settings: ToolCallerSettings,
) -> ToolCallerResult:
    cfg = OpenAIModelConfig(
        model=model_name,
        api_key=SecretStr(api_key),
        base_url=base_url,
        temperature=0.0,
        reasoning_effort="low",
    )
    model = build_chat_model(cfg)

    @tool
    def ping() -> str:
        """Return a short acknowledgement."""
        return "pong"

    agent = create_agent(
        model=model,
        tools=[ping],
    )

    return await tool_caller(
        agent,
        [HumanMessage(content=prompt)],
        settings,
    )
