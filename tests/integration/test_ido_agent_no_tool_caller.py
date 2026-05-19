import os

import dotenv
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, SecretStr

from ido_agents.agents.ido_agent import create_ido_agent
from ido_agents.models.openai import OpenAIModelConfig, build_chat_model
from ido_agents.utils.console import console_print


class PetResult(BaseModel):
    favorite: str = Field(description="Which pet was chosen")
    reason: str = Field(description="Short reason for the choice")


@pytest.mark.anyio
async def test_ido_agent_structured_output_without_tool_caller():
    pytest.importorskip("langchain_openai")
    dotenv.load_dotenv(".env.test")

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model_name = os.getenv("LLM_MODEL")
    if not (api_key and base_url and model_name):
        pytest.skip("LLM env vars not set")

    cfg = OpenAIModelConfig(
        model=model_name,
        api_key=SecretStr(api_key),
        base_url=base_url,
        temperature=0.0,
        reasoning_effort="low",
    )
    model = build_chat_model(cfg)

    @tool
    def web_search(query: str) -> list[str]:
        """Mock web search."""
        return [f"Result: {query}"]

    agent = create_ido_agent(model=model, tools=[web_search])

    result = await (
        agent.with_structured_output(PetResult)
        .ainvoke(
            [
                HumanMessage(
                    content=(
                        "Pick between dogs and cats and explain why. "
                        "Your answer must be your own, not copied from any examples."
                    )
                )
            ]
        )
    )

    console_print(result)
    assert isinstance(result, PetResult)
    assert result.favorite
    assert result.reason
