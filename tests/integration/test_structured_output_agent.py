import os

import dotenv
import pytest
from pydantic import BaseModel, SecretStr

from idoagents.agents import create_structured_agent
from idoagents.models.openai import OpenAIModelConfig, build_chat_model
from idoagents.utils.console import console_print


class PingResult(BaseModel):
    message: str
    loving_words: str


@pytest.mark.anyio
async def test_structured_output_agent_live_call_if_env_set():
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
    agent = create_structured_agent(model, PingResult)

    result = await agent.ainvoke(
        "Respond with a short ping message. If you have any spare love to spread, feel free to do so. Message should be formal, the latter, informal."
    )
    console_print(result)
    assert isinstance(result, PingResult)
    assert result.message
