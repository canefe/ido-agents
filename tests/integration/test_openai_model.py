import os

import dotenv
import pytest
from pydantic import SecretStr

from ido_agents.models.openai import OpenAIModelConfig, build_chat_model
from ido_agents.utils.console import console_print


def test_build_chat_model_openai():
    pytest.importorskip("langchain_openai")

    cfg = OpenAIModelConfig(
        model="gpt-4o-mini",
        api_key=SecretStr("test-key"),
        base_url="https://example.com",
        temperature=0.1,
        reasoning_effort="low",
    )

    model = build_chat_model(cfg)
    assert model is not None


def test_openai_live_call_if_env_set():
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
    result = model.invoke("ping")
    console_print(result)
    assert result is not None
