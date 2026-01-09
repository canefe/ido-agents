from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from pydantic import SecretStr

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class OpenAIModelConfig:
    model: str
    api_key: SecretStr
    base_url: Optional[str] = None
    temperature: float = 0.2
    reasoning_effort: str = "low"


def build_chat_model(cfg: OpenAIModelConfig) -> "ChatOpenAI":
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "Install with: `pip install idoagents[openai]` (or `uv add idoagents[openai]`)."
        ) from exc

    return ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        reasoning_effort=cfg.reasoning_effort,
    )


def with_low_effort(cfg: OpenAIModelConfig) -> OpenAIModelConfig:
    return OpenAIModelConfig(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        reasoning_effort="low",
    )
