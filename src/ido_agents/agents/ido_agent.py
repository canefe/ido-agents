from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Type

if TYPE_CHECKING:
    from ido_agents.utils.structured_output import IdoRunnable


def create_ido_agent(
    *,
    model: Any,
    tools: Iterable[Any],
    response_model: Type[Any] | None = None,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> "IdoRunnable[Any]":
    from langchain.agents import create_agent as create_langchain_agent

    from ido_agents.utils.structured_output import IdoRunnable

    agent = create_langchain_agent(model=model, tools=list(tools), **kwargs)
    wrapped = IdoRunnable(agent)

    if response_model is None:
        return wrapped

    return wrapped.with_structured_output(response_model, system_prompt=system_prompt)
