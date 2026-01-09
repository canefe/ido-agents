from __future__ import annotations

from typing import Any, Iterable, Type, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

TModel = TypeVar("TModel", bound=BaseModel)


class StructuredOutputAgent:
    def __init__(
        self,
        model: Any,
        response_model: Type[TModel],
        system_prompt: str | None = None,
    ) -> None:
        from idoagents.utils.structured_output import IdoRunnable

        self._model = IdoRunnable(model).with_structured_output(
            response_model, system_prompt=""
        )
        self._system_prompt = system_prompt or (
            "You are a structured output agent. Return only valid JSON that matches "
            "the requested schema. Do not include any extra keys or commentary."
        )

    def invoke(self, prompt: str) -> TModel:
        messages = [HumanMessage(content=prompt)]
        if self._system_prompt:
            messages.insert(0, SystemMessage(content=self._system_prompt))

        return self._model.invoke(messages)

    async def ainvoke(self, prompt: str) -> TModel:
        messages = [HumanMessage(content=prompt)]
        if self._system_prompt:
            messages.insert(0, SystemMessage(content=self._system_prompt))

        return await self._model.ainvoke(messages)


def create_structured_agent(
    model: Any,
    response_model: Type[TModel],
    system_prompt: str | None = None,
) -> StructuredOutputAgent:
    return StructuredOutputAgent(
        model=model, response_model=response_model, system_prompt=system_prompt
    )


class ToolStructuredAgent:
    def __init__(
        self,
        agent: Any,
        system_prompt: str | None = None,
    ) -> None:
        self._agent = agent
        self._system_prompt = system_prompt or (
            "You are a structured output agent. Return only valid JSON that matches "
            "the requested schema. Do not include any extra keys or commentary."
        )

    def invoke(self, prompt: str) -> TModel:
        messages = [HumanMessage(content=prompt)]
        messages.insert(0, SystemMessage(content=self._system_prompt))
        return self._agent.invoke({"messages": messages})

    async def ainvoke(self, prompt: str) -> TModel:
        messages = [HumanMessage(content=prompt)]
        messages.insert(0, SystemMessage(content=self._system_prompt))
        return await self._agent.ainvoke({"messages": messages})


def create_tool_structured_agent(
    *,
    model: Any,
    tools: Iterable[Any],
    response_model: Type[TModel],
    system_prompt: str | None = None,
    **kwargs: Any,
) -> ToolStructuredAgent:
    from langchain.agents import create_agent
    from idoagents.utils.structured_output import IdoRunnable

    agent = create_agent(model=model, tools=list(tools), **kwargs)
    wrapped = IdoRunnable(agent)
    structured = wrapped.with_structured_output(
        response_model, system_prompt=system_prompt
    )
    return ToolStructuredAgent(agent=structured, system_prompt=None)
