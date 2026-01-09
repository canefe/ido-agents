from __future__ import annotations

import json
from typing import Any, Generic, Type, TypeVar

from pydantic import BaseModel

TModel = TypeVar("TModel", bound=BaseModel)
TOut = TypeVar("TOut", bound=BaseModel)


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        if "content" in value:
            return _extract_text(value["content"])
        if "messages" in value and value["messages"]:
            return _extract_text(value["messages"][-1])

    if hasattr(value, "content"):
        return _extract_text(getattr(value, "content"))

    if isinstance(value, list):
        return "\n".join(_extract_text(item) for item in value if item is not None)

    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


def parse_structured_output(response_model: Type[TModel], value: Any) -> TModel:
    if isinstance(value, response_model):
        return value

    if isinstance(value, dict):
        return response_model.model_validate(value)

    text = _extract_text(value)
    try:
        return response_model.model_validate_json(text)
    except Exception:
        extracted = _extract_json_blob(text)
        if extracted is None:
            raise
        return response_model.model_validate_json(extracted)


def _extract_json_blob(text: str) -> str | None:
    text = text.strip()
    for open_brace, close_brace in (("{", "}"), ("[", "]")):
        start = text.find(open_brace)
        end = text.rfind(close_brace)
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
    return None


def _default_system_prompt(response_model: Type[BaseModel]) -> str:
    schema = response_model.model_json_schema()
    schema_json = json.dumps(schema, indent=2)
    required = schema.get("required", [])
    required_hint = ""
    if required:
        required_hint = f"Required fields: {', '.join(required)}. "
    return (
        "You are a structured output agent. Return only valid JSON that "
        "matches the requested schema. Do not include any extra keys or commentary. "
        f"{required_hint}"
        "Use the following JSON schema:\n"
        f"{schema_json}"
    )


class IdoRunnable(Generic[TOut]):
    def __init__(
        self,
        runnable: Any,
        response_model: Type[TOut] | None = None,
        system_prompt: str | None = None,
    ):
        self._runnable = runnable
        self._response_model = response_model
        self._system_prompt = system_prompt

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runnable, name)

    def with_structured_output(
        self,
        response_model: Type[TModel],
        system_prompt: str | None = None,
    ) -> "IdoRunnable[TModel]":
        if system_prompt is None:
            system_prompt = _default_system_prompt(response_model)

        if hasattr(self._runnable, "with_structured_output"):
            structured = self._runnable.with_structured_output(response_model)
            return IdoRunnable[TModel](
                structured, response_model=response_model, system_prompt=system_prompt
            )

        return StructuredRunnable[TModel](
            runnable=self._runnable,
            response_model=response_model,
            system_prompt=system_prompt,
        )

    def with_structured_outpuit(
        self,
        response_model: Type[TModel],
        system_prompt: str | None = None,
    ) -> "IdoRunnable[TModel]":
        return self.with_structured_output(
            response_model=response_model, system_prompt=system_prompt
        )

    def with_retry(
        self,
        max_retries: int = 2,
        retry_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> "IdoRunnable[TOut]":
        return RetryingRunnable(
            runnable=self,
            response_model=self._response_model,
            system_prompt=self._system_prompt,
            max_retries=max_retries,
            retry_exceptions=retry_exceptions,
        )

    def with_tool_caller(self, settings: Any) -> "ToolCallerRunnable[TOut]":
        return ToolCallerRunnable(
            runnable=self._runnable,
            settings=settings,
            response_model=self._response_model,
            system_prompt=self._system_prompt,
        )

    def invoke(self, input: Any, **kwargs: Any) -> Any | TOut:
        return self._runnable.invoke(input, **kwargs)

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any | TOut:
        return await self._runnable.ainvoke(input, **kwargs)


class StructuredRunnable(IdoRunnable[TModel]):
    def __init__(
        self,
        runnable: Any,
        response_model: Type[TModel],
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(
            runnable, response_model=response_model, system_prompt=system_prompt
        )

    def invoke(self, input: Any, **kwargs: Any) -> TModel:
        output = self._runnable.invoke(self._inject_system_prompt(input), **kwargs)
        return parse_structured_output(self._response_model, output)

    async def ainvoke(self, input: Any, **kwargs: Any) -> TModel:
        output = await self._runnable.ainvoke(
            self._inject_system_prompt(input), **kwargs
        )
        return parse_structured_output(self._response_model, output)

    def _inject_system_prompt(self, input: Any) -> Any:
        if not self._system_prompt:
            return input

        try:
            from langchain_core.messages import SystemMessage
        except Exception:
            SystemMessage = None

        system_msg = (
            SystemMessage(content=self._system_prompt)
            if SystemMessage is not None
            else {"type": "system", "content": self._system_prompt}
        )

        if isinstance(input, dict) and "messages" in input:
            messages = list(input["messages"])
            messages.insert(0, system_msg)
            new_input = dict(input)
            new_input["messages"] = messages
            return new_input

        if isinstance(input, list):
            messages = list(input)
            messages.insert(0, system_msg)
            return messages

        return input


class RetryingRunnable(IdoRunnable[TOut]):
    def __init__(
        self,
        runnable: Any,
        response_model: Type[TOut] | None,
        system_prompt: str | None,
        max_retries: int,
        retry_exceptions: tuple[type[Exception], ...],
    ) -> None:
        super().__init__(
            runnable, response_model=response_model, system_prompt=system_prompt
        )
        self._max_retries = max(0, max_retries)
        self._retry_exceptions = retry_exceptions

    def with_structured_output(
        self,
        response_model: Type[TModel],
        system_prompt: str | None = None,
    ) -> "IdoRunnable[TModel]":
        base = super().with_structured_output(
            response_model=response_model, system_prompt=system_prompt
        )
        return RetryingRunnable(
            runnable=base,
            response_model=base._response_model,
            system_prompt=base._system_prompt,
            max_retries=self._max_retries,
            retry_exceptions=self._retry_exceptions,
        )

    def with_structured_outpuit(
        self,
        response_model: Type[TModel],
        system_prompt: str | None = None,
    ) -> "IdoRunnable[TModel]":
        return self.with_structured_output(
            response_model=response_model, system_prompt=system_prompt
        )

    def with_tool_caller(self, settings: Any) -> "ToolCallerRunnable[TOut]":
        base = super().with_tool_caller(settings)
        return RetryingRunnable(
            runnable=base,
            response_model=base._response_model,
            system_prompt=base._system_prompt,
            max_retries=self._max_retries,
            retry_exceptions=self._retry_exceptions,
        )

    def with_retry(
        self,
        max_retries: int = 2,
        retry_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> "IdoRunnable[TOut]":
        return RetryingRunnable(
            runnable=self._runnable,
            response_model=self._response_model,
            system_prompt=self._system_prompt,
            max_retries=max_retries,
            retry_exceptions=retry_exceptions,
        )

    def invoke(self, input: Any, **kwargs: Any) -> Any | TOut:
        attempts = 0
        while True:
            try:
                return self._runnable.invoke(input, **kwargs)
            except self._retry_exceptions:
                if attempts >= self._max_retries:
                    raise
                attempts += 1

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any | TOut:
        attempts = 0
        while True:
            try:
                return await self._runnable.ainvoke(input, **kwargs)
            except self._retry_exceptions:
                if attempts >= self._max_retries:
                    raise
                attempts += 1


class ToolCallerRunnable(IdoRunnable[TOut]):
    def __init__(
        self,
        runnable: Any,
        settings: Any,
        response_model: Type[TOut] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(
            runnable, response_model=response_model, system_prompt=system_prompt
        )
        self.settings = settings

    def invoke(self, input: Any, **kwargs: Any) -> Any | TOut:
        return _run_tool_caller(
            runnable=self._runnable,
            settings=self.settings,
            response_model=self._response_model,
            input_value=input,
            system_prompt=self._system_prompt,
        )

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any | TOut:
        return await _run_tool_caller_async(
            runnable=self._runnable,
            settings=self.settings,
            response_model=self._response_model,
            input_value=input,
            system_prompt=self._system_prompt,
        )


def _extract_messages(input_value: Any) -> list[Any]:
    if isinstance(input_value, dict) and "messages" in input_value:
        return list(input_value["messages"])

    if isinstance(input_value, list):
        return list(input_value)

    return [input_value]


def _inject_system_prompt(messages: list[Any], system_prompt: str | None) -> list[Any]:
    if not system_prompt:
        return messages

    try:
        from langchain_core.messages import SystemMessage
    except Exception:
        SystemMessage = None

    system_msg = (
        SystemMessage(content=system_prompt)
        if SystemMessage is not None
        else {"type": "system", "content": system_prompt}
    )
    return [system_msg] + messages


def _run_tool_caller(
    *,
    runnable: Any,
    settings: Any,
    response_model: Type[BaseModel] | None,
    input_value: Any,
    system_prompt: str | None = None,
) -> Any:
    from ido_agents.agents.tool_runner import ToolCallerResult, tool_caller

    messages = _inject_system_prompt(
        _extract_messages(input_value), system_prompt=system_prompt
    )
    result: ToolCallerResult = _sync_wait(tool_caller(runnable, messages, settings))
    return _maybe_parse_result(result, response_model)


async def _run_tool_caller_async(
    *,
    runnable: Any,
    settings: Any,
    response_model: Type[BaseModel] | None,
    input_value: Any,
    system_prompt: str | None = None,
) -> Any:
    from ido_agents.agents.tool_runner import ToolCallerResult, tool_caller

    messages = _inject_system_prompt(
        _extract_messages(input_value), system_prompt=system_prompt
    )
    result: ToolCallerResult = await tool_caller(runnable, messages, settings)
    return _maybe_parse_result(result, response_model)


def _maybe_parse_result(result: Any, response_model: Type[BaseModel] | None) -> Any:
    if response_model is None:
        return result.text

    return parse_structured_output(response_model, result.text)


def _sync_wait(coro: Any) -> Any:
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        raise RuntimeError("Cannot call sync tool runner while event loop is running.")

    return asyncio.run(coro)
