<div align="center" style="margin-bottom: 1rem;">
  <h1 style="font-size: 2rem; margin: 0;">ido-agents</h1>
Less time figuring things out, more time building AI agents.
<br>
<br>
  <p>
    <img src="https://img.shields.io/badge/python-3.13%2B-3776AB.svg?style=flat-square" alt="Python Version" />
    <img src="https://img.shields.io/badge/langchain-1.2%2B-blue.svg?style=flat-square" alt="LangChain" />
  </p>
</div>

Declarative, structured agents on top of LangChain. Define a Pydantic model and tools,
then let the agent return validated JSON without you wiring parsing or retries by hand.

## Fast Start

Minimal setup with `create_ido_agent` and a response model:

```python
from pydantic import BaseModel

from idoagents.agents.ido_agent import create_ido_agent


class Result(BaseModel):
    summary: str


agent = create_ido_agent(model=model, tools=[])
result = agent.with_structured_output(Result).invoke("Summarize this.")
```

Buildable pattern (structured output + tools + tool caller):

```python
agent = create_ido_agent(model=model, tools=[my_tool])
result = (
    agent.with_structured_output(Result)
    .with_tool_caller(settings)
    .invoke("Use the tool, then respond.")
)
```

## Quick Example

Shows a task-oriented flow: define the shape of a task result, give the agent a helper tool, then ask it to plan and report the task outcome in structured form.

```python
import os
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, SecretStr

from idoagents.agents.ido_agent import create_ido_agent
from idoagents.agents.tool_runner import ToolCallerSettings
from idoagents.models.openai import OpenAIModelConfig, build_chat_model


class TaskResult(BaseModel):
    status: Literal["success", "needs_followup"] = Field(
        description="High-level outcome for the task",
        examples=["success"],
    )
    summary: str = Field(
        description="Short, user-facing summary of the result",
        examples=["Created a plan and listed the next steps."],
    )
    next_steps: list[str] = Field(
        description="Actionable follow-ups if any",
        examples=[["Send the draft to the team", "Schedule a review"]],
    )


@tool
def suggest_next_steps(topic: str) -> list[str]:
    """Provide short, actionable next steps for a topic."""
    return [
        f"Draft the {topic} scope",
        f"Estimate effort for {topic}",
        f"Schedule a quick review for {topic}",
    ]


async def main() -> TaskResult:
    cfg = OpenAIModelConfig(
        model=os.environ["LLM_MODEL"],
        api_key=SecretStr(os.environ["LLM_API_KEY"]),
        base_url=os.environ["LLM_BASE_URL"],
        temperature=0.0,
        reasoning_effort="low",
    )
    model = build_chat_model(cfg)

    agent = create_ido_agent(
        model=model,
        tools=[suggest_next_steps],
    )

    result = await (
        agent.with_structured_output(TaskResult)
        .with_tool_caller(ToolCallerSettings(max_tool_calls=2))
        .ainvoke(
            [
                HumanMessage(
                    content=(
                        "Create a short action plan for shipping a small feature. "
                        "Call suggest_next_steps and include its output."
                    )
                )
            ]
        )
    )

    return result
```

## Why this feels good

- You only model the response once with Pydantic, and the agent enforces it.
- Tool calling is explicit and bounded; no manual parsing or post-processing.
- You can layer `with_retry` on top if you want automatic re-tries on bad output.
