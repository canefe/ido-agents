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

from ido_agents.agents.ido_agent import create_ido_agent


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

## Example

Define the response shape, give the agent a helper tool, then ask it to answer about dogs and cats in structured form.

```python
class PetResult(BaseModel):
    favorite: Literal["dog", "cat"] = Field(
        description="Which pet is the favorite in this answer",
        examples=["dog"],
    )
    reason: str = Field(
        description="Short reason for the preference",
        examples=["Dogs are loyal and love outdoor adventures."],
    )
    tips: list[str] = Field(
        description="Helpful tips for the chosen pet",
        examples=[["Take them for daily walks", "Keep fresh water available"]],
    )


@tool
def web_search(query: str) -> list[str]:
    """Mock web search results for a query."""
    return [
        f"Result: {query} daily care checklist",
        f"Result: {query} vet visit schedule",
        f"Result: {query} enrichment ideas",
    ]


agent = create_ido_agent(
    model=model,
    tools=[web_search],
)

result = (
    agent.with_structured_output(PetResult)
    .with_tool_caller(ToolCallerSettings(max_tool_calls=2))
    .invoke(
        [
            HumanMessage(
                    content=(
                        "Pick between dogs and cats, explain why, and include care tips. "
                    )
                )
            ]
        )
)
```

## Why this feels good

- You only model the response once with Pydantic, and the agent enforces it.
- Tool calling is explicit and bounded; no manual parsing or post-processing.
- You can layer `with_retry` on top if you want automatic re-tries on bad output.
