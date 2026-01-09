from dataclasses import dataclass
from typing import Any, List, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from ido_agents.utils.console import console_print


@dataclass
class ToolCallerSettings:
    """Tool caller settings"""

    max_tool_calls: int = -1
    max_iterations: int = -1  # Map to recursion_limit
    max_retries: int = 3
    # Note: Retries are usually handled by the LLM binding (.with_retry)
    # or the Agent Executor internally.


@dataclass
class ToolCallerResult:
    text: str
    tool_calls: int
    error: str | None = None
    parsed: Any | None = None
    parse_error: str | None = None


def extract_text_content(content: Union[str, List[Union[str, dict]]]) -> str:
    """Helper to safely extract text from various LangChain message formats."""
    if isinstance(content, str):
        return content

    text_parts = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif "text" in block:
                    text_parts.append(block["text"])

    return "\n".join(text_parts).strip()


async def tool_caller(
    agent: Any, messages: list[BaseMessage], settings: ToolCallerSettings
) -> ToolCallerResult:
    total_tool_calls = 0
    final_output = ""

    # config manages the internal recursion (iterations)
    config = {}
    if settings.max_iterations != -1:
        config = {"recursion_limit": settings.max_iterations}

    try:
        # stream_mode="values" yields the full state after every step
        async for state in agent.astream(
            {"messages": messages}, config=config, stream_mode="values"
        ):
            current_messages = state.get("messages", [])
            if not current_messages:
                continue

            latest_msg = current_messages[-1]

            # 1. Track Tool Calls
            if isinstance(latest_msg, AIMessage) and latest_msg.tool_calls:
                new_calls = len(latest_msg.tool_calls)
                total_tool_calls += new_calls
                console_print(
                    f"[dim]➔ Agent requested {new_calls} tools (Total: {total_tool_calls})[/dim]"
                )

                # 2. Check Tool Limit
                if (
                    settings.max_tool_calls != -1
                    and total_tool_calls >= settings.max_tool_calls
                ):
                    console_print(
                        f"[yellow]⚠ Tool call limit ({settings.max_tool_calls}) reached.[/yellow]"
                    )
                    final_output = await _force_final_response(
                        agent=agent,
                        messages=current_messages,
                        config=config,
                        total_tool_calls=total_tool_calls,
                    )
                    break

            # 3. Log Tool Results
            elif isinstance(latest_msg, ToolMessage):
                console_print(f"[green]✓ Tool '{latest_msg.name}' completed.[/green]")

            # Always update the potential final response
            if isinstance(latest_msg, AIMessage) and not latest_msg.tool_calls:
                final_output = extract_text_content(latest_msg.content)

    except Exception as e:
        console_print(f"[red]Streaming Error: {e}[/red]")
        return ToolCallerResult(
            text="Error occurred during agent execution.",
            tool_calls=total_tool_calls,
            error=str(e),
        )

    if not final_output:
        console_print("[yellow]Agent ended without a final text response.[/yellow]")
        return ToolCallerResult(text="[]", tool_calls=total_tool_calls)

    console_print(
        f"[bold blue]Final Response Ready ({len(final_output)} chars)[/bold blue]"
    )
    return ToolCallerResult(text=final_output, tool_calls=total_tool_calls)


async def _force_final_response(
    agent: Any,
    messages: list[BaseMessage],
    config: dict[str, Any],
    total_tool_calls: int,
) -> str:
    forced_messages = messages + [
        HumanMessage(
            content=(
                "Provide a final response now without calling any tools. "
                f"You made {total_tool_calls} tool calls so far; do not claim a different number."
            )
        )
    ]

    try:
        if hasattr(agent, "ainvoke"):
            result = await agent.ainvoke({"messages": forced_messages}, config=config)
        else:
            result = agent.invoke({"messages": forced_messages}, config=config)
    except Exception as exc:
        console_print(f"[red]Final response error: {exc}[/red]")
        return "Error occurred during agent execution."

    if isinstance(result, dict):
        result_messages = result.get("messages", [])
        if result_messages:
            latest_msg = result_messages[-1]
            if isinstance(latest_msg, AIMessage):
                return extract_text_content(latest_msg.content)

    if isinstance(result, AIMessage):
        return extract_text_content(result.content)

    return ""


# Example Usage:
# response = await tool_caller(my_agent, [HumanMessage(content="test")], ToolCallerSettings())
