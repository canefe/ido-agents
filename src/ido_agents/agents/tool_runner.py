import json
import re
from dataclasses import dataclass, field
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
    track_tool_usage: bool = False
    # When True, tool_caller appends a SystemMessage with prior tool usage context
    # (tool names, args, context_range) so the LLM knows what it already looked up.
    prior_tool_usage: list["ToolUsage"] = field(default_factory=list)
    # Populated after tool_caller runs — contains tool usage from this invocation only.
    last_tool_usage: list["ToolUsage"] = field(default_factory=list)
    # Populated after tool_caller runs — total tool calls made in this invocation.
    last_tool_call_count: int = 0


@dataclass
class ToolUsage:
    """Record of a single tool call with its args and key metadata from the result."""
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    context_range: str | None = None


@dataclass
class ToolCallerResult:
    text: str
    tool_calls: int
    error: str | None = None
    parsed: Any | None = None
    parse_error: str | None = None
    tool_usage: list[ToolUsage] = field(default_factory=list)


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


def _extract_context_range(content: Any) -> str | None:
    """Try to extract context_range from a tool message content."""
    text = content if isinstance(content, str) else str(content)
    # Try JSON parse first
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "context_range" in data:
            return data["context_range"]
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: regex
    m = re.search(r'"context_range"\s*:\s*"([^"]+)"', text)
    return m.group(1) if m else None


def _format_tool_usage_context(usage: list["ToolUsage"]) -> str:
    """Format prior tool usage into a context string for the LLM."""
    if not usage:
        return ""
    lines = ["## Previously Looked Up Resources"]
    for u in usage:
        args_str = ", ".join(f"{k}={v!r}" for k, v in u.args.items()) if u.args else ""
        line = f"- {u.tool_name}({args_str})"
        if u.context_range:
            line += f" — fetched {u.context_range}"
        lines.append(line)
    lines.append("\nYou already have these. Avoid re-fetching the same data unless you need a different range.")
    return "\n".join(lines)


async def tool_caller(
    agent: Any, messages: list[BaseMessage], settings: ToolCallerSettings
) -> ToolCallerResult:
    total_tool_calls = 0
    final_output = ""
    limit_reached = False
    seen_tool_msg_ids: set[str] = set()  # track which ToolMessages we've logged
    all_tool_usage: list[ToolUsage] = []

    # Inject prior tool usage context if tracking is enabled
    if settings.track_tool_usage and settings.prior_tool_usage:
        ctx = _format_tool_usage_context(settings.prior_tool_usage)
        if ctx:
            from langchain_core.messages import SystemMessage
            messages = list(messages) + [SystemMessage(content=ctx)]

    # config manages the internal recursion (iterations)
    config = {}
    if settings.max_iterations != -1:
        config = {"recursion_limit": settings.max_iterations}

    try:
        # stream_mode="values" yields the FULL state after each node runs.
        # After the LLM node: latest message is AIMessage (possibly with tool_calls).
        # After the tools node: state has all ToolMessages appended at the end.
        async for state in agent.astream(
            {"messages": messages}, config=config, stream_mode="values"
        ):
            current_messages = state.get("messages", [])
            if not current_messages:
                continue

            latest_msg = current_messages[-1]

            # 1. LLM requested tool calls
            if isinstance(latest_msg, AIMessage) and latest_msg.tool_calls:
                new_calls = len(latest_msg.tool_calls)
                total_tool_calls += new_calls
                console_print(
                    f"[dim]➔ Agent requested {new_calls} tools (Total: {total_tool_calls})[/dim]"
                )
                for tc in latest_msg.tool_calls:
                    tc_args = ", ".join(f"{k}={v!r}" for k, v in tc.get("args", {}).items())
                    console_print(f"[orange3]  ⚡ {tc['name']}({tc_args})[/orange3]")

                if (
                    settings.max_tool_calls != -1
                    and total_tool_calls >= settings.max_tool_calls
                ):
                    limit_reached = True
                    console_print(
                        f"[yellow]⚠ Tool call limit ({settings.max_tool_calls}) reached. "
                        f"Waiting for tools to finish...[/yellow]"
                    )

            # 2. Tools node finished — latest_msg is a ToolMessage
            #    Log all new ToolMessages we haven't seen yet.
            elif isinstance(latest_msg, ToolMessage):
                # Build a lookup of tool_call_id -> args from AIMessages
                tool_call_args: dict[str, dict] = {}
                for m in current_messages:
                    if isinstance(m, AIMessage) and m.tool_calls:
                        for tc in m.tool_calls:
                            tool_call_args[tc["id"]] = tc.get("args", {})

                for msg in current_messages:
                    if isinstance(msg, ToolMessage) and id(msg) not in seen_tool_msg_ids:
                        seen_tool_msg_ids.add(id(msg))
                        args = tool_call_args.get(msg.tool_call_id, {})
                        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items()) if args else ""
                        console_print(f"[green]✓ Tool '{msg.name}' completed.[/green] [dim]({args_str})[/dim]" if args_str else f"[green]✓ Tool '{msg.name}' completed.[/green]")
                        all_tool_usage.append(ToolUsage(
                            tool_name=msg.name,
                            args=args,
                            context_range=_extract_context_range(msg.content),
                        ))

                # After tools finish and limit was reached, force final response
                if limit_reached:
                    console_print("[yellow]All tools done. Forcing final response.[/yellow]")
                    final_output = await _force_final_response(
                        agent=agent,
                        messages=current_messages,
                        config=config,
                        total_tool_calls=total_tool_calls,
                    )
                    break

            # 3. LLM produced a final text response (no tool calls)
            if isinstance(latest_msg, AIMessage) and not latest_msg.tool_calls:
                final_output = extract_text_content(latest_msg.content)

    except Exception as e:
        console_print(f"[red]Streaming Error: {e}[/red]")
        settings.last_tool_usage = all_tool_usage
        return ToolCallerResult(
            text="Error occurred during agent execution.",
            tool_calls=total_tool_calls,
            error=str(e),
            tool_usage=all_tool_usage,
        )

    settings.last_tool_usage = all_tool_usage

    if not final_output:
        console_print("[yellow]Agent ended without a final text response.[/yellow]")
        return ToolCallerResult(text="[]", tool_calls=total_tool_calls, tool_usage=all_tool_usage)

    console_print(
        f"[bold blue]Final Response Ready ({len(final_output)} chars)[/bold blue]"
    )
    return ToolCallerResult(text=final_output, tool_calls=total_tool_calls, tool_usage=all_tool_usage)


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
