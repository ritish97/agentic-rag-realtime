import sys
from typing import List, Optional
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table
    from rich.text import Text
    _HAS_RICH = True
    _console = Console(file=sys.stdout, highlight=False)
except ImportError:
    _HAS_RICH = False
    _console = None
_BOX_WIDTH = 68
TOOL_BADGES = {
    "get_stock_data": ("📈", "Live Stocks (yfinance)"),
    "search_knowledge_base": ("📚", "Internal Knowledge Base (FAISS)"),
    "get_weather": ("🌦️", "Current Weather (OpenWeather)"),
    "search_wikipedia": ("📖", "Wikipedia Encyclopedia"),
    "web_search": ("🌐", "Live Web Search (Tavily)"),
}
def get_tool_display_name(tool_name: str) -> str:
    """Return an icon + human readable name for a tool."""
    if tool_name in TOOL_BADGES:
        icon, name = TOOL_BADGES[tool_name]
        return f"{icon} {name}"
    return f"🔧 {tool_name}"

def format_response(
    answer: str,
    tools_used: List[str],
    agent_steps: Optional[list] = None,
    duration_s: Optional[float] = None,
) -> str:
    lines: List[str] = []
    lines.append("╔" + "═" * _BOX_WIDTH + "╗")
    header_title = "  ⚡ AGENT RESPONSE"
    if duration_s is not None:
        header_title += f" ({duration_s:.2f}s)"
    lines.append(f"║{header_title.ljust(_BOX_WIDTH)}║")
    lines.append("╚" + "═" * _BOX_WIDTH + "╝")
    lines.append("")
    lines.append(answer)
    lines.append("")

    if tools_used:
        badges = [get_tool_display_name(t) for t in tools_used]
        tools_str = " | ".join(badges)
    else:
        tools_str = "Direct Reasoning (no external tools required)"

    lines.append("┌" + "─" * _BOX_WIDTH + "┐")
    tools_line = f" Tools Activated: {tools_str}"
    if len(tools_line) > _BOX_WIDTH - 1:
        tools_line = tools_line[: _BOX_WIDTH - 4] + "…"
    lines.append("│" + tools_line.ljust(_BOX_WIDTH) + "│")
    lines.append("└" + "─" * _BOX_WIDTH + "┘")

    return "\n".join(lines)


def display_rich_response(
    answer: str,
    tools_used: List[str],
    agent_steps: Optional[list] = None,
    show_trace: bool = True,
    duration_s: Optional[float] = None,
    model_name: Optional[str] = None,
) -> None:
    if not _HAS_RICH or _console is None:
        print("\n" + format_response(answer, tools_used, agent_steps, duration_s))
        return

    if show_trace and agent_steps:
        _console.print("\n[bold cyan]🧠 Agent Reasoning & Tool Activation Trace[/bold cyan]")
        for i, step in enumerate(agent_steps, start=1):
            try:
                action, observation = step[0], step[1]
                tool_name = getattr(action, "tool", "tool")
                tool_input = getattr(action, "tool_input", "")
                log = getattr(action, "log", "").strip()

                icon, label = TOOL_BADGES.get(tool_name, ("🔧", tool_name))

                step_table = Table(show_header=False, box=None, padding=(0, 1))
                step_table.add_column("Key", style="bold yellow", width=14)
                step_table.add_column("Value", style="white")

                if log:
                    thought = log.split("\n")[0].replace("Thought:", "").strip()
                    step_table.add_row("💭 Thought", f"[italic]{thought}[/italic]")

                step_table.add_row("⚡ Action", f"[bold green]{icon} {tool_name}[/bold green]([cyan]{tool_input!r}[/cyan])")

                obs_str = str(observation).strip()
                if len(obs_str) > 400:
                    obs_str = obs_str[:400] + " … [dim](truncated)[/dim]"
                step_table.add_row("👁️ Observation", obs_str)

                _console.print(Panel(step_table, title=f"[bold]Step {i}[/bold]", border_style="cyan", expand=False))
            except Exception:
                continue
    if tools_used:
        badge_text = " • ".join(get_tool_display_name(t) for t in tools_used)
    else:
        badge_text = "Direct Synthesis"

    meta_parts = [badge_text]
    if model_name:
        meta_parts.append(f"Model: {model_name}")
    if duration_s is not None:
        meta_parts.append(f"⏱️ {duration_s:.2f}s")

    panel_title = f"[bold green]✨ Grounded Agent Answer[/bold green] [dim]({' | '.join(meta_parts)})[/dim]"
    _console.print(Panel(Markdown(answer), title=panel_title, border_style="green", padding=(1, 2)))


def extract_tools_from_steps(agent_steps: list) -> List[str]:
    seen = set()
    tools: List[str] = []

    for step in agent_steps or []:
        try:
            action = step[0]
            tool_name = getattr(action, "tool", None)
            if tool_name and tool_name not in seen:
                tools.append(tool_name)
                seen.add(tool_name)
        except (IndexError, TypeError):
            continue

    return tools

def format_agent_trace(agent_steps: list) -> str:
    if not agent_steps:
        return "(No intermediate steps recorded)"

    lines: List[str] = ["── Agent Reasoning Trace ──"]

    for i, step in enumerate(agent_steps, start=1):
        try:
            action, observation = step[0], step[1]
            tool_name = getattr(action, "tool", "unknown_tool")
            tool_input = getattr(action, "tool_input", "")
            log = getattr(action, "log", "").strip()

            lines.append(f"\nStep {i}:")
            if log:
                thought_line = log.split("\n")[0]
                lines.append(f"  Thought   : {thought_line}")

            lines.append(f"  Action    : {tool_name}({tool_input!r})")
            obs_str = str(observation)
            if len(obs_str) > 300:
                obs_str = obs_str[:300] + "…"
            lines.append(f"  Observation: {obs_str}")

        except Exception:
            lines.append(f"\nStep {i}: (could not parse step)")

    return "\n".join(lines)
