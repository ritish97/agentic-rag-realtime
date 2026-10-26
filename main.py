import argparse
import os
import subprocess
import sys
import time

from dotenv import load_dotenv

# Reconfigure stdout/stderr on Windows to avoid charmap codec issues with Unicode
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load .env before importing project modules
load_dotenv()

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.text import Text
    _HAS_RICH = True
    _console = Console(file=sys.stdout, highlight=False)
except ImportError:
    _HAS_RICH = False
    _console = None
def _check_api_keys(config: dict) -> bool:
    groq_key = config.get("groq_api_key")
    tavily_key = config.get("tavily_api_key")
    owm_key = config.get("openweathermap_api_key")
    has_llm = bool(groq_key)

    if _HAS_RICH and _console is not None:
        table = Table(title="⚡ Agentic RAG — Tool & Service Readiness", border_style="cyan")
        table.add_column("Service / Tool", style="bold white", width=26)
        table.add_column("Status", width=36)
        table.add_column("Type", style="dim", width=22)

        # Groq LLM Engine
        if groq_key:
            table.add_row("⚡ Groq LLM Engine", f"[green]✅ Connected ({config['groq_model']})[/green]", "Ultra-Fast Inference")
        else:
            table.add_row("⚡ Reasoning Engine", "[yellow]⚠️ Simulation Mode (no key)[/yellow]", "Interactive Demo")

        table.add_row("📚 Knowledge Base (FAISS)", "[green]✅ Ready (Local Vector Store)[/green]", "Internal Documents")
        table.add_row("📈 Finance Tool (yfinance)", "[green]✅ Ready (Yahoo Finance)[/green]", "Live Market Data")
        table.add_row("📖 Wikipedia Tool", "[green]✅ Ready (REST API)[/green]", "Encyclopaedia")

        if tavily_key:
            table.add_row("🌐 Web Search (Tavily)", "[green]✅ Connected[/green]", "Live Web Crawl")
        else:
            table.add_row("🌐 Web Search", "[dim]⚠️ Mock Mode (add TAVILY_API_KEY)[/dim]", "Fallback Stub")

        if owm_key:
            table.add_row("🌦️ Weather Tool", "[green]✅ Connected (OpenWeather)[/green]", "Live Global Weather")
        else:
            table.add_row("🌦️ Weather Tool", "[dim]⚠️ Mock Mode (add OPENWEATHERMAP_API_KEY)[/dim]", "Fallback Stub")

        _console.print(table)
    else:
        print("\n" + "=" * 65)
        print("  ⚡ Agentic RAG — Tool Availability")
        print("=" * 65)
        print(f"  ⚡ Groq Engine   : {'✅ Connected (' + config['groq_model'] + ')' if groq_key else '⚠️ Simulation Mode'}")
        print("  📚 Knowledge Base: ✅ Ready (Local FAISS)")
        print("  📈 Finance Tool  : ✅ Ready (yfinance — live data)")
        print("  📖 Wikipedia Tool: ✅ Ready (Wikipedia API)")
        print(f"  🌐 Web Search    : {'✅ Connected' if tavily_key else '⚠️ Mock Mode'}")
        print(f"  🌦️ Weather Tool  : {'✅ Connected' if owm_key else '⚠️ Mock Mode'}")
        print("=" * 65)

    if not has_llm:
        notice = (
            "\n[Notice] GROQ_API_KEY not found in environment.\n"
            "Entering Interactive Simulation Mode: all tools (FAISS vector store,\n"
            "live Yahoo Finance stock metrics, Wikipedia, and weather) execute natively.\n"
            "Add GROQ_API_KEY to .env (from https://console.groq.com/keys) for full\n"
            "autonomous LLM multi-step tool calling powered by Groq LPU inference."
        )
        if _HAS_RICH and _console is not None:
            _console.print(Panel(notice, border_style="yellow", title="Demo / Simulation Mode"))
        else:
            print(notice)

    return has_llm


def _print_example_queries() -> None:
    """Print suggested example queries so users know what to explore."""
    examples = [
        ("📈 Finance + Internal RAG", 'What is AAPL stock price and how does it compare to our internal valuation?'),
        ("🌦️ Weather + Event Policy", 'What\'s the weather in London today and what are our event guidelines?'),
        ("📖 Wikipedia Research", 'Search Wikipedia for transformer neural network architecture and explain it.'),
        ("🌐 Live Web Search", 'What are recent breakthroughs in quantum computing?'),
        ("📚 Internal Strategy RAG", 'What does our internal strategy document say about Agentic RAG and data governance?'),
    ]

    if _HAS_RICH and _console is not None:
        table = Table(title="💡 Suggested Example Queries", border_style="blue", box=None)
        table.add_column("Category", style="cyan", width=28)
        table.add_column("Query", style="white")
        for cat, q in examples:
            table.add_row(cat, f'"{q}"')
        _console.print(table)
    else:
        print("\nExample queries to try:")
        for cat, q in examples:
            print(f"  • [{cat}] \"{q}\"")
        print()


def _build_config() -> dict:
    """Read configuration from environment variables and return as a dict."""
    return {
        "groq_api_key":            os.getenv("GROQ_API_KEY", ""),
        "groq_model":              os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
        "tavily_api_key":          os.getenv("TAVILY_API_KEY", ""),
        "openweathermap_api_key":  os.getenv("OPENWEATHERMAP_API_KEY", ""),
        "domain_description":      "internal company documents, valuation forecasts, and policies",
    }

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic RAG with Groq API and real-time tools (finance, weather, web search, Wikipedia).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the interactive Streamlit Web Application in your browser.",
    )
    parser.add_argument(
        "--kb-dir",
        default=os.getenv("KNOWLEDGE_BASE_DIR", "data/knowledge_base"),
        help="Directory containing .pdf, .txt, and .md files to index (default: data/knowledge_base)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Groq model name to use (default: qwen/qwen3.8-27b, or openai/gpt-oss-120b)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Run a single query and exit.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive multi-turn Q&A session.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable conversation memory (each query is independent).",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing of the knowledge base into FAISS.",
    )
    verbose_group = parser.add_mutually_exclusive_group()
    verbose_group.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=True,
        help="Show the agent's reasoning trace (default: on).",
    )
    verbose_group.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="Hide the agent's reasoning trace.",
    )
    return parser.parse_args()
def main() -> None:
    args = _parse_args()

    # Launch Streamlit UI if requested
    if args.ui:
        app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
        print("\n🚀 Launching Agentic RAG Web Application with Streamlit...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
        return

    config = _build_config()

    if args.model:
        config["groq_model"] = args.model

    # Check API keys and display readiness banner
    has_llm = _check_api_keys(config)

    # Step 1: Index / load knowledge base
    if _HAS_RICH and _console is not None:
        _console.print(f"\n[cyan]⚙️  Loading Knowledge Base from '[bold]{args.kb_dir}[/bold]' …[/cyan]")
    else:
        print(f"\n[Setup] Indexing knowledge base from '{args.kb_dir}' …")

    from src.knowledge_indexer import index_knowledge_base, get_indexed_documents_info
    vector_store = index_knowledge_base(
        kb_dir=args.kb_dir,
        index_path=os.path.join(args.kb_dir, ".faiss_index"),
        force_reindex=args.reindex,
    )

    # Step 2: Build tool registry
    from src.tool_registry import build_tool_registry, get_tool_descriptions
    tools = build_tool_registry(vector_store, config)
    print(get_tool_descriptions(tools))

    # Step 3: Instantiate Groq LLM or Simulation
    llm = None
    if has_llm:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model=config["groq_model"],
                groq_api_key=config["groq_api_key"],
                temperature=0,
            )
            print(f"[Setup] Connected to Groq model '{config['groq_model']}'")
        except Exception as exc:
            print(f"[Setup] Notice: Could not connect to Groq ({exc}). Falling back to Simulation Mode.")
            llm = None

    # Step 4: Create Agent
    use_memory = not args.no_memory
    from src.agent import create_agent
    agent = create_agent(tools, llm, memory=use_memory, verbose=args.verbose)

    # Step 5: Formatters
    from src.response_formatter import (
        format_response,
        display_rich_response,
        extract_tools_from_steps,
    )

    def _execute_query(user_query: str) -> None:
        """Run a query through the agent and display the formatted output."""
        start_time = time.time()
        model_display = config["groq_model"] if has_llm and llm is not None else "Simulation Engine"

        if _HAS_RICH and _console is not None:
            _console.print(f"\n[bold yellow]❓ Query:[/bold yellow] [bold white]{user_query}[/bold white]\n")
            with _console.status("[bold cyan]⚡ Groq Agent Reasoning & Selecting Tools...[/bold cyan]"):
                try:
                    result = agent.invoke({"input": user_query})
                    answer = result.get("output", str(result))
                    steps = result.get("intermediate_steps", [])
                    tools_used = extract_tools_from_steps(steps)
                except Exception as exc:
                    answer = f"Agent encountered an error: {exc}"
                    steps = []
                    tools_used = []

            duration = time.time() - start_time
            display_rich_response(
                answer=answer,
                tools_used=tools_used,
                agent_steps=steps,
                show_trace=args.verbose,
                duration_s=duration,
                model_name=model_display,
            )
        else:
            print(f"\n[Query] {user_query}\n")
            try:
                result = agent.invoke({"input": user_query})
                answer = result.get("output", str(result))
                steps = result.get("intermediate_steps", [])
                tools_used = extract_tools_from_steps(steps)
            except Exception as exc:
                answer = f"Agent encountered an error: {exc}"
                tools_used = []

            duration = time.time() - start_time
            print("\n" + format_response(answer, tools_used, steps, duration_s=duration))

    if args.query:
        _execute_query(args.query)

    elif args.interactive:
        _print_example_queries()
        if _HAS_RICH and _console is not None:
            _console.print("[dim]Commands: [bold]/help[/bold] (help), [bold]/tools[/bold] (list tools), [bold]/docs[/bold] (view KB), [bold]/model[/bold] (view model), [bold]/clear[/bold] (clear chat), [bold]/exit[/bold] (quit)[/dim]\n")
        else:
            print("Commands: /help, /tools, /docs, /model, /clear, /exit\n")

        while True:
            try:
                user_input = input("\nYou > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd in {"quit", "exit", "q", "/exit", "/quit"}:
                print("Goodbye!")
                break
            elif cmd in {"/help", "help"}:
                _print_example_queries()
                continue
            elif cmd in {"/tools", "tools"}:
                print("\n" + get_tool_descriptions(tools))
                continue
            elif cmd in {"/docs", "/kb", "docs"}:
                doc_infos = get_indexed_documents_info(args.kb_dir)
                print(f"\nKnowledge Base Directory: '{args.kb_dir}' ({len(doc_infos)} files)")
                for doc in doc_infos:
                    print(f"  • {doc['name']} ({doc['size_kb']:.1f} KB)")
                continue
            elif cmd in {"/model", "model"}:
                print(f"\nCurrent Engine: {'Groq LLM (' + config['groq_model'] + ')' if has_llm else 'Simulation Mode'}")
                continue
            elif cmd in {"/clear", "clear"}:
                if hasattr(agent, "history"):
                    agent.history.clear()
                print("Conversation history cleared.")
                continue

            _execute_query(user_input)

    else:
        print("\nNo mode selected. To get started:")
        print("  • Launch Web UI  : python main.py --ui")
        print("  • Interactive CLI: python main.py --interactive")
        print("  • Single Query   : python main.py --query \"What is AAPL stock price?\"")
        _print_example_queries()
        print("Run with --help for all CLI options.")


if __name__ == "__main__":
    main()
