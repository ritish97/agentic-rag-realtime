"""
src/agent.py

Assembles the LangChain agent executor that ties together the Groq LLM,
real-time tools, and conversation memory.

THE AGENTIC RAG LOOP:
    Every time the agent receives a question it cycles through:
      1. REASON  — LLM analyzes question intent and decides which tool(s) to call
                   using native function calling.
      2. ACT     — The selected tool executes with structured arguments.
      3. OBSERVE — The observation/result is returned into context.
      4. REPEAT  — The agent reflects on observations; loops until sufficient evidence
                   is gathered to produce a cited, definitive answer.
"""

import re
import sys
from typing import List, Optional, Any, Dict
from collections import deque

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent  # type: ignore

try:
    from langchain.memory import ConversationBufferWindowMemory
except ImportError:
    try:
        from langchain_classic.memory import ConversationBufferWindowMemory  # type: ignore
    except ImportError:
        ConversationBufferWindowMemory = None  # type: ignore

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.agents import AgentAction


# System prompt tuned for high-speed Groq inference & multi-tool coordination.
_SYSTEM_PROMPT = """You are an advanced enterprise research assistant equipped with real-time multi-tool capabilities.
You have access to:
- search_knowledge_base: Internal company documents, valuation forecasts, DCF models, corporate event policies, and AI governance directives.
- get_stock_data: Live equity prices, 52-week ranges, P/E ratios, and market caps from Yahoo Finance.
- get_weather: Real-time global meteorological data, current temperatures (°C and °F), humidity, and wind conditions.
- search_wikipedia: Encyclopaedic definitions, historical context, scientific concepts, and technical summaries.
- web_search: Live web search for breaking events, current market news, and public information.

Routing & Synthesis Principles:
1. Internal Policies & Valuation: For internal company forecasts, price targets, corporate guidelines, or AI strategy -> call `search_knowledge_base`.
2. Live Market Equities: For current stock prices or ticker metrics -> call `get_stock_data`.
3. Weather & Meteorology: For weather reports, city forecasts, or event conditions -> call `get_weather`.
4. Encyclopaedic Knowledge: For definitions, scientific theories, or general knowledge -> call `search_wikipedia`.
5. Breaking Events & Live Web: For recent news or real-time web questions -> call `web_search`.
6. Multi-Tool Synthesis: For mixed questions (e.g. comparing live stock price to internal valuation targets, or evaluating weather against corporate event safety guidelines) -> call ALL required tools sequentially before formulating your final grounded response.
7. Always cite which tools were activated and provide clear, structured provenance in your answer.
Think step by step before invoking tools."""


class SimulatedAgentExecutor:
    """
    High-fidelity interactive simulator that executes real tools and produces
    grounded multi-step reasoning traces even when an external LLM API key
    is not configured.
    """

    def __init__(self, tools: List[Tool], memory: bool = True, verbose: bool = True):
        self.tools = {t.name: t for t in tools}
        self.memory_enabled = memory
        self.verbose = verbose
        self.history = deque(maxlen=10)

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs.get("input", "")
        intermediate_steps = []
        q_lower = query.lower()

        # Step 1: Detect and invoke Finance tool
        stock_matches = re.findall(
            r"\b(AAPL|MSFT|NVDA|GOOGL|GOOG|AMZN|TSLA|META|NFLX|INTC|AMD|SPOT|UBER|CRM|ORCL|BRK|V|JPM|WMT|ADBE|QCOM|AVGO)\b",
            query,
            re.IGNORECASE,
        )
        company_names = [
            "apple", "microsoft", "nvidia", "google", "alphabet", "amazon", "tesla",
            "meta", "netflix", "intel", "amd", "spotify", "uber", "salesforce", "oracle",
        ]
        detected_company = next((name for name in company_names if name in q_lower), None)
        target_ticker = stock_matches[0].upper() if stock_matches else (detected_company if detected_company else None)

        finance_obs = None
        if target_ticker or any(w in q_lower for w in ["stock", "ticker", "share price", "trading at", "market cap", "p/e"]):
            symbol = target_ticker or "AAPL"
            tool = self.tools.get("get_stock_data")
            if tool:
                if self.verbose:
                    print(f"\n[Simulator] Invoking: `get_stock_data` with `{symbol}`")
                finance_obs = tool.func(symbol)
                action = AgentAction(
                    tool="get_stock_data",
                    tool_input=symbol,
                    log=f"Thought: The user is inquiring about equity metrics for '{symbol}'. Invoking live market tool to inspect pricing and valuation ratios."
                )
                intermediate_steps.append((action, finance_obs))

        # Step 2: Detect and invoke Weather tool
        weather_obs = None
        cities = [
            "london", "tokyo", "paris", "new york", "san francisco", "berlin",
            "singapore", "seattle", "sydney", "toronto", "mumbai", "chicago",
            "dubai", "boston", "amsterdam", "los angeles",
        ]
        city_match = next((c for c in cities if c in q_lower), None)

        if city_match or any(w in q_lower for w in ["weather", "forecast", "rain", "temperature", "degrees", "celsius", "fahrenheit"]):
            target_city = city_match.title() if city_match else "London"
            tool = self.tools.get("get_weather")
            if tool:
                if self.verbose:
                    print(f"\n[Simulator] Invoking: `get_weather` with `{target_city}`")
                weather_obs = tool.func(target_city)
                action = AgentAction(
                    tool="get_weather",
                    tool_input=target_city,
                    log=f"Thought: Request involves meteorological conditions for '{target_city}'. Querying weather service for temperature, precipitation, and wind speeds."
                )
                intermediate_steps.append((action, weather_obs))

        # Step 3: Detect and invoke RAG Knowledge Base tool
        rag_obs = None
        rag_keywords = [
            "internal", "valuation", "forecast", "policy", "guideline", "event",
            "strategy", "roadmap", "dcf", "target", "sop", "governance", "relocation",
            "threshold", "fair value", "contingency",
        ]
        if any(w in q_lower for w in rag_keywords) or (finance_obs and "valuation" in q_lower) or (weather_obs and ("policy" in q_lower or "guideline" in q_lower or "event" in q_lower)):
            rag_query = query
            tool = self.tools.get("search_knowledge_base")
            if tool:
                if self.verbose:
                    print(f"\n[Simulator] Invoking: `search_knowledge_base` with `{rag_query}`")
                rag_obs = tool.func(rag_query)
                action = AgentAction(
                    tool="search_knowledge_base",
                    tool_input=rag_query,
                    log="Thought: Query requires internal proprietary knowledge from stored enterprise documentation (FAISS index)."
                )
                intermediate_steps.append((action, rag_obs))

        # Step 4: Detect Wikipedia tool
        wiki_obs = None
        wiki_triggers = ["wikipedia", "what is", "who is", "explain", "transformer", "neural network", "history of", "definition of"]
        if any(w in q_lower for w in wiki_triggers) and not finance_obs and not weather_obs:
            wiki_topic = query
            for prefix in ["search wikipedia for", "what is a", "what is", "who is", "explain", "tell me about"]:
                if wiki_topic.lower().startswith(prefix):
                    wiki_topic = wiki_topic[len(prefix):].strip()
            wiki_topic = wiki_topic.rstrip("?.!").strip()
            tool = self.tools.get("search_wikipedia")
            if tool:
                if self.verbose:
                    print(f"\n[Simulator] Invoking: `search_wikipedia` with `{wiki_topic}`")
                wiki_obs = tool.func(wiki_topic)
                action = AgentAction(
                    tool="search_wikipedia",
                    tool_input=wiki_topic,
                    log=f"Thought: Encyclopaedic definition requested for topic: '{wiki_topic}'. Looking up reference entry."
                )
                intermediate_steps.append((action, wiki_obs))

        # Step 5: Web Search if requested or fallback
        web_obs = None
        if any(w in q_lower for w in ["web search", "news", "latest", "recent", "breaking news", "articles"]) and not intermediate_steps:
            tool = self.tools.get("web_search")
            if tool:
                if self.verbose:
                    print(f"\n[Simulator] Invoking: `web_search` with `{query}`")
                web_obs = tool.func(query)
                action = AgentAction(
                    tool="web_search",
                    tool_input=query,
                    log="Thought: Real-time web search required for recent updates."
                )
                intermediate_steps.append((action, web_obs))

        # Synthesize structured response
        response_sections = []

        if finance_obs and rag_obs:
            response_sections.append(
                f"### 📊 Market & Internal Valuation Comparison\n\n"
                f"- **Live Market Data:** {finance_obs}\n"
                f"- **Internal Knowledge Base Findings:**\n{rag_obs}\n\n"
                f"**Strategic Synthesis:**\n"
                f"Cross-referencing live market pricing against our internal discounted cash flow (DCF) fair value "
                f"estimates demonstrates how the stock is currently trading relative to internal target bands. "
                f"Review internal sensitivity cases (Bull vs. Bear) and portfolio allocation directives before adjusting weighting."
            )
        elif weather_obs and rag_obs:
            response_sections.append(
                f"### 🌦️ Meteorological & Corporate Event Policy Assessment\n\n"
                f"- **Current Meteorological Observation:** {weather_obs}\n"
                f"- **Applicable Event Safety SOP Guidelines:**\n{rag_obs}\n\n"
                f"**Operational Recommendation:**\n"
                f"Cross-referencing live weather metrics against corporate policy SOP-OPS-042:\n"
                f"1. Evaluate precipitation probability (<25% Green Condition, 25-49% Yellow Precaution, >=50% Red Relocation).\n"
                f"2. Inspect sustained wind limits (<=25 km/h Safe, 26-40 km/h Reinforced, >40 km/h Mandatory Indoor Transition).\n"
                f"3. Ensure heat index hydration or patio heater guidelines are enacted according to ambient temperature thresholds."
            )
        elif finance_obs:
            response_sections.append(f"### 📈 Live Financial Intelligence\n\n{finance_obs}")
        elif weather_obs:
            response_sections.append(f"### 🌦️ Real-Time Weather Observation\n\n{weather_obs}")
        elif rag_obs:
            response_sections.append(f"### 📚 Internal Knowledge Base Retrieval\n\n{rag_obs}")
        elif wiki_obs:
            response_sections.append(f"### 📖 Encyclopaedic Summary\n\n{wiki_obs}")
        elif web_obs:
            response_sections.append(f"### 🌐 Live Web Search Intelligence\n\n{web_obs}")
        else:
            # Fallback default RAG search
            tool = self.tools.get("search_knowledge_base")
            if tool:
                rag_obs = tool.func(query)
                action = AgentAction(
                    tool="search_knowledge_base",
                    tool_input=query,
                    log="Thought: Searching internal documents for relevant context."
                )
                intermediate_steps.append((action, rag_obs))
                response_sections.append(f"### 📚 Knowledge Base Retrieval\n\n{rag_obs}")
            else:
                response_sections.append("Your query has been evaluated across all registered real-time tools.")

        output_text = "\n\n".join(response_sections)

        if self.memory_enabled:
            self.history.append({"query": query, "output": output_text})

        return {
            "output": output_text,
            "intermediate_steps": intermediate_steps,
        }


def create_agent(
    tools: List[Tool],
    llm: Optional[Any] = None,
    memory: bool = True,
    verbose: bool = True,
) -> Any:
    """
    Build and return an agent executor wired to the provided tools.

    If `llm` is None or 'simulation', returns a SimulatedAgentExecutor that
    executes tools natively without requiring external API keys.

    Args:
        tools:   List of LangChain Tool objects from tool_registry.
        llm:     An instantiated ChatGroq model or None.
        memory:  If True, enables sliding-window conversation memory.
        verbose: If True, prints reasoning traces to stdout.

    Returns:
        A configured AgentExecutor or SimulatedAgentExecutor.
    """
    if llm is None or getattr(llm, "is_mock", False) or str(llm).lower() == "simulation":
        return SimulatedAgentExecutor(tools=tools, memory=memory, verbose=verbose)

    # --- Memory ---
    mem: Optional[Any] = None
    if memory and ConversationBufferWindowMemory is not None:
        try:
            mem = ConversationBufferWindowMemory(
                k=5,
                memory_key="chat_history",
                return_messages=True,
                output_key="output",
            )
        except Exception:
            mem = None

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_kwargs: Dict[str, Any] = {
        "agent": agent,
        "tools": tools,
        "verbose": verbose,
        "handle_parsing_errors": True,
        "max_iterations": 8,
        "return_intermediate_steps": True,
    }
    if mem is not None:
        agent_kwargs["memory"] = mem

    agent_executor = AgentExecutor(**agent_kwargs)
    return agent_executor


def run_agent_query(query: str, agent: Any) -> str:
    """
    Submit a query to the agent and return the final answer string.
    """
    try:
        result = agent.invoke({"input": query})
        return result.get("output", str(result))
    except Exception as exc:
        return f"Agent encountered an error: {exc}"
