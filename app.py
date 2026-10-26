"""
app.py — Interactive Web Application for Agentic RAG with Groq API & Real-Time Tools

A high-performance, polished web interface featuring:
  • Conversational Agent Chat with multi-turn memory powered by Groq LPU inference
  • Expandable Step-by-Step Reasoning Trace visualizer (ReAct: Reason -> Act -> Observe)
  • Dynamic Tool Activation Badges & Execution Latency Metrics
  • Knowledge Base Studio (document upload, file inspector, live FAISS re-indexing, search sandbox)
  • Direct Tool Playgrounds (Finance, Weather, Wikipedia, Web Search, Knowledge Base)
  • Architecture visualizer and real-time tool readiness diagnostics
"""

import os
import sys
import time
import json
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# Ensure local modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

# Reconfigure stdout/stderr on Windows to avoid charmap codec issues
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.knowledge_indexer import (
    index_knowledge_base,
    search_knowledge_base_with_metadata,
    get_indexed_documents_info,
)
from src.tool_registry import build_tool_registry, get_tool_descriptions
from src.agent import create_agent
from src.response_formatter import extract_tools_from_steps, TOOL_BADGES, get_tool_display_name


# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic RAG — Real-Time Multi-Tool Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling: Modern Glassmorphic Dark Aesthetic
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main gradient title */
    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #10b981 0%, #38bdf8 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.02rem;
        line-height: 1.5;
        margin-bottom: 1.25rem;
    }

    /* Status cards & badges */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 9999px;
        padding: 0.35rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 600;
        color: #38bdf8;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }

    .status-pill-green {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 9999px;
        padding: 0.35rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 600;
        color: #34d399;
    }

    .status-pill-yellow {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 9999px;
        padding: 0.35rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 600;
        color: #fbbf24;
    }

    .status-pill-muted {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 9999px;
        padding: 0.35rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 500;
        color: #94a3b8;
    }

    .tool-pill {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 6px;
        margin-right: 0.35rem;
        margin-top: 0.35rem;
        background-color: #0f172a;
        color: #38bdf8;
        border: 1px solid #1e293b;
    }

    .latency-tag {
        display: inline-flex;
        align-items: center;
        font-size: 0.75rem;
        color: #64748b;
        margin-left: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Content cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .result-chunk {
        background: rgba(30, 41, 59, 0.6);
        border-left: 3px solid #38bdf8;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .metric-badge {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


def _get_config_value(key: str, default: str = "") -> str:
    """Read configuration from environment variables or st.secrets (for cloud deployments)."""
    val = os.getenv(key)
    if val:
        return val
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "config" not in st.session_state:
    default_groq_key = _get_config_value("GROQ_API_KEY", "")
    st.session_state.config = {
        "groq_api_key": default_groq_key,
        "groq_model": _get_config_value("GROQ_MODEL", "qwen/qwen3.8-27b"),
        "tavily_api_key": _get_config_value("TAVILY_API_KEY", ""),
        "openweathermap_api_key": _get_config_value("OPENWEATHERMAP_API_KEY", ""),
        "kb_dir": _get_config_value("KNOWLEDGE_BASE_DIR", "data/knowledge_base"),
        "use_memory": True,
        "verbose": True,
        "temperature": 0.0,
        "use_simulation": not bool(default_groq_key),
    }


# -----------------------------------------------------------------------------
# Cached Resources: FAISS Vector Store & Agent
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_vector_store(kb_dir: str, force_reindex: bool = False):
    """Load or build the FAISS vector store index."""
    index_path = os.path.join(kb_dir, ".faiss_index")
    return index_knowledge_base(kb_dir=kb_dir, index_path=index_path, force_reindex=force_reindex)


def get_agent_instance(config: dict, force_rebuild_kb: bool = False):
    """Assemble tools and construct the Groq ChatGroq agent or simulation executor."""
    vector_store = get_vector_store(config["kb_dir"], force_reindex=force_rebuild_kb)
    tools = build_tool_registry(vector_store, config)

    llm = None
    if not config.get("use_simulation") and config.get("groq_api_key"):
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model=config["groq_model"],
                groq_api_key=config["groq_api_key"],
                temperature=config.get("temperature", 0.0),
            )
        except Exception as e:
            st.warning(f"Could not connect to Groq ({e}). Falling back to simulation mode.")
            llm = None

    agent = create_agent(
        tools=tools,
        llm=llm,
        memory=config.get("use_memory", True),
        verbose=config.get("verbose", True),
    )
    return agent, tools, vector_store


# -----------------------------------------------------------------------------
# Sidebar: System Controls & Groq Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Inference Engine")

    engine_modes = ["Autonomous Agent (Groq API)", "Interactive Simulation Mode (Zero Key Needed)"]
    current_engine_idx = 1 if st.session_state.config.get("use_simulation") else 0
    selected_mode = st.radio("Execution Mode", engine_modes, index=current_engine_idx)
    st.session_state.config["use_simulation"] = (selected_mode == engine_modes[1])

    if not st.session_state.config["use_simulation"]:
        groq_api_input = st.text_input(
            "Groq API Key",
            value=st.session_state.config["groq_api_key"],
            type="password",
            help="Get your free key from https://console.groq.com/keys",
        )
        st.session_state.config["groq_api_key"] = groq_api_input
        st.caption("🔑 [Get free Groq API key](https://console.groq.com/keys)")

        groq_models = [
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "allam-2-7b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ]
        curr_m = st.session_state.config["groq_model"]
        m_idx = groq_models.index(curr_m) if curr_m in groq_models else 0
        st.session_state.config["groq_model"] = st.selectbox(
            "Groq Model",
            groq_models,
            index=m_idx,
            help="qwen/qwen3.8-27b offers state-of-the-art tool routing accuracy and ultra-low latency.",
        )

        st.session_state.config["temperature"] = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Set to 0.0 for deterministic tool routing and factual grounding.",
        )
    else:
        st.info("💡 **Simulation Mode Active:** FAISS vector store, live Yahoo Finance market metrics, Wikipedia, and weather execute natively without requiring any external API key.")

    st.markdown("---")
    st.markdown("### 🔑 Auxiliary Tool Keys")

    tavily_key = st.text_input(
        "Tavily API Key (Web Search)",
        value=st.session_state.config["tavily_api_key"],
        type="password",
        help="Enables real-time web crawling via Tavily. Free tier gives 1,000 queries/mo.",
    )
    st.session_state.config["tavily_api_key"] = tavily_key

    owm_key = st.text_input(
        "OpenWeatherMap Key (Weather)",
        value=st.session_state.config["openweathermap_api_key"],
        type="password",
        help="Enables live meteorological observations. Free tier gives 60 calls/min.",
    )
    st.session_state.config["openweathermap_api_key"] = owm_key

    st.markdown("---")
    st.markdown("### 🧠 Agent Parameters")
    st.session_state.config["use_memory"] = st.toggle("Sliding Window Memory (k=5)", value=st.session_state.config["use_memory"])
    st.session_state.config["verbose"] = st.toggle("Show Reasoning Traces", value=st.session_state.config["verbose"])

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col_btn2:
        if st.session_state.messages:
            chat_export = json.dumps(st.session_state.messages, indent=2, default=str)
            st.download_button(
                "💾 Export",
                data=chat_export,
                file_name="agentic_rag_session.json",
                mime="application/json",
                use_container_width=True,
            )


# -----------------------------------------------------------------------------
# Initialize Agent Instance
# -----------------------------------------------------------------------------
try:
    agent, tools, vector_store = get_agent_instance(st.session_state.config)
except Exception as exc:
    st.error(f"Error initializing Agentic RAG system: {exc}")
    st.stop()


# -----------------------------------------------------------------------------
# Header & Dynamic Tool Status Bar
# -----------------------------------------------------------------------------
st.markdown('<div class="hero-title">⚡ Agentic RAG — Real-Time Multi-Tool Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Autonomous reasoning pipeline powered by <b>Groq LPU Inference</b>. Seamlessly orchestrates internal corporate documents with live stock feeds, current weather, Wikipedia, and web search.</div>',
    unsafe_allow_html=True,
)

# Tool Readiness Status Bar
col_b1, col_b2, col_b3, col_b4, col_b5, col_b6 = st.columns(6)

kb_docs = get_indexed_documents_info(st.session_state.config["kb_dir"])
col_b1.markdown(f'<div class="status-pill-green">📚 KB: {len(kb_docs)} Docs</div>', unsafe_allow_html=True)
col_b2.markdown('<div class="status-pill-green">📈 Stocks: yfinance</div>', unsafe_allow_html=True)
col_b3.markdown('<div class="status-pill-green">📖 Wikipedia: Live</div>', unsafe_allow_html=True)

if st.session_state.config.get("openweathermap_api_key"):
    col_b4.markdown('<div class="status-pill-green">🌦️ Weather: Live API</div>', unsafe_allow_html=True)
else:
    col_b4.markdown('<div class="status-pill-yellow">🌦️ Weather: Mock Mode</div>', unsafe_allow_html=True)

if st.session_state.config.get("tavily_api_key"):
    col_b5.markdown('<div class="status-pill-green">🌐 Web: Live Tavily</div>', unsafe_allow_html=True)
else:
    col_b5.markdown('<div class="status-pill-muted">🌐 Web: Mock Mode</div>', unsafe_allow_html=True)

if st.session_state.config.get("use_simulation") or not st.session_state.config.get("groq_api_key"):
    col_b6.markdown('<div class="status-pill-yellow">⚡ Groq: Simulation</div>', unsafe_allow_html=True)
else:
    col_b6.markdown(f'<div class="status-pill-green">⚡ Groq: {st.session_state.config["groq_model"].split("-")[0].capitalize()}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
tab_chat, tab_kb, tab_playground, tab_arch = st.tabs([
    "💬 Agent Chat",
    "📚 Knowledge Base Studio",
    "🧪 Tool Playgrounds",
    "🏗️ Architecture & Insights",
])


# -----------------------------------------------------------------------------
# TAB 1: Conversational Agent Chat
# -----------------------------------------------------------------------------
with tab_chat:
    st.markdown("**Quick Prompts (Cross-Tool Scenarios):**")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)

    prompt_to_send = None
    if q_col1.button("📈 AAPL Stock + Q3 Valuation Forecast", use_container_width=True):
        prompt_to_send = "What is AAPL's current stock price and how does it compare to our internal valuation forecast?"
    if q_col2.button("🌦️ London Weather + Event Contingency", use_container_width=True):
        prompt_to_send = "What's the weather in London today and should we proceed per our corporate event guidelines?"
    if q_col3.button("📖 Wikipedia: Transformer Architecture", use_container_width=True):
        prompt_to_send = "Search Wikipedia for transformer neural network architecture and explain how it works."
    if q_col4.button("📋 AI Strategy Directive & Data Privacy", use_container_width=True):
        prompt_to_send = "What does our internal strategy document say about Agentic RAG and data governance?"

    st.markdown("---")

    # Render previous conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "⚡"):
            st.markdown(msg["content"])

            # Footer with tools and latency
            footer_parts = []
            if msg.get("tools_used"):
                pills = "".join([f'<span class="tool-pill">{get_tool_display_name(t)}</span>' for t in msg["tools_used"]])
                footer_parts.append(pills)

            if msg.get("latency_s") is not None:
                footer_parts.append(f'<span class="latency-tag">⏱️ {msg["latency_s"]:.2f}s</span>')

            if footer_parts:
                st.markdown(f"<div style='margin-top: 0.5rem;'>{' '.join(footer_parts)}</div>", unsafe_allow_html=True)

            # Expandable reasoning trace
            if msg.get("steps") and st.session_state.config["verbose"]:
                with st.expander(f"🔍 Reasoning & Tool Execution Trace ({len(msg['steps'])} step(s))", expanded=False):
                    for idx, step in enumerate(msg["steps"], 1):
                        try:
                            action, obs = step[0], step[1]
                            t_name = getattr(action, "tool", "tool")
                            t_input = getattr(action, "tool_input", "")
                            t_log = getattr(action, "log", "")
                            st.markdown(f"**Step {idx} — Tool Activated:** `{t_name}`")
                            if t_log:
                                thought_line = t_log.split("\n")[0].replace("Thought:", "").strip()
                                st.caption(f"💭 {thought_line}")
                            st.code(f"Input: {t_input}\nObservation:\n{obs}", language="text")
                        except Exception:
                            st.write(str(step))

    # Chat Input
    user_input = st.chat_input("Ask a question (combines live stocks, weather, Wikipedia, web search, or internal docs)...")
    final_query = prompt_to_send or user_input

    if final_query:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": final_query})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(final_query)

        # Generate Agent Response
        with st.chat_message("assistant", avatar="⚡"):
            start_time = time.time()
            status_container = st.status("🧠 Groq Agent Reasoning & Selecting Tools...", expanded=True)

            with status_container:
                st.write("Evaluating query intent and reading tool schemas...")

                try:
                    result = agent.invoke({"input": final_query})
                    answer = result.get("output", str(result))
                    steps = result.get("intermediate_steps", [])
                    tools_used = extract_tools_from_steps(steps)
                    duration_s = time.time() - start_time

                    if steps:
                        st.write(f"Executed **{len(steps)}** tool call(s): {', '.join(tools_used)}")
                    else:
                        st.write("Resolved directly via internal synthesis.")

                    status_container.update(label=f"✅ Completed in {duration_s:.2f}s", state="complete", expanded=False)
                except Exception as exc:
                    status_container.update(label="❌ Execution Error", state="error", expanded=True)
                    answer = f"Agent encountered an error: {exc}"
                    steps = []
                    tools_used = []
                    duration_s = time.time() - start_time

            # Display final answer
            st.markdown(answer)

            # Display tool pills and latency
            footer_parts = []
            if tools_used:
                pills = "".join([f'<span class="tool-pill">{get_tool_display_name(t)}</span>' for t in tools_used])
                footer_parts.append(pills)
            footer_parts.append(f'<span class="latency-tag">⏱️ {duration_s:.2f}s</span>')
            st.markdown(f"<div style='margin-top: 0.5rem;'>{' '.join(footer_parts)}</div>", unsafe_allow_html=True)

            # Display reasoning trace expander
            if steps and st.session_state.config["verbose"]:
                with st.expander(f"🔍 Reasoning & Tool Execution Trace ({len(steps)} step(s))", expanded=False):
                    for idx, step in enumerate(steps, 1):
                        try:
                            action, obs = step[0], step[1]
                            t_name = getattr(action, "tool", "tool")
                            t_input = getattr(action, "tool_input", "")
                            t_log = getattr(action, "log", "")
                            st.markdown(f"**Step {idx} — Tool Activated:** `{t_name}`")
                            if t_log:
                                thought_line = t_log.split("\n")[0].replace("Thought:", "").strip()
                                st.caption(f"💭 {thought_line}")
                            st.code(f"Input: {t_input}\nObservation:\n{obs}", language="text")
                        except Exception:
                            st.write(str(step))

        # Store assistant message in session history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "tools_used": tools_used,
            "steps": steps,
            "latency_s": duration_s,
        })


# -----------------------------------------------------------------------------
# TAB 2: Knowledge Base Studio
# -----------------------------------------------------------------------------
with tab_kb:
    st.markdown("### 📚 Knowledge Base Studio")
    st.markdown("Manage local documents indexed in FAISS for the `search_knowledge_base` tool.")

    col_kb_l, col_kb_r = st.columns([1, 1])

    with col_kb_l:
        st.markdown("#### 📂 Indexed Documents Inventory")
        kb_path = st.session_state.config["kb_dir"]
        doc_infos = get_indexed_documents_info(kb_path)

        if doc_infos:
            for d in doc_infos:
                with st.expander(f"📄 {d['name']} ({d['size_kb']:.1f} KB)", expanded=False):
                    try:
                        with open(d["path"], "r", encoding="utf-8", errors="replace") as f_preview:
                            content = f_preview.read()
                            st.text_area("File Content", value=content, height=180, disabled=True)
                    except Exception as e:
                        st.caption(f"Preview unavailable: {e}")
        else:
            st.info("No documents found in knowledge base. Upload files on the right.")

        st.markdown("#### 🔄 Vector Store Maintenance")
        if st.button("Re-index Knowledge Base Now", use_container_width=True):
            with st.spinner("Splitting documents, computing local MiniLM embeddings, and persisting FAISS index..."):
                get_vector_store(kb_path, force_reindex=True)
                st.success("FAISS index successfully rebuilt!")
                st.rerun()

    with col_kb_r:
        st.markdown("#### ⬆️ Upload New Documents")
        uploaded_files = st.file_uploader(
            "Drop .txt, .pdf, or .md files to add to the knowledge base",
            type=["txt", "pdf", "md"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            for up in uploaded_files:
                save_dest = os.path.join(kb_path, up.name)
                with open(save_dest, "wb") as f_out:
                    f_out.write(up.getbuffer())
                st.success(f"Saved: {up.name}")
            st.info("Click 'Re-index Knowledge Base Now' on the left to embed new files into FAISS.")

    st.markdown("---")
    st.markdown("#### 🔍 Semantic Search Sandbox")
    st.caption("Inspect what the FAISS index retrieves for any test query with similarity scores.")

    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        test_search_query = st.text_input("Semantic Query", value="AAPL valuation forecast")
    with col_s2:
        top_k = st.slider("Top Chunks", min_value=1, max_value=8, value=3)

    if test_search_query:
        search_results = search_knowledge_base_with_metadata(test_search_query, vector_store, k=top_k)
        if search_results:
            for idx, res in enumerate(search_results, 1):
                score_str = f" | Score: {res['score']:.4f}" if res.get("score") is not None else ""
                st.markdown(f"**Chunk {idx}** — Source: `{res['source']}`{score_str}")
                st.markdown(f'<div class="result-chunk">{res["content"]}</div>', unsafe_allow_html=True)
        else:
            st.warning("No matches found.")


# -----------------------------------------------------------------------------
# TAB 3: Tool Playgrounds
# -----------------------------------------------------------------------------
with tab_playground:
    st.markdown("### 🧪 Direct Tool Playgrounds")
    st.markdown("Test each tool independently without invoking the full agent reasoning loop.")

    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.markdown("#### 📈 Live Finance Tool (yfinance)")
        st.caption("Quick Presets:")
        f_presets = ["NVDA", "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
        f_cols = st.columns(6)
        selected_ticker = "NVDA"
        for i, tick in enumerate(f_presets):
            if f_cols[i].button(tick, key=f"f_pre_{tick}", use_container_width=True):
                selected_ticker = tick

        ticker_input = st.text_input("Ticker or Company", value=selected_ticker, key="play_ticker")
        if st.button("Fetch Live Stock Data", use_container_width=True):
            f_tool = next((t for t in tools if t.name == "get_stock_data"), None)
            if f_tool:
                with st.spinner("Fetching Yahoo Finance data..."):
                    res = f_tool.func(ticker_input)
                    st.success(res)

        st.markdown("---")
        st.markdown("#### 🌦️ Current Weather Tool")
        st.caption("Quick City Presets:")
        w_presets = ["London", "Tokyo", "New York", "San Francisco", "Paris"]
        w_cols = st.columns(5)
        selected_city = "Tokyo"
        for i, city in enumerate(w_presets):
            if w_cols[i].button(city, key=f"w_pre_{city}", use_container_width=True):
                selected_city = city

        city_input = st.text_input("City Name", value=selected_city, key="play_city")
        if st.button("Fetch Weather Observation", use_container_width=True):
            w_tool = next((t for t in tools if t.name == "get_weather"), None)
            if w_tool:
                with st.spinner("Fetching meteorological data..."):
                    res = w_tool.func(city_input)
                    st.info(res)

    with p_col2:
        st.markdown("#### 📖 Wikipedia Encyclopedia Tool")
        st.caption("Quick Topic Presets:")
        wiki_presets = ["Artificial intelligence", "Large language model", "Quantum computing"]
        wiki_cols = st.columns(3)
        selected_wiki = "Artificial intelligence"
        for i, top in enumerate(wiki_presets):
            if wiki_cols[i].button(top, key=f"k_pre_{top}", use_container_width=True):
                selected_wiki = top

        wiki_input = st.text_input("Wikipedia Topic", value=selected_wiki, key="play_wiki")
        if st.button("Search Wikipedia", use_container_width=True):
            k_tool = next((t for t in tools if t.name == "search_wikipedia"), None)
            if k_tool:
                with st.spinner("Querying Wikipedia API..."):
                    res = k_tool.func(wiki_input)
                    st.markdown(res)

        st.markdown("---")
        st.markdown("#### 🌐 Live Web Search Tool (Tavily)")
        web_input = st.text_input("Web Search Query", value="latest breakthroughs in quantum computing", key="play_web")
        if st.button("Execute Web Search", use_container_width=True):
            s_tool = next((t for t in tools if t.name == "web_search"), None)
            if s_tool:
                with st.spinner("Searching the web..."):
                    res = s_tool.func(web_input)
                    st.code(res, language="text")


# -----------------------------------------------------------------------------
# TAB 4: Architecture & Insights
# -----------------------------------------------------------------------------
with tab_arch:
    st.markdown("### 🏗️ Agentic RAG System Architecture")
    st.markdown("""
Unlike traditional RAG pipelines that blindly query a vector database on every prompt,
an **Agentic RAG** system uses an LLM reasoning loop to dynamically choose **which tool to activate**
based on user intent, cross-referencing internal static documents with live real-time APIs.
""")

    st.code("""
                                User Question
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Groq LLM Reasoning      │  ← Evaluates intent & reads tool schemas
                        │ (llama-3.3-70b-versatile) │
                        └─────────────┬─────────────┘
                                      │
         Autonomous Tool-Calling Loop: (Reason → Act → Observe)
                                      │
        ┌─────────────────────────────┴─────────────────────────────┐
        │                        Tool Registry                      │
        │                                                           │
        │   ┌──────────────────────┐     ┌──────────────────────┐   │
        │   │ search_knowledge_base│     │    get_stock_data    │   │
        │   │  (FAISS + MiniLM-L6) │     │   (Yahoo Finance)    │   │
        │   └──────────────────────┘     └──────────────────────┘   │
        │                                                           │
        │   ┌──────────────────────┐     ┌──────────────────────┐   │
        │   │      web_search      │     │     get_weather      │   │
        │   │     (Tavily API)     │     │  (OpenWeatherMap)    │   │
        │   └──────────────────────┘     └──────────────────────┘   │
        │                                                           │
        │   ┌──────────────────────┐                                │
        │   │   search_wikipedia   │                                │
        │   │ (Wikipedia REST API) │                                │
        │   └──────────────────────┘                                │
        └─────────────────────────────┬─────────────────────────────┘
                                      │
                                      ▼
                   Final Grounded Answer + Provenance Badges
""", language="text")

    st.markdown("""
#### Core Engineering Principles:
1. **High-Speed Groq LPU Inference:** Provides near-instantaneous tool evaluation and synthesis, eliminating latency bottlenecks in multi-step ReAct loops.
2. **Local Vector Privacy:** Document vectorization runs entirely on CPU using `sentence-transformers/all-MiniLM-L6-v2`. Sensitive internal files are never shared with external embedding APIs.
3. **Intent-Driven Routing:** Tool descriptions serve as strict routing specifications, avoiding superfluous retrieval when answering live questions like stock prices or weather.
4. **Resilient Dual Modes:** Runs seamlessly either with a live Groq API key or in high-fidelity offline simulation mode.
""")
