# Agentic RAG with Real-Time Multi-Tool Intelligence

An autonomous Agentic RAG system powered by the **Groq API** (LPU Inference Engine) that dynamically decides **which tool to activate** for every question — retrieving private enterprise knowledge via local FAISS vector search, fetching live stock metrics from Yahoo Finance, checking real-time weather observations, looking up Wikipedia definitions, or crawling the live web.

Includes both a **modern, interactive Web Application** (Streamlit) and an **enhanced terminal interface** (Rich) with full reasoning trace inspection, document management, and interactive playgrounds.

---

## ⚡ Agentic RAG vs Standard RAG

| Traditional RAG | Agentic RAG |
|---|---|
| ❌ Always executes FAISS retrieval regardless of question | ✅ Evaluates intent and dynamically picks the exact tool needed |
| ❌ Single retrieval step | ✅ Multi-step autonomous reasoning loops (ReAct: Reason → Act → Observe) |
| ❌ Limited strictly to static stored documents | ✅ Cross-references internal knowledge with live data |
| ❌ Cannot answer real-time questions (stocks, weather, news) | ✅ Seamlessly queries live financial, weather, and web APIs |
| ❌ Fixed deterministic code path | ✅ Dynamic, intent-driven execution with provenance citations |
| ❌ High latency on multi-step reasoning | ✅ Blazing-fast inference powered by Groq LPU technology |

**When to use Agentic RAG:** When users ask questions that require combining internal private knowledge with live data (e.g. *"What is AAPL's current market price and how does it compare to our internal Q3 valuation forecast?"* or *"What's the weather in London today and should we proceed per our corporate event guidelines?"*).

---

## 🏗️ Architecture

```
                                User Question
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │    Groq LLM Reasoning     │  ← Evaluates intent & reads tool schemas
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
```

---

## 🛠️ Integrated Tool Suite

| Tool | Provider | Description | API Key Required? |
|---|---|---|---|
| **`search_knowledge_base`** | Local FAISS + MiniLM-L6 | Searches enterprise documents (.pdf, .txt, .md) | ❌ Free / Runs locally |
| **`get_stock_data`** | Yahoo Finance (yfinance) | Real-time stock prices, 52W high/low, P/E, market cap | ❌ Free / Built-in |
| **`search_wikipedia`** | Wikipedia REST API | Encyclopaedic definitions, historical context, summaries | ❌ Free / Built-in |
| **`get_weather`** | OpenWeatherMap | Current weather conditions, temperature (°C/°F), wind | ⚠️ Optional (has mock mode) |
| **`web_search`** | Tavily Search | Live web search optimized for LLM context windows | ⚠️ Optional (has mock mode) |

---

## 🖥️ User Interfaces

### 🌟 1. Interactive Web Application (Streamlit)

Launch the browser application:

```bash
python main.py --ui
# or directly:
streamlit run app.py
```

**Web App Features:**
- **Conversational Chat**: Interactive multi-turn chat with quick-prompt starter chips and conversation memory.
- **Reasoning Trace Inspector**: Expandable step-by-step accordion showing the exact tool calls, thought process, parameters, and observations.
- **Execution Latency Tracker**: Live display of tool duration and response time.
- **Knowledge Base Studio**: Inspect indexed documents, preview file contents, upload new `.pdf`, `.txt`, and `.md` files, and trigger live FAISS re-indexing.
- **Semantic Search Sandbox**: Test similarity queries directly to inspect chunk retrieval and source file metadata with relevance scores.
- **Tool Playgrounds**: Direct test benches with one-click presets for Stock Data, Weather, Wikipedia, and Web Search.
- **Architecture Visualizer**: Interactive breakdown of ReAct execution and tool schemas.

---

### 💻 2. Terminal Interface (Rich CLI)

#### Interactive Session with Live Rich Formatting:
```bash
python main.py --interactive
```

Available slash commands inside interactive mode:
- `/help` — Display example queries and command list
- `/tools` — View all registered tools and their functional descriptions
- `/docs` — View list of indexed documents in the knowledge base
- `/model` — Display current engine and active Groq model
- `/clear` — Clear conversation history
- `/exit` — Exit session

#### Single-Shot Query:
```bash
# Cross-reference live stock with internal Q3 valuation
python main.py --query "What is AAPL's current price and how does it compare to our internal valuation?"

# Meteorological event safety check
python main.py --query "What is the weather in London today and what are our corporate event guidelines?"
```

#### CLI Options:
```bash
python main.py --help

  --ui            Launch the Streamlit Web Application in browser
  --interactive   Start an interactive multi-turn terminal session
  --query TEXT    Run a single query and output formatted response
  --model NAME    Groq model name (default: llama-3.3-70b-versatile)
  --no-memory     Disable conversation memory (stateless mode)
  --reindex       Force rebuild of the local FAISS index
  --no-verbose    Hide intermediate step-by-step reasoning traces
```

---

## 📁 Preloaded Enterprise Knowledge Base

The `data/knowledge_base/` folder comes preloaded with sample enterprise documents so multi-tool queries work immediately out of the box:

1. **`financial_forecasts_q3.txt`**: Internal discounted cash flow (DCF) fair value estimates, 12-month price targets, and analyst ratings for tech equities (AAPL, MSFT, NVDA, GOOGL).
2. **`corporate_event_weather_policy.txt`**: Operational weather thresholds, rain relocation procedures, and wind/heat safety protocols.
3. **`enterprise_ai_strategy_roadmap.txt`**: Internal AI governance, security boundaries, and model deployment standards.

To add your own documents, simply drop `.pdf`, `.txt`, or `.md` files into `data/knowledge_base/` and run `python main.py --reindex` or click **"Re-index Knowledge Base Now"** in the Web UI.

---

## 🧩 Project Structure

```
agentic-rag-realtime/
├── app.py                     # Streamlit interactive Web Application
├── main.py                    # Entry point — CLI pipeline and Web UI launcher
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable configuration template
├── data/
│   └── knowledge_base/        # Source documents (.txt, .pdf, .md)
│       ├── financial_forecasts_q3.txt
│       ├── corporate_event_weather_policy.txt
│       └── enterprise_ai_strategy_roadmap.txt
└── src/
    ├── __init__.py
    ├── agent.py               # Tool-calling agent & interactive simulation engine
    ├── knowledge_indexer.py   # FAISS vector store indexing & semantic retrieval
    ├── tool_registry.py       # Assembles and registers all agent tools
    ├── response_formatter.py  # Rich terminal & markdown response formatter
    └── tools/
        ├── __init__.py
        ├── rag_tool.py        # FAISS knowledge base retrieval tool
        ├── finance_tool.py    # Live Yahoo Finance market metrics (yfinance)
        ├── weather_tool.py    # OpenWeatherMap current weather & dual units
        ├── web_search_tool.py # Tavily live web search API tool
        └── wiki_tool.py       # Wikipedia encyclopaedic lookup tool
```

---

## 🛡️ License

This project is licensed under the MIT License.
