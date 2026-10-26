from typing import List
try:
    from langchain_core.tools import Tool
except ImportError:
    from langchain.tools import Tool  # type: ignore

from langchain_community.vectorstores import FAISS

from src.tools.rag_tool import create_rag_tool
from src.tools.web_search_tool import create_web_search_tool, create_mock_web_search_tool
from src.tools.finance_tool import create_finance_tool
from src.tools.weather_tool import create_weather_tool, create_mock_weather_tool
from src.tools.wiki_tool import create_wiki_tool
def build_tool_registry(
    vector_store: FAISS,
    config: dict,
) -> List[Tool]:
    tools: List[Tool] = []
    domain = config.get("domain_description", "internal company documents, valuation models, and operational policies")
    rag = create_rag_tool(vector_store, domain_description=domain)
    tools.append(rag)

    # --- 2. Finance Tool ---
    finance = create_finance_tool()
    tools.append(finance)

    # --- 3. Wikipedia Tool ---
    wiki = create_wiki_tool()
    tools.append(wiki)

    # --- 4. Web Search Tool ---
    tavily_key = config.get("tavily_api_key")
    if tavily_key:
        web = create_web_search_tool(tavily_key)
    else:
        web = create_mock_web_search_tool()
    tools.append(web)

    # --- 5. Weather Tool ---
    owm_key = config.get("openweathermap_api_key")
    if owm_key:
        weather = create_weather_tool(owm_key)
    else:
        weather = create_mock_weather_tool()
    tools.append(weather)

    return tools
def get_tool_descriptions(tools: List[Tool]) -> str:
    icon_map = {
        "get_stock_data": "📈",
        "search_knowledge_base": "📚",
        "get_weather": "🌦️",
        "search_wikipedia": "📖",
        "web_search": "🌐",
    }
    lines = ["Registered Agent Tools:"]
    for tool in tools:
        icon = icon_map.get(tool.name, "🔧")
        first_sentence = tool.description.split(".")[0] + "."
        lines.append(f"  {icon} {tool.name:22} : {first_sentence}")
    return "\n".join(lines)
