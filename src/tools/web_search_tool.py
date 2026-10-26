"""
src/tools/web_search_tool.py

Provides live web search capability via the Tavily API so the agent can answer
questions about current events, breaking news, or anything not in the static
knowledge base.
"""

try:
    from langchain_core.tools import Tool
except ImportError:
    from langchain.tools import Tool  # type: ignore


def create_web_search_tool(tavily_api_key: str) -> Tool:
    """
    Build a LangChain Tool that queries the Tavily web search API.
    """

    def _web_search(query: str) -> str:
        """
        Call the Tavily search API and format the top results as structured text.
        """
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=tavily_api_key)
            response = client.search(query, max_results=3)

            results = response.get("results", [])
            if not results:
                return "Web search returned no results for that query."

            lines = []
            for r in results:
                title = r.get("title", "No title")
                snippet = r.get("content", r.get("snippet", "No snippet available"))
                url = r.get("url", "")
                lines.append(f"Title: {title}\nSnippet: {snippet}\nURL: {url}\n---")

            return "\n".join(lines)

        except ImportError:
            return _tavily_http_fallback(query, tavily_api_key)

        except Exception as exc:
            return f"Web search unavailable: {exc}"

    def _tavily_http_fallback(query: str, api_key: str) -> str:
        """Direct HTTP fallback when tavily-python is missing."""
        import requests

        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key, "query": query, "max_results": 3},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return "Web search returned no results."
            lines = []
            for r in results:
                lines.append(
                    f"Title: {r.get('title', '')}\n"
                    f"Snippet: {r.get('content', '')}\n"
                    f"URL: {r.get('url', '')}\n---"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"Web search unavailable: {exc}"

    return Tool(
        name="web_search",
        func=_web_search,
        description=(
            "Search the live web for current information, breaking news, or recent developments. "
            "Use this when the query requires real-time facts not covered by internal documents. "
            "Input: a search query string."
        ),
    )


def create_mock_web_search_tool() -> Tool:
    """
    Fallback tool returned when no Tavily API key is configured.
    """

    def _mock_search(query: str) -> str:
        return (
            "Web search is not configured. "
            "Please configure TAVILY_API_KEY in .env or the UI sidebar to enable live web search."
        )

    return Tool(
        name="web_search",
        func=_mock_search,
        description=(
            "Search the live web for current information, news, or recent events. "
            "NOTE: Web search is currently in mock mode (no API key configured). "
            "Input: a search query string."
        ),
    )
