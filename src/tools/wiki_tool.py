"""
src/tools/wiki_tool.py

Provides Wikipedia lookups as a lightweight alternative to full web search.
Wikipedia is ideal for factual, encyclopaedic questions where live web crawling
isn't needed but the answer isn't in the internal knowledge base either.
"""

try:
    from langchain_core.tools import Tool
except ImportError:
    from langchain.tools import Tool  # type: ignore


def create_wiki_tool() -> Tool:
    """
    Build a LangChain Tool that searches Wikipedia for encyclopaedic information.
    """

    def search_wikipedia(query: str) -> str:
        """
        Search Wikipedia for the query and return a concise summary with article link.
        """
        try:
            import wikipedia

            query = query.strip()
            page_title = query
            page_url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"

            try:
                page = wikipedia.page(query, auto_suggest=True)
                summary = wikipedia.summary(page.title, sentences=4)
                page_title = page.title
                page_url = page.url
            except wikipedia.exceptions.DisambiguationError as e:
                if e.options:
                    try:
                        page = wikipedia.page(e.options[0])
                        summary = wikipedia.summary(page.title, sentences=4)
                        page_title = page.title
                        page_url = page.url
                    except Exception:
                        summary = wikipedia.summary(e.options[0], sentences=4)
                        page_title = e.options[0]
                else:
                    return f"Wikipedia: '{query}' is ambiguous. Please be more specific."
            except wikipedia.exceptions.PageError:
                return f"Wikipedia: No article found for '{query}'. Try a different search term."

            if len(summary) > 850:
                summary = summary[:850] + "…"

            return f"Wikipedia [{page_title}] ({page_url}):\n{summary}"

        except ImportError:
            return (
                "Wikipedia tool is unavailable. "
                "Install the 'wikipedia' package: pip install wikipedia"
            )
        except Exception as exc:
            return f"Wikipedia lookup failed for '{query}'. Error: {exc}"

    return Tool(
        name="search_wikipedia",
        func=search_wikipedia,
        description=(
            "Search Wikipedia for factual, encyclopaedic information about any concept or topic. "
            "Use this for definitions, historical background, scientific theories, or general knowledge. "
            "No API key required. "
            "Input: a topic or search term string."
        ),
    )
