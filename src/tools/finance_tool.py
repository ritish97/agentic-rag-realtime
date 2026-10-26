"""
src/tools/finance_tool.py

Fetches live stock and financial metrics via yfinance (Yahoo Finance).
Provides real-time pricing, 52-week high/low ranges, P/E ratios, and market caps.
"""

try:
    from langchain_core.tools import Tool
except ImportError:
    from langchain.tools import Tool  # type: ignore


# Common company-name → ticker fallback mapping for high-frequency queries.
_NAME_TO_TICKER = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "berkshire": "BRK-B",
    "visa": "V",
    "jpmorgan": "JPM",
    "walmart": "WMT",
    "intel": "INTC",
    "amd": "AMD",
    "oracle": "ORCL",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "qualcomm": "QCOM",
    "broadcom": "AVGO",
    "uber": "UBER",
    "spotify": "SPOT",
}


def _resolve_ticker(input_str: str) -> str:
    """
    Best-effort conversion of user/agent input to a valid ticker symbol.
    """
    stripped = input_str.strip()
    clean_symbol = stripped.rstrip("?.,!").strip()

    # If it already looks like a ticker (1-6 uppercase letters/symbols)
    if len(clean_symbol) <= 6 and clean_symbol.replace("-", "").replace(".", "").isalpha():
        return clean_symbol.upper()

    # Check local dictionary
    lower = clean_symbol.lower()
    for name, ticker in _NAME_TO_TICKER.items():
        if name in lower:
            return ticker

    return clean_symbol.upper()


def create_finance_tool() -> Tool:
    """
    Build a LangChain Tool that returns live stock metrics from Yahoo Finance.
    """

    def get_stock_data(input_str: str) -> str:
        """
        Fetch key financial metrics for a given ticker or company name.
        """
        try:
            import yfinance as yf

            ticker_symbol = _resolve_ticker(input_str)
            ticker = yf.Ticker(ticker_symbol)

            # Retrieve fast_info first
            fast = getattr(ticker, "fast_info", None)
            price = getattr(fast, "last_price", None) if fast else None

            # Fallback 1: check ticker.info
            info = {}
            try:
                info = ticker.info or {}
            except Exception:
                pass

            if price is None:
                price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("ask")

            # Fallback 2: fetch most recent day from history
            if price is None:
                try:
                    hist = ticker.history(period="5d")
                    if not hist.empty:
                        price = float(hist["Close"].iloc[-1])
                except Exception:
                    pass

            if price is None:
                return (
                    f"Could not find stock data for '{input_str}' (resolved ticker: {ticker_symbol}). "
                    "Please provide a valid ticker symbol (e.g. AAPL, MSFT, NVDA)."
                )

            high_52w = getattr(fast, "year_high", None) or info.get("fiftyTwoWeekHigh")
            low_52w = getattr(fast, "year_low", None) or info.get("fiftyTwoWeekLow")
            market_cap = getattr(fast, "market_cap", None) or info.get("marketCap")
            pe_ratio = info.get("trailingPE") or info.get("forwardPE")
            currency = getattr(fast, "currency", None) or info.get("currency", "USD")

            def _fmt_cap(cap) -> str:
                if cap is None:
                    return "N/A"
                if cap >= 1e12:
                    return f"${cap / 1e12:.2f}T"
                if cap >= 1e9:
                    return f"${cap / 1e9:.2f}B"
                if cap >= 1e6:
                    return f"${cap / 1e6:.2f}M"
                return f"${cap:,.0f}"

            pe_str = f"{pe_ratio:.1f}" if pe_ratio else "N/A"
            high_str = f"${high_52w:.2f}" if high_52w else "N/A"
            low_str = f"${low_52w:.2f}" if low_52w else "N/A"

            return (
                f"Stock: {ticker_symbol} | "
                f"Price: ${price:.2f} {currency} | "
                f"52W High: {high_str} | "
                f"52W Low: {low_str} | "
                f"P/E: {pe_str} | "
                f"Market Cap: {_fmt_cap(market_cap)}"
            )

        except Exception as exc:
            return (
                f"Could not find stock data for '{input_str}'. "
                f"Error: {exc}. "
                "Please provide a valid ticker symbol (e.g. AAPL, MSFT, NVDA)."
            )

    return Tool(
        name="get_stock_data",
        func=get_stock_data,
        description=(
            "Get current stock and financial metrics for any publicly traded company. "
            "Input: a stock ticker symbol (e.g., AAPL, MSFT, GOOGL) or company name. "
            "Returns live price, 52-week range, P/E ratio, and market capitalization."
        ),
    )
