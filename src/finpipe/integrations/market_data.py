import json
import logging
import re
from enum import Enum

import pandas as pd
import yfinance as yf
from mcp.server import MCPServer

logger = logging.getLogger(__name__)


# Define an enum for the type of financial statement
class FinancialType(str, Enum):
    income_stmt = "income_stmt"
    quarterly_income_stmt = "quarterly_income_stmt"
    balance_sheet = "balance_sheet"
    quarterly_balance_sheet = "quarterly_balance_sheet"
    cashflow = "cashflow"
    quarterly_cashflow = "quarterly_cashflow"


class HolderType(str, Enum):
    major_holders = "major_holders"
    institutional_holders = "institutional_holders"
    mutualfund_holders = "mutualfund_holders"
    insider_transactions = "insider_transactions"
    insider_purchases = "insider_purchases"
    insider_roster_holders = "insider_roster_holders"


class RecommendationType(str, Enum):
    recommendations = "recommendations"
    upgrades_downgrades = "upgrades_downgrades"


# --- Ticker normalization ------------------------------------------------
# Yahoo Finance represents US class shares with a hyphen (e.g. "BRK-B"), but
# users and LLMs routinely type them with a dot or slash ("BRK.B", "BRK/B").
# For the dotted form yfinance raises nothing and returns *empty* price data,
# so the bad request fails silently. Normalize the known single-letter US
# share-class suffixes to the hyphen form.
#
# Only a single trailing "A"/"B" class letter is converted. Exchange suffixes
# such as .TO, .L, .HK, .T, .AX are legitimate yfinance tickers and are left
# untouched (e.g. "SHOP.TO", "RIO.L", "7203.T" are returned unchanged).
_CLASS_SHARE_SUFFIXES = {"A", "B"}


def normalize_ticker(ticker: str) -> str:
    if not ticker:
        return ticker
    match = re.fullmatch(r"\s*([A-Za-z]{1,6})[./-]([A-Za-z])\s*", ticker)
    if match is None or match[2].upper() not in _CLASS_SHARE_SUFFIXES:
        return ticker
    return "-".join(part.upper() for part in match.groups())


# Initialize MCP server
yfinance_mcp = MCPServer(
    "yfinance",
    instructions="""
# Yahoo Finance MCP Server

This server is used to get information about a given ticker symbol from yahoo finance.

Available tools:
- get_historical_stock_prices: Get historical stock prices for a given ticker symbol from yahoo finance. Include the following information: Date, Open, High, Low, Close, Volume, Adj Close.
- get_stock_info: Get stock information for a given ticker symbol from yahoo finance. Include the following information: Stock Price & Trading Info, Company Information, Financial Metrics, Earnings & Revenue, Margins & Returns, Dividends, Balance Sheet, Ownership, Analyst Coverage, Risk Metrics, Other.
- get_yahoo_finance_news: Get news for a given ticker symbol from yahoo finance.
- get_stock_actions: Get stock dividends and stock splits for a given ticker symbol from yahoo finance.
- get_financial_statement: Get financial statement for a given ticker symbol from yahoo finance. You can choose from the following financial statement types: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow.
- get_holder_info: Get holder information for a given ticker symbol from yahoo finance. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.
- get_option_expiration_dates: Fetch the available options expiration dates for a given ticker symbol.
- get_option_chain: Fetch the option chain for a given ticker symbol, expiration date, and option type.
- get_recommendations: Get recommendations or upgrades/downgrades for a given ticker symbol from yahoo finance. You can also specify the number of months back to get upgrades/downgrades for, default is 12.
""",
)


@yfinance_mcp.tool(
    name="get_historical_stock_prices",
    description="""Get historical stock prices for a given ticker symbol from yahoo finance. Include the following information: Date, Open, High, Low, Close, Volume, Adj Close.
Args:
    ticker: str
        The ticker symbol of the stock to get historical prices for, e.g. "AAPL"
    period : str
        Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        Either Use period parameter or use start and end
        Default is "1mo"
    interval : str
        Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        Intraday data cannot extend last 60 days
        Default is "1d"
""",
)
async def get_historical_stock_prices(
    ticker: str, period: str = "1mo", interval: str = "1d"
) -> str:
    """Get historical stock prices for a given ticker symbol

    Args:
        ticker: str
            The ticker symbol of the stock to get historical prices for, e.g. "AAPL"
        period : str
            Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            Either Use period parameter or use start and end
            Default is "1mo"
        interval : str
            Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            Intraday data cannot extend last 60 days
            Default is "1d"
    """
    return _request(
        ticker,
        "historical stock prices",
        lambda company: _records(
            company.history(period=period, interval=interval),
            index="Date",
            iso=True,
        ),
    )


@yfinance_mcp.tool(
    name="get_stock_info",
    description="""Get stock information for a given ticker symbol from yahoo finance. Include the following information:
Stock Price & Trading Info, Company Information, Financial Metrics, Earnings & Revenue, Margins & Returns, Dividends, Balance Sheet, Ownership, Analyst Coverage, Risk Metrics, Other.

Args:
    ticker: str
        The ticker symbol of the stock to get information for, e.g. "AAPL"
""",
)
async def get_stock_info(ticker: str) -> str:
    """Get stock information for a given ticker symbol"""

    def read(company):
        info = company.info
        if not info or not {"symbol", "quoteType"}.issubset(info):
            return _report(f"No stock info found for ticker {ticker}.")
        return json.dumps(info)

    return _request(ticker, "stock information", read)


@yfinance_mcp.tool(
    name="get_yahoo_finance_news",
    description="""Get news for a given ticker symbol from yahoo finance.

Args:
    ticker: str
        The ticker symbol of the stock to get news for, e.g. "AAPL"
""",
)
async def get_yahoo_finance_news(ticker: str) -> str:
    """Get news for a given ticker symbol

    Args:
        ticker: str
            The ticker symbol of the stock to get news for, e.g. "AAPL"
    """

    def read(company):
        news = company.news
        if not news:
            return _report(f"No news found for ticker {ticker}.")
        stories = [
            _news_story(item["content"])
            for item in news
            if item.get("content", {}).get("contentType", "") == "STORY"
        ]
        return (
            "\n\n".join(stories)
            if stories
            else _report(
                f"No news found for company that searched with {ticker} ticker."
            )
        )

    return _request(ticker, "news", read)


@yfinance_mcp.tool(
    name="get_stock_actions",
    description="""Get stock dividends and stock splits for a given ticker symbol from yahoo finance.

Args:
    ticker: str
        The ticker symbol of the stock to get stock actions for, e.g. "AAPL"
""",
)
async def get_stock_actions(ticker: str) -> str:
    """Get stock dividends and stock splits for a given ticker symbol"""
    return _request(
        ticker,
        "stock actions",
        lambda company: _records(
            company.actions,
            index="Date",
            iso=True,
        ),
    )


@yfinance_mcp.tool(
    name="get_financial_statement",
    description="""Get financial statement for a given ticker symbol from yahoo finance. You can choose from the following financial statement types: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow.

Args:
    ticker: str
        The ticker symbol of the stock to get financial statement for, e.g. "AAPL"
    financial_type: str
        The type of financial statement to get. You can choose from the following financial statement types: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow.
""",
)
async def get_financial_statement(ticker: str, financial_type: str) -> str:
    """Get financial statement for a given ticker symbol"""

    def read(company):
        if financial_type not in [kind.value for kind in FinancialType]:
            return _invalid_type("financial", financial_type, FinancialType)
        frame = getattr(company, FinancialType(financial_type).value)
        if frame.empty:
            return _report(f"No financial statement data found for ticker {ticker}.")
        return json.dumps(_statement_rows(frame))

    return _request(ticker, "financial statement", read)


@yfinance_mcp.tool(
    name="get_holder_info",
    description="""Get holder information for a given ticker symbol from yahoo finance. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.

Args:
    ticker: str
        The ticker symbol of the stock to get holder information for, e.g. "AAPL"
    holder_type: str
        The type of holder information to get. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.
""",
)
async def get_holder_info(ticker: str, holder_type: str) -> str:
    """Get holder information for a given ticker symbol"""

    def read(company):
        if holder_type not in [kind.value for kind in HolderType]:
            return _invalid_type("holder", holder_type, HolderType)
        kind = HolderType(holder_type)
        frame = getattr(company, kind.value)
        return _records(
            frame,
            index="metric" if kind == HolderType.major_holders else None,
            iso=kind
            not in (HolderType.major_holders, HolderType.institutional_holders),
        )

    return _request(ticker, "holder info", read)


@yfinance_mcp.tool(
    name="get_option_expiration_dates",
    description="""Fetch the available options expiration dates for a given ticker symbol.

Args:
    ticker: str
        The ticker symbol of the stock to get option expiration dates for, e.g. "AAPL"
""",
)
async def get_option_expiration_dates(ticker: str) -> str:
    """Fetch the available options expiration dates for a given ticker symbol."""

    def read(company):
        dates = company.options
        return (
            json.dumps(dates)
            if dates
            else _report(f"No options expiration dates found for ticker {ticker}.")
        )

    return _request(ticker, "option expiration dates", read)


def _spot_price(company: yf.Ticker) -> float | None:
    def from_history():
        history = company.history(period="1d", interval="1d")
        if history is None or history.empty or "Close" not in history:
            return None
        closes = history["Close"].dropna()
        return closes.iloc[-1] if not closes.empty else None

    for read in (lambda: company.fast_info.get("lastPrice"), from_history):
        try:
            value = read()
            if value is not None and float(value) > 0:
                return float(value)
        except Exception:
            logger.debug("Spot-price source unavailable", exc_info=True)
    return None


def _window_strikes(
    company: yf.Ticker, chain: pd.DataFrame, strike_window_pct: float | None
) -> pd.DataFrame:
    if strike_window_pct is None or "strike" not in chain:
        return chain
    try:
        window = float(strike_window_pct)
    except (TypeError, ValueError):
        return chain
    if window <= 0:
        return chain
    spot = _spot_price(company)
    if spot is None:
        return chain
    selected = chain.loc[
        chain["strike"].between(spot * (1 - window), spot * (1 + window))
    ]
    return chain if selected.empty else selected


def _project_fields(chain: pd.DataFrame, fields: list[str] | None) -> pd.DataFrame:
    requested = set(fields or ())
    if not requested or not requested.issubset(chain.columns):
        return chain
    selected = [column for column in chain.columns if column in requested]
    if "strike" in chain and "strike" not in selected:
        selected.insert(0, "strike")
    return chain.loc[:, selected]


@yfinance_mcp.tool(
    name="get_option_chain",
    description="""Fetch the option chain for a given ticker symbol, expiration date, and option type.

A full chain is large (a liquid US name runs 40-90 strikes and ~17KB of JSON per
expiration). Prefer `strike_window_pct` and `fields` to request only the strikes
and columns you actually need — pulling several full chains into one analysis is
the main driver of oversized requests.

Args:
    ticker: str
        The ticker symbol of the stock to get option chain for, e.g. "AAPL"
    expiration_date: str
        The expiration date for the options chain (format: 'YYYY-MM-DD')
    option_type: str
        The type of option to fetch ('calls' or 'puts')
    strike_window_pct: float | None
        Keep only strikes within +/- this fraction of the current spot price
        (e.g. 0.15 keeps strikes from 85% to 115% of spot). Omit for every strike.
    fields: list[str] | None
        Only return these columns. `strike` is always included. Omit for every column.
        Names must match these columns EXACTLY (they are case-sensitive, and none
        contain spaces or underscores):
            contractSymbol, lastTradeDate, strike, lastPrice, bid, ask, change,
            percentChange, volume, openInterest, impliedVolatility, inTheMoney,
            contractSize, currency
        Note "lastPrice" (not "last"), "impliedVolatility" (not "implied volatility"),
        "openInterest" (not "open interest"). If ANY name is not in that list the
        projection is dropped and the full chain is returned, so a typo costs
        payload rather than silently omitting a column you asked for.
        A good pricing set: ["strike", "bid", "ask", "lastPrice",
        "impliedVolatility", "openInterest", "volume"].
""",
)
async def get_option_chain(
    ticker: str,
    expiration_date: str,
    option_type: str,
    strike_window_pct: float | None = None,
    fields: list[str] | None = None,
) -> str:
    """Fetch the option chain for a given ticker symbol, expiration date, and option type.

    Args:
        ticker: The ticker symbol of the stock
        expiration_date: The expiration date for the options chain (format: 'YYYY-MM-DD')
        option_type: The type of option to fetch ('calls' or 'puts')
        strike_window_pct: Keep only strikes within +/- this fraction of spot.
            None returns every strike (the historical behavior).
        fields: Only return these columns, matched exactly against the chain's
            own column names. None returns every column (the historical
            behavior). `strike` is always retained so rows stay identifiable.
            If any name is unrecognized the projection is dropped entirely —
            see :func:`_project_fields`.

    Returns:
        str: JSON string containing the option chain data
    """

    def read(company):
        if expiration_date not in company.options:
            return f"Error: No options available for the date {expiration_date}. You can use `get_option_expiration_dates` to get the available expiration dates."
        if option_type not in ("calls", "puts"):
            return "Error: Invalid option type. Please use 'calls' or 'puts'."
        options = company.option_chain(expiration_date)
        chain = getattr(options, option_type)
        return _records(
            _project_fields(_window_strikes(company, chain, strike_window_pct), fields),
            iso=True,
        )

    return _request(ticker, "option chain", read)


@yfinance_mcp.tool(
    name="get_recommendations",
    description="""Get recommendations or upgrades/downgrades for a given ticker symbol from yahoo finance. You can also specify the number of months back to get upgrades/downgrades for, default is 12.

Args:
    ticker: str
        The ticker symbol of the stock to get recommendations for, e.g. "AAPL"
    recommendation_type: str
        The type of recommendation to get. You can choose from the following recommendation types: recommendations, upgrades_downgrades.
    months_back: int
        The number of months back to get upgrades/downgrades for, default is 12.
""",
)
async def get_recommendations(
    ticker: str, recommendation_type: str, months_back: int = 12
) -> str:
    """Get recommendations or upgrades/downgrades for a given ticker symbol"""

    def read(company):
        if recommendation_type not in [kind.value for kind in RecommendationType]:
            return _invalid_type(
                "recommendation", recommendation_type, RecommendationType
            )
        kind = RecommendationType(recommendation_type)
        frame = getattr(company, kind.value)
        if frame.empty:
            return "[]"
        if kind == RecommendationType.recommendations:
            return _records(frame)
        grades = frame.reset_index()
        cutoff = pd.Timestamp.now() - pd.DateOffset(months=months_back)
        latest = (
            grades.loc[grades["GradeDate"] >= cutoff]
            .sort_values(
                "GradeDate",
                ascending=False,
            )
            .drop_duplicates(subset=["Firm"])
        )
        return _records(latest, iso=True)

    return _request(ticker, "recommendations", read)


def _report(message: str) -> str:
    print(message)
    return message


def _request(ticker: str, action: str, read) -> str:
    """Convert provider errors to the existing tool response format."""
    try:
        return read(yf.Ticker(normalize_ticker(ticker)))
    except Exception as error:
        logger.debug("Market data request failed", exc_info=True)
        return _report(f"Error: getting {action} for {ticker}: {error}")


def _records(
    frame: pd.DataFrame, *, index: str | None = None, iso: bool = False
) -> str:
    if index is not None:
        frame = frame.reset_index(names=index)
    options = {"orient": "records"}
    if iso:
        options["date_format"] = "iso"
    return frame.to_json(**options)


def _invalid_type(label: str, value: str, kinds: type[Enum]) -> str:
    options = ", ".join(str(kind) for kind in kinds)
    return f"Error: invalid {label} type {value}. Please use one of the following: {options}."


def _news_story(content: dict) -> str:
    fields = [
        ("Title", content.get("title", "")),
        ("Summary", content.get("summary", "")),
        ("Description", content.get("description", "")),
        ("URL", content.get("canonicalUrl", {}).get("url", "")),
    ]
    return "\n".join(f"{label}: {value}" for label, value in fields)


def _statement_rows(frame: pd.DataFrame) -> list[dict]:
    return [
        {
            "date": column.strftime("%Y-%m-%d")
            if isinstance(column, pd.Timestamp)
            else str(column),
            **{
                metric: None if pd.isna(value) else value
                for metric, value in frame[column].items()
            },
        }
        for column in frame.columns
    ]
