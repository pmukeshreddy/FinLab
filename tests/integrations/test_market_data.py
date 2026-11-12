import asyncio
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from finpipe.integrations import market_data as market


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("BRK.B", "BRK-B"),
        ("brk/b", "BRK-B"),
        (" BF-A ", "BF-A"),
        ("SHOP.TO", "SHOP.TO"),
        ("RIO.L", "RIO.L"),
        ("7203.T", "7203.T"),
        ("", ""),
    ],
)
def test_ticker_normalization_preserves_exchange_suffixes(symbol, expected):
    assert market.normalize_ticker(symbol) == expected


def test_price_history_normalizes_symbol_and_serializes_dates(monkeypatch):
    calls = []

    def history(**kwargs):
        assert kwargs == {"period": "1mo", "interval": "1d"}
        return pd.DataFrame({"Close": [101.5]}, index=pd.to_datetime(["2026-01-02"]))

    def ticker(symbol):
        calls.append(symbol)
        return SimpleNamespace(history=history)

    monkeypatch.setattr(market.yf, "Ticker", ticker)
    result = asyncio.run(market.get_historical_stock_prices("BRK.B"))
    assert json.loads(result) == [{"Date": "2026-01-02T00:00:00.000", "Close": 101.5}]
    assert calls == ["BRK-B"]


def test_financial_statements_preserve_dates_and_null_values(monkeypatch):
    company = SimpleNamespace(
        income_stmt=pd.DataFrame(
            {pd.Timestamp("2025-12-31"): [12.0, float("nan")]},
            index=["Revenue", "Profit"],
        )
    )
    monkeypatch.setattr(market.yf, "Ticker", lambda symbol: company)
    result = asyncio.run(market.get_financial_statement("EXAMPLE", "income_stmt"))
    assert json.loads(result) == [
        {"date": "2025-12-31", "Revenue": 12.0, "Profit": None}
    ]


def test_news_filters_non_stories_and_preserves_fields(monkeypatch):
    company = SimpleNamespace(
        news=[
            {"content": {"contentType": "VIDEO", "title": "Ignored"}},
            {
                "content": {
                    "contentType": "STORY",
                    "title": "Headline",
                    "summary": "Summary",
                    "description": "Details",
                    "canonicalUrl": {"url": "https://example.com/story"},
                }
            },
        ]
    )
    monkeypatch.setattr(market.yf, "Ticker", lambda symbol: company)
    assert asyncio.run(market.get_yahoo_finance_news("EXAMPLE")) == (
        "Title: Headline\nSummary: Summary\nDescription: Details\nURL: https://example.com/story"
    )


@pytest.mark.parametrize(
    "function,action",
    [
        (market.get_stock_info, "stock information"),
        (market.get_stock_actions, "stock actions"),
        (market.get_option_expiration_dates, "option expiration dates"),
    ],
)
def test_provider_errors_remain_tool_results(monkeypatch, function, action):
    def unavailable(symbol):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(market.yf, "Ticker", unavailable)
    assert (
        asyncio.run(function("EXAMPLE"))
        == f"Error: getting {action} for EXAMPLE: provider unavailable"
    )


def test_option_window_and_projection_preserve_contract(monkeypatch):
    chain = pd.DataFrame(
        {"strike": [80, 100, 120], "bid": [1.0, 2.0, 3.0], "ask": [2.0, 3.0, 4.0]}
    )
    company = SimpleNamespace(
        options=["2030-01-18"],
        fast_info={"lastPrice": 100},
        option_chain=lambda date: SimpleNamespace(calls=chain, puts=chain),
    )
    monkeypatch.setattr(market.yf, "Ticker", lambda symbol: company)
    result = asyncio.run(
        market.get_option_chain("EXAMPLE", "2030-01-18", "calls", 0.1, ["bid"])
    )
    assert json.loads(result) == [{"strike": 100, "bid": 2.0}]
    assert market._window_strikes(company, chain, 0.001).shape[0] == 1
    assert list(market._project_fields(chain, ["bid", "not-a-field"]).columns) == [
        "strike",
        "bid",
        "ask",
    ]
    assert market._window_strikes(
        SimpleNamespace(fast_info={"lastPrice": 500}), chain, 0.01
    ).equals(chain)
    assert (
        asyncio.run(market.get_option_chain("EXAMPLE", "2030-01-18", "invalid"))
        == "Error: Invalid option type. Please use 'calls' or 'puts'."
    )


def test_spot_price_falls_back_to_latest_valid_close():
    company = SimpleNamespace(
        fast_info={"lastPrice": None},
        history=lambda **kwargs: pd.DataFrame({"Close": [90.0, 100.0, float("nan")]}),
    )
    assert market._spot_price(company) == 100.0


def test_recommendations_keep_latest_grade_per_firm(monkeypatch):
    grades = pd.DataFrame(
        {"Firm": ["A", "A", "B"], "ToGrade": ["Hold", "Buy", "Sell"]},
        index=pd.DatetimeIndex(
            ["2030-01-01", "2030-02-01", "2030-01-15"], name="GradeDate"
        ),
    )
    monkeypatch.setattr(
        market.yf, "Ticker", lambda symbol: SimpleNamespace(upgrades_downgrades=grades)
    )
    result = asyncio.run(market.get_recommendations("EXAMPLE", "upgrades_downgrades"))
    assert json.loads(result) == [
        {"GradeDate": "2030-02-01T00:00:00.000", "Firm": "A", "ToGrade": "Buy"},
        {"GradeDate": "2030-01-15T00:00:00.000", "Firm": "B", "ToGrade": "Sell"},
    ]
