import asyncio
from types import SimpleNamespace

from finpipe.integrations import tools


def test_tool_filter_keeps_only_allowed_tools_in_provider_order():
    listed = [
        SimpleNamespace(name="github_search_code"),
        SimpleNamespace(name="alpaca_place_order"),
        SimpleNamespace(name="yahoo_get_stock_info"),
    ]

    class Adapter:
        async def list_tools(self):
            return listed

    assert asyncio.run(tools.get_tools(Adapter())) == [listed[0], listed[2]]


def test_optional_clients_follow_environment_credentials(monkeypatch):
    for key in ("GITHUB_PAT", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(tools, "Client", lambda target, **kwargs: (target, kwargs))
    monkeypatch.setattr(tools, "ClientGroup", dict)
    assert list(tools.init_mcp_servers()) == ["kaggle", "yahoo", "crypto_com"]
    monkeypatch.setenv("GITHUB_PAT", "test-token")
    assert list(tools.init_mcp_servers()) == ["kaggle", "yahoo", "crypto_com", "github"]


def test_alpaca_initializes_only_when_both_credentials_exist(monkeypatch):
    for key in ("GITHUB_PAT", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    builds = []
    monkeypatch.setattr(tools, "alpaca_mcp_server", None)
    monkeypatch.setenv("ALPACA_TOOLSETS", "")
    monkeypatch.setattr(
        tools, "build_server", lambda: builds.append("built") or "alpaca-server"
    )
    monkeypatch.setattr(tools, "Client", lambda target, **kwargs: target)
    monkeypatch.setattr(tools, "ClientGroup", dict)
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    assert "alpaca" not in tools.init_mcp_servers()
    assert builds == []
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    assert tools.init_mcp_servers()["alpaca"] == "alpaca-server"
    assert builds == ["built"]
