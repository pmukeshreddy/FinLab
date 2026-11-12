import logging
import os

from alpaca_mcp_server.server import build_server
from fastmcp import Client
from fastmcp.client import BearerAuth
from fastmcp.client.group import ClientGroup
from langchain.mcp import MCPAdapter
from langchain_core.tools import BaseTool

from finpipe.integrations.market_data import yfinance_mcp

ALPACA_TOOLSETS = "assets,stock-data,news,crypto-data,options-data,fixed-income-data"

KAGGLE_MCP_URL = "https://www.kaggle.com/mcp/"
CRYPTO_COM_MCP_URL = "https://mcp.crypto.com/market-data/mcp/"
COINDESK_MCP_URL = "https://mcp.coindesk.com/mcp/"
GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/readonly/"

WHITELIST_TOOLS = {
    "kaggle_get_competition_data_files_summary",
    "kaggle_download_dataset",
    "kaggle_search_content",
    "kaggle_get_dataset_files_summary",
    "kaggle_get_dataset_status",
    "kaggle_get_dataset_info",
    "kaggle_list_competition_data_files",
    "kaggle_get_dataset_metadata",
    "kaggle_list_dataset_files",
    "kaggle_list_competition_data_tree_files",
    "kaggle_list_competition_topics",
    "kaggle_download_competition_data_files",
    "kaggle_search_competitions",
    "kaggle_search_datasets",
    "kaggle_list_dataset_tree_files",
    "yahoo_get_stock_info",
    "yahoo_get_yahoo_finance_news",
    "crypto_com_get_ticker",
    "crypto_com_get_instrument",
    "crypto_com_get_tickers",
    "crypto_com_get_instruments",
    "github_get_file_contents",
    "github_search_code",
    "github_search_repositories",
    "alpaca_search_alpaca_docs",
    "alpaca_fetch_alpaca_doc",
    "alpaca_search_alpaca_api_specs",
    "alpaca_list_alpaca_api_endpoints",
    "alpaca_get_alpaca_endpoint_docs",
    "alpaca_get_all_assets",
    "alpaca_get_asset",
    "alpaca_get_most_active_stocks",
}

alpaca_mcp_server = None

logger = logging.getLogger(__name__)


def _alpaca_server():
    global alpaca_mcp_server
    if alpaca_mcp_server is None:
        os.environ["ALPACA_TOOLSETS"] = ALPACA_TOOLSETS
        alpaca_mcp_server = build_server()
    return alpaca_mcp_server


def _has_credentials(*names: str) -> bool:
    return all(name in os.environ for name in names)


def init_mcp_servers() -> ClientGroup:
    servers = {
        name: Client(target)
        for name, target in (
            ("kaggle", KAGGLE_MCP_URL),
            ("yahoo", yfinance_mcp),
            ("crypto_com", CRYPTO_COM_MCP_URL),
        )
    }

    if _has_credentials("GITHUB_PAT"):
        servers["github"] = Client(
            GITHUB_MCP_URL.rstrip("/"),
            auth=BearerAuth(os.getenv("GITHUB_PAT")),
        )
    else:
        logger.info("GITHUB_PAT not provided, not using Github MCP tool")

    if _has_credentials("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        servers["alpaca"] = Client(_alpaca_server())
    else:
        logger.info("ALPACA_API_KEY or ALPACA_SECRET_KEY not set, not using Alpaca mcp")

    return ClientGroup(servers)


async def get_tools(mcp_adapter: MCPAdapter) -> list[BaseTool]:
    available = await mcp_adapter.list_tools()
    return [tool for tool in available if tool.name in WHITELIST_TOOLS]
