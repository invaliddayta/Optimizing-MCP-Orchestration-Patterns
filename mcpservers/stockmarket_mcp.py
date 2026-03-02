from __future__ import annotations
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("stockmarket-mcp")

STATIC_CLOSE_PRICES: Dict[str, float] = {
    "TSLA": 242.17,
    "AAPL": 189.12,
    "GOOG": 172.33,
    "MSFT": 418.55,
}

@mcp.tool()
def get_price_history(symbol: str, period: str, interval: str) -> Dict[str, Any]:
    ticker = (symbol or "").upper()
    close_price = STATIC_CLOSE_PRICES.get(ticker)

    if close_price is None:
        return {
            "symbol": ticker,
            "period": period,
            "interval": interval,
            "error": "unknown_symbol",
            "message": f"No static stub price for {ticker}",
        }

    return {
        "symbol": ticker,
        "period": period,
        "interval": interval,
        "close": close_price,
    }

if __name__ == "__main__":
    mcp.run()
