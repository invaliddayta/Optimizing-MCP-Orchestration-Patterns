from __future__ import annotations
from typing import Dict, Any, Optional
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("inventory-mcp")

INVENTORY: Dict[str, Dict[str, Any]] = {
    "chip_module": {
        "stock_units": 120,
        "base_price_eur": 500.0,
        "weight_per_unit_lb": 1.2,
        "tsla_shares_per_unit": 0.05,
    },
    "battery_module": {
        "stock_units": 50,
        "base_price_eur": 800.0,
        "weight_per_unit_lb": 5.0,
        "tsla_shares_per_unit": 0.0,
    },
}

@mcp.tool()
def get_inventory(product: Optional[str] = None) -> Any:
    # returns dict for specific product or full inventory
    if product:
        payload = INVENTORY.get(product)
        return {product: payload} if payload else {}

    return INVENTORY

if __name__ == "__main__":
    mcp.run()
