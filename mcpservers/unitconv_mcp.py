from __future__ import annotations
from typing import Literal
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("unitconv-mcp")

LB_TO_KG = 0.45359237
MS_TO_KMH = 3.6

def _convert_mass(value: float, from_u: str, to_u: str) -> float:
    if from_u == "lb" and to_u == "kg":
        return value * LB_TO_KG
    if from_u == "kg" and to_u == "lb":
        return value / LB_TO_KG
    raise ValueError("Unsupported mass conversion")

def _convert_speed(value: float, from_u: str, to_u: str) -> float:
    if from_u == "km/h" and to_u == "m/s":
        return value / MS_TO_KMH
    if from_u == "m/s" and to_u == "km/h":
        return value * MS_TO_KMH
    raise ValueError("Unsupported speed conversion")

@mcp.tool()
def convert_units(
    value: float,
    from_: Literal["lb", "kg", "km/h", "m/s"],
    to:   Literal["lb", "kg", "km/h", "m/s"],
) -> float:
    # mass
    if (from_ in ("lb", "kg")) and (to in ("lb", "kg")):
        return round(_convert_mass(value, from_, to), 9)

    # speed
    if (from_ in ("km/h", "m/s")) and (to in ("km/h", "m/s")):
        return round(_convert_speed(value, from_, to), 9)

    raise ValueError("Unsupported conversion")

if __name__ == "__main__":
    mcp.run()
