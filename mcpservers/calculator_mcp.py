from __future__ import annotations

import ast
import math
import operator

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calculator-mcp")

_MAX_LEN = 256
_MAX_DEPTH = 16
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calculate(expression: str) -> float:
    if not isinstance(expression, str):
        raise TypeError("expression must be str")
    expression = expression.strip()
    if not expression:
        raise ValueError("invalid expression")
    if len(expression) > _MAX_LEN:
        raise ValueError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid expression") from exc
    value = _eval_node(tree.body, 0)
    if not math.isfinite(value):
        raise ValueError("non-finite result")
    return float(value)


def _eval_node(node: ast.AST, depth: int) -> float:
    if depth > _MAX_DEPTH:
        raise ValueError("expression too deep")
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("unsupported expression")
        value = float(node.value)
        if not math.isfinite(value):
            raise ValueError("non-finite number")
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return float(_UNARYOPS[type(node.op)](_eval_node(node.operand, depth + 1)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)
        if isinstance(node.op, ast.Div) and right == 0.0:
            raise ValueError("division by zero")
        result = float(_BINOPS[type(node.op)](left, right))
        if not math.isfinite(result):
            raise ValueError("non-finite result")
        return result
    raise ValueError("unsupported expression")


@mcp.tool(name="calculate")
def calculate_tool(expression: str) -> float:
    """Evaluate arithmetic with + - * / parentheses and unary signs."""
    return calculate(expression)


if __name__ == "__main__":
    mcp.run()
