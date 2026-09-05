from __future__ import annotations

import unittest

from mcpservers.calculator_mcp import calculate


class CalculatorTests(unittest.TestCase):
    def test_add_sub_mul_div(self) -> None:
        self.assertEqual(calculate("1+2"), 3.0)
        self.assertEqual(calculate("5-2"), 3.0)
        self.assertEqual(calculate("3*4"), 12.0)
        self.assertEqual(calculate("8/2"), 4.0)

    def test_parentheses_and_unary(self) -> None:
        self.assertEqual(calculate("(1+2)*3"), 9.0)
        self.assertEqual(calculate("-(2+3)"), -5.0)
        self.assertEqual(calculate("+-4"), -4.0)
        self.assertEqual(calculate("1-(2-(3))"), 2.0)

    def test_whitespace_and_floats(self) -> None:
        self.assertEqual(calculate("  1.5 + 2.5 "), 4.0)
        self.assertEqual(calculate(".5*2"), 1.0)

    def test_zero_and_falsy_numeric(self) -> None:
        self.assertEqual(calculate("0"), 0.0)
        self.assertEqual(calculate("1-1"), 0.0)

    def test_rejects_names_calls_powers(self) -> None:
        with self.assertRaises(ValueError):
            calculate("os")
        with self.assertRaises(ValueError):
            calculate("__import__('os')")
        with self.assertRaises(ValueError):
            calculate("2**3")
        with self.assertRaises(ValueError):
            calculate("pow(2,3)")

    def test_rejects_non_arithmetic(self) -> None:
        with self.assertRaises(ValueError):
            calculate("1 if 1 else 0")
        with self.assertRaises(ValueError):
            calculate("[1]")
        with self.assertRaises(ValueError):
            calculate("1;2")
        with self.assertRaises(ValueError):
            calculate("")

    def test_bounds_length_depth_finite(self) -> None:
        with self.assertRaises(ValueError):
            calculate("1+" * 200 + "1")
        nested = "1"
        for _ in range(20):
            nested = f"({nested}+1)"
        with self.assertRaises(ValueError):
            calculate(nested)
        with self.assertRaises(ValueError):
            calculate("1e1000")
        with self.assertRaises(ValueError):
            calculate("1/0")

    def test_type_and_no_eval(self) -> None:
        with self.assertRaises(TypeError):
            calculate(1)  # type: ignore[arg-type]
        self.assertEqual(calculate("10/4"), 2.5)
