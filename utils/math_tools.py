"""
utils/math_tools.py - Symbolic math computation tools using SymPy
"""
import sympy as sp
from sympy import symbols, solve, diff, integrate, limit, simplify, expand, factor
from sympy import Matrix, det
from sympy import Rational, sqrt, pi, E, oo
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from typing import Any, Optional


TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def safe_eval_math(expression: str) -> dict:
    """
    Safely evaluate a math expression using SymPy.
    Returns dict with result or error.
    """
    try:
        expr = parse_expr(expression, transformations=TRANSFORMATIONS)
        result = simplify(expr)
        return {
            "success": True,
            "result": str(result),
            "numeric": float(result.evalf()) if result.is_number else None,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def solve_equation(equation_str: str, var_str: str = "x") -> dict:
    """Solve an equation symbolically."""
    try:
        var = symbols(var_str)
        # Support "lhs = rhs" format
        if "=" in equation_str:
            parts = equation_str.split("=", 1)
            lhs = parse_expr(parts[0].strip(), transformations=TRANSFORMATIONS)
            rhs = parse_expr(parts[1].strip(), transformations=TRANSFORMATIONS)
            eq = lhs - rhs
        else:
            eq = parse_expr(equation_str, transformations=TRANSFORMATIONS)

        solutions = solve(eq, var)
        return {
            "success": True,
            "solutions": [str(s) for s in solutions],
            "numeric": [float(s.evalf()) for s in solutions if s.is_number],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def differentiate(expr_str: str, var_str: str = "x", order: int = 1) -> dict:
    """Compute derivative."""
    try:
        var = symbols(var_str)
        expr = parse_expr(expr_str, transformations=TRANSFORMATIONS)
        result = diff(expr, var, order)
        return {"success": True, "result": str(result), "latex": sp.latex(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def integrate_expr(expr_str: str, var_str: str = "x",
                   lower: Optional[str] = None, upper: Optional[str] = None) -> dict:
    """Compute integral (definite or indefinite)."""
    try:
        var = symbols(var_str)
        expr = parse_expr(expr_str, transformations=TRANSFORMATIONS)

        if lower is not None and upper is not None:
            lo = parse_expr(lower, transformations=TRANSFORMATIONS)
            hi = parse_expr(upper, transformations=TRANSFORMATIONS)
            result = integrate(expr, (var, lo, hi))
        else:
            result = integrate(expr, var)

        return {"success": True, "result": str(result), "latex": sp.latex(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def compute_limit(expr_str: str, var_str: str = "x", point_str: str = "0") -> dict:
    """Compute limit."""
    try:
        var = symbols(var_str)
        expr = parse_expr(expr_str, transformations=TRANSFORMATIONS)
        point = parse_expr(point_str, transformations=TRANSFORMATIONS)
        result = limit(expr, var, point)
        return {"success": True, "result": str(result), "latex": sp.latex(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def matrix_ops(matrix_data: list, operation: str = "det") -> dict:
    """Perform matrix operations."""
    try:
        M = Matrix(matrix_data)
        ops = {
            "det": lambda: str(det(M)),
            "inverse": lambda: str(M.inv()),
            "eigenvalues": lambda: str(eigenvals(M)),
            "rank": lambda: str(M.rank()),
            "rref": lambda: str(M.rref()),
        }
        if operation not in ops:
            return {"success": False, "error": f"Unknown operation: {operation}"}

        result = ops[operation]()
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def verify_answer(student_answer: str, correct_expr: str) -> dict:
    """Check if student_answer is symbolically equivalent to correct_expr."""
    try:
        a = parse_expr(student_answer, transformations=TRANSFORMATIONS)
        b = parse_expr(correct_expr, transformations=TRANSFORMATIONS)
        diff_expr = simplify(a - b)
        is_correct = diff_expr == 0
        return {
            "is_correct": is_correct,
            "difference": str(diff_expr),
        }
    except Exception as e:
        return {"is_correct": False, "error": str(e)}
