"""
Utilities.
"""

from .linear_operators import MatVecOperator, BlockDiagonalMatrix
from .root_finding import secant, newton

from .linear import (
    block_cg_solve,
    solve_ne,
    lstsq_iterative_solve,
    linear_iterative_solve,
    get_preconditioner,
)

__all__ = [
    "BlockDiagonalMatrix",
    "secant",
    "newton",
    "block_cg_solve",
    "solve_ne",
    "lstsq_iterative_solve",
    "linear_iterative_solve",
    "get_preconditioner",
]
