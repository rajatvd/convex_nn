"""Pre-defined CVXPY programs.

TODO:
    - Look into merging some of the CVXPY code to reduce boilerplate.
"""

from .cvxpy_solver import CVXPYSolver

from .training_programs import (
    CVXPYGatedReLUSolver,
    CVXPYReLUSolver,
    CVXPYGatedLassoNetSolver,
    CVXPYReLULassoNetSolver,
)

from .cone_decomposition import (
    MinL2Decomposition,
    MinL1Decomposition,
    FeasibleDecomposition,
    MinRelaxedL2Decomposition,
    SOCPDecomposition,
)


__all__ = [
    "CVXPYSolver",
    "CVXPYGatedReLUSolver",
    "CVXPYReLUSolver",
    "CVXPYGatedLassoNetSolver",
    "CVXPYReLULassoNetSolver",
    "MinL2Decomposition",
    "MinL1Decomposition",
    "FeasibleDecomposition",
    "MinRelaxedL2Decomposition",
    "SOCPDecomposition",
]
