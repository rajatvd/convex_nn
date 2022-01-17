"""
Basic interface for CVXPY-base solvers.
"""

from typing import Dict, Any, Tuple

import cvxpy as cp
import numpy as np

from convex_nn.methods.external_solver import ExternalSolver
from convex_nn.models import Model


class CVXPYSolver(ExternalSolver):
    """
    Interface for solvers based on the CVXPY DSL.
    """

    def __init__(self, solver: str = "ecos", kwargs={}):
        """
        :param solver: the solver to use with CVXPY.
        """

        self.solver_kwargs = kwargs

        # save the desired solver
        if solver == "ecos":
            self.solver = cp.ECOS_BB
        elif solver == "cvxopt":
            self.solver = cp.CVXOPT
        elif solver == "scs":
            self.solver = cp.SCS
        # note: the following are commercial solvers requiring a licence.
        elif solver == "gurobi":
            self.solver = cp.GUROBI
        elif solver == "mosek":
            self.solver = cp.MOSEK
        else:
            raise ValueError(f"CVXPY solver {solver} not recognized!")

    def __call__(
        self, model: Model, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Model, Dict[str, Any]]:
        """Call the CVXPY solver to fit the model."""

        raise NotImplementedError("A CVXPY-based solver must implement '__call__'.")
