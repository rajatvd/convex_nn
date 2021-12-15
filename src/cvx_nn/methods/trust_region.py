"""
Solvers for trust-region problems.
"""

from typing import Optional, Callable, Tuple, Dict, Any

import numpy as np

# import trust-region solvers from within SciPy.
import scipy.optimize._trustregion_ncg as ncg  # type: ignore
import scipy.optimize._trustregion_krylov as krylov  # type: ignore
from scipy.sparse.linalg import LinearOperator  # type: ignore

# TODO: This implementation of Steihaug-Toint doesn't support preconditioners.
#   An alternative is the Dominique Orban's [implementation](https://github.com/optimizers/nlpy/blob/master/nlpy/krylov/pcg.py),
#   which does. This latter library isn't activity maintained, however.


def steihaug_cg_solver(
    radius: float,
    w: np.ndarray,
    obj_fn: Callable,
    grad_fn: Callable,
    hessian_fn: Optional[Callable] = None,
    hessian_op: Optional[LinearOperator] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve trust-region sub-problem using the Steihaug-Toint method.
        This function wraps around an underlying SciPy solver, which
        may be subject to internal API changes. Note that only
        one of 'hessian' and 'hessian_op' needs to be supplied.
    :param radius: radius of the trust region.
    :param w: the parameters to be updated.
    :param obj_fn: function that returns the objective when called.
    :param grad_fn: function that returns the gradient of the objective function when called.
    :param hessian: the Hessian of the objective function.
    :param hessian_op: a SciPy LinearOperator which can be used to evaluate
        hessian-vector products.
    :returns: (w_next, exit_state) -- the updated parameters and exit state of the CG solver.
    """

    solver = ncg.CGSteihaugSubproblem(w, obj_fn, grad_fn, hessian_fn, hessian_op)
    w_plus, on_boundary = solver.solve(radius)
    exit_state = {"on_boundary": on_boundary}
    return w_plus, exit_state


def krylov_solver(
    radius: float,
    w: np.ndarray,
    obj_fn: Callable,
    grad_fn: Callable,
    hessian_fn: Optional[Callable] = None,
    hessian_op: Optional[LinearOperator] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve trust-region sub-problem using a Krylov sub-space method.
        This function wraps around an underlying SciPy solver, which
        may be subject to internal API changes. Note that only
        one of 'hessian' and 'hessian_op' needs to be supplied.
    :param radius: radius of the trust region.
    :param w: the parameters to be updated.
    :param obj_fn: function that returns the objective when called.
    :param grad_fn: function that returns the gradient of the objective function when called.
    :hessian: the Hessian of the objective function.
    :hessian_op: a SciPy LinearOperator which can be used to evaluate
        hessian-vector products.
    :returns: solution to the trust-region problem,
    """

    # There are several options for implementing this solver.
    # 1) Access the protected SciPy module for this solver.
    #   This is a wrapper around [TRLIB](https://trlib.readthedocs.io/en/latest/installation.html).
    # 2) Compile the TRLIB binaries directly.
    #   This has the advantage of support for preconditioners, which the SciPy wrappers do not allow.

    raise NotImplementedError("Krylov sub-space solvers haven't been implemented yet.")
