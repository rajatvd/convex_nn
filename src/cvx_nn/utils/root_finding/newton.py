"""
Newton's method for computing roots of non-linear functions.
"""
from typing import Callable, Tuple, Dict, Any, Optional

import lab

# constants

TOL = 1e-6


# root-finders


def newton(
    obj_fn: Callable,
    grad_fn: Callable,
    w_0: lab.Tensor,
    tol: float = TOL,
    max_iters: int = 1000,
    upper_b: Optional[lab.Tensor] = None,
    lower_b: Optional[lab.Tensor] = None,
) -> Tuple[lab.Tensor, Dict[str, Any]]:
    """Find the roots of P functions in parallel using a (guarded) Newton's method. Computing
    the roots in parallel allows for the updates to be vectorized, which can be faster
    when solving a large number of simple root-finding problems. This comes at the cost
    of early termination for "easy" problems.
    :param obj_fn: a function that returns the objective when called at w.
    :param obj_fn: a function that returns the gradient when called at w.
    :param w_0: initial guess for the root for 'obj_fn'.
    :param tol: (optional) tolerance for finding a root.
    :param max_iters: (optional) the maximum number of iterations to run.
    :param upper_b: (optional) upper_bounds on the solution to the root-finding problem.
    :param lower_b: (optional) lower_bounds on the solution to the root-finding problem.
    :returns: TODO
    """
    success = False

    # setup
    i = 0
    w = w_0

    f_curr = obj_fn(w_0)
    g_curr = grad_fn(w_0)

    while i <= max_iters:
        # check stopping conditions for all iterates.
        if lab.max(lab.abs(f_curr)) < tol:
            success = True
            break
        i += 1

        # compute step.
        step = lab.safe_divide(
            -f_curr,
            g_curr,
        )

        # take step.
        w = w + step

        # guard the iterates
        if upper_b is not None:
            w = lab.minimum(w, upper_b)
        if lower_b is not None:
            w = lab.maximum(w, lower_b)

        f_curr = obj_fn(w)
        g_curr = grad_fn(w)

    exit_status = {"success": success, "n_iters": i}

    return w, exit_status
