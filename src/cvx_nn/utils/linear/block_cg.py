"""
Vectorized implementation of the conjugate-gradient method for solving block-diagonal linear systems.
"""
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy.sparse.linalg import LinearOperator  # type: ignore

import lab

from cvx_nn.utils import BlockDiagonalMatrix


TOL = 1e-6


def block_cg_solve(
    linear_op: BlockDiagonalMatrix,
    targets: lab.Tensor,
    P: int,
    d: int,
    starting_blocks: Optional[lab.Tensor] = None,
    max_iters: int = 1000,
    tol: float = TOL,
    flatten: bool = False,
) -> Tuple[lab.Tensor, Dict[str, Any]]:
    """Use an iterative method to solve the block-diagonal linear system
        Xw = b,
    where X is a block-diagonal matrix given by 'linear_op', y by 'targets', and lambda by 'lam'.
    All block are must be the same shape (d x d) and there must be exactly P blocks.
    The number of targets must be exactly P * d.
    :param linear_op: a generalized linear-operator that evaluates matrix-vector products for
        the data matrix matrix X.
    :param targets: the response/targets y to predict.
    :param P: the number of (square) matrix blocks in the system.
    :param d: the dimensionality of each block.
    :param max_iters: (optional) the maximum number of iterations to run the solver.
    :param tol: (optional) the tolerance to which the linear system should be solved.
    :param flatten: (optional) whether or not to flatten the solution.
    :returns: (solution, exit_status) -- the solution to the least-squares problem and status of the solver.
    """

    # setup #
    itr = 0

    # we start with some blocks left out.
    blocks_to_solve = starting_blocks
    if blocks_to_solve is None:
        blocks_to_solve = lab.arange(P)  # defaults to all blocks.

    # subset targets
    targets = targets.reshape((P, d))
    reduced_targets = targets[blocks_to_solve]

    # compute relative stopping tolerance.
    target_norm = lab.sqrt(lab.sum(reduced_targets ** 2))
    atol = max(tol, target_norm * tol)

    # status
    exit_status: Dict[str, Any] = {"success": True}
    # init solution at 0.
    w_opt = lab.zeros_like(targets)

    # initialize at 0.
    w = lab.zeros_like(reduced_targets)

    # initial residual
    residual = -reduced_targets
    rho = reduced_targets

    r_sqr = lab.sum(residual ** 2, axis=1)
    # termination criteria
    while itr < max_iters:
        itr += 1

        # run P independent iterations of CG at once.
        A_rho = linear_op.dot(rho, blocks_to_solve)
        rho_A_rho = lab.sum(lab.multiply(rho, A_rho), axis=1)

        alpha = lab.expand_dims(lab.divide(r_sqr, rho_A_rho), axis=-1)
        w = w + alpha * rho

        # TODO: it may be better to recalculate the residual to
        # avoid accumulating numerical errors. This requires
        # one more matrix-vector product and so will be slower.

        residual = residual + alpha * A_rho

        # update beta, r_sqr
        r_sqr_plus = lab.sum(residual ** 2, axis=1)
        beta = lab.expand_dims(lab.divide(r_sqr_plus, r_sqr), axis=-1)
        r_sqr = r_sqr_plus

        rho = beta * rho - residual

        # remove solved systems to prevent extra work (and dangerous underflows)
        solved = lab.sqrt(r_sqr) <= atol
        unsolved = lab.sqrt(r_sqr) > atol

        if lab.any(solved):
            indices = blocks_to_solve[solved]
            # save solutions
            w_opt[indices] = w[solved]

            # the problem is solved; exit.
            if not lab.any(unsolved):
                break

            # form reduced system
            w = w[unsolved]
            rho = rho[unsolved]
            residual = residual[unsolved]
            r_sqr = r_sqr[unsolved]
            # update system index
            blocks_to_solve = blocks_to_solve[unsolved]

    if itr == max_iters:
        # tolerance wasn't reached for some blocks
        exit_status["success"] = False
        # save those blocks anyway
        w_opt[blocks_to_solve] = w[blocks_to_solve]

    # record the number of iterations
    exit_status["iterations"] = itr

    if flatten:
        w_opt = w_opt.reshape(-1)

    return w_opt, exit_status
