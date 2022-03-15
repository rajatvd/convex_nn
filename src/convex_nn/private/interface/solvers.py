"""
Convert solvers from :module:`convex_nn.solvers` into internal optimizers and optimization procedures.
"""

from typing import Optional

from convex_nn.solvers import (
    Optimizer,
    RFISTA,
    AL,
    ConeDecomposition,
    LeastSquaresSolver,
    CVXPYSolver,
)
from convex_nn.regularizers import Regularizer, NeuronGL1, FeatureGL1, L2, L1
from convex_nn.metrics import Metrics
from .models import (
    ConvexReLU,
    ConvexGatedReLU,
)

from convex_nn.private.methods import (
    OptimizationProcedure,
    IterativeOptimizationProcedure,
    DoubleLoopProcedure,
    FISTA,
    AugmentedLagrangian,
    GradientNorm,
    ConstrainedOptimality,
    QuadraticBound,
    MultiplicativeBacktracker,
    Lassplore,
    LinearSolver,
    CVXPYGatedReLUSolver,
    CVXPYReLUSolver,
)
from convex_nn.private.methods import Optimizer as InteralOpt
import convex_nn.private.prox as prox


def build_prox_operator(
    regularizer: Optional[Regularizer] = None,
) -> prox.ProximalOperator:
    """Convert public facing regularizer into proximal operator.

    Args:
        regularizer: a public-facing regularizer object.

    Returns:
        A proximal operator for the regularizer.
    """
    lam = 0.0
    op: prox.ProximalOperator

    if regularizer is not None:
        lam = regularizer.lam

    if isinstance(regularizer, L2):
        op = prox.L2(lam)
    elif isinstance(regularizer, L1):
        op = prox.L1(lam)
    elif isinstance(regularizer, NeuronGL1):
        op = prox.GroupL1(lam)
    elif isinstance(regularizer, FeatureGL1):
        op = prox.GroupL1(lam, group_by_feature=True)
    elif regularizer is None:
        op = prox.Identity()
    else:
        raise ValueError(f"Optimizer does not support regularizer {regularizer}.")

    return op


def build_fista(
    regularizer: Optional[Regularizer],
) -> FISTA:
    """Helper function for constructing a default instance of R-FISTA.

    Args:
        regularizer: a public-facing regularizer object.

    Returns:
        Instance of the R-FISTA optimizer.
    """

    prox = build_prox_operator(regularizer)

    return FISTA(
        10.0,
        QuadraticBound(),
        MultiplicativeBacktracker(beta=0.8),
        Lassplore(alpha=1.25, threshold=5.0),
        prox=prox,
    )


def build_optimizer(
    optimizer: Optimizer,
    regularizer: Optional[Regularizer],
    metrics: Metrics,
) -> OptimizationProcedure:
    """Convert public facing solver into an internal optimization procedure.

    Args:
        optimizer: a public-facing optimizer object.
        regularizer: a public-facing regularizer object.
        metrics: a metrics object specifying which metrics to collect during optimization.

    Returns:
        An optimization procedure.
    """

    opt: InteralOpt
    opt_proc: OptimizationProcedure

    if isinstance(optimizer, RFISTA):

        max_iters = optimizer.max_iters
        term_criterion = GradientNorm(optimizer.tol)
        opt = build_fista(regularizer)

        opt_proc = IterativeOptimizationProcedure(
            opt,
            max_iters,
            term_criterion,
            name="fista",
            divergence_check=True,
            log_freq=metrics.metric_freq,
        )

    elif isinstance(optimizer, AL):

        inner_term_criterion = GradientNorm(optimizer.tol)
        outer_term_criterion = ConstrainedOptimality(
            optimizer.tol, optimizer.constraint_tol
        )

        sub_opt = build_fista(regularizer)
        opt = AugmentedLagrangian(
            use_delta_init=True,
            subprob_tol=optimizer.tol,
        )
        opt_proc = DoubleLoopProcedure(
            sub_opt,
            opt,
            optimizer.max_primal_iters,
            optimizer.max_dual_iters,
            inner_term_criterion,
            outer_term_criterion,
            max_total_iters=optimizer.max_primal_iters,
            name="al",
            divergence_check=False,
            log_freq=metrics.metric_freq,
        )
    elif isinstance(optimizer, LeastSquaresSolver):
        if not (regularizer is None or isinstance(regularizer, L2)):
            raise ValueError(
                "LeastSquaresSolver only supports L2-regularization, or no regularizer."
            )
        linear_solver = LinearSolver(
            optimizer.solver, optimizer.max_iters, optimizer.tol, None
        )

        opt_proc = OptimizationProcedure(linear_solver)

    elif isinstance(optimizer, CVXPYSolver):
        # check which formulation we need to solve.
        model = optimizer.model

        if isinstance(model, ConvexReLU):
            opt = CVXPYReLUSolver(optimizer.solver, optimizer.solver_kwargs)
        elif isinstance(model, ConvexGatedReLU):
            opt = CVXPYGatedReLUSolver(optimizer.solver, optimizer.solver_kwargs)
        else:
            raise ValueError(f"Model {model} not recognized by CVXPYSolver.")

        opt_proc = OptimizationProcedure(opt)

    elif isinstance(optimizer, ConeDecomposition):
        raise NotImplementedError(
            "Optimization by Cone Decomposition is not supported yet!"
        )
    else:
        raise ValueError(f"Optimizer object {optimizer} not supported.")

    return opt_proc