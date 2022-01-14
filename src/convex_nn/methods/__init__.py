"""
Methods.
"""

# ===== module exports ===== #

from .optimization_procedures import (
    OptimizationProcedure,
    IterativeOptimizationProcedure,
    TorchLoop,
    DoubleLoopProcedure,
    METRIC_FREQ,
    ITER_LOG_FREQ,
    EPOCH_LOG_FREQ,
)

from .callbacks import ObservedSignPatterns

from .core import (
    ls,
    gradient_step,
    gd_ls,
    proximal_gradient_step,
    proximal_gradient_ls,
    fista_step,
    fista_ls,
    update_multipliers,
    acc_update_multipliers,
)

from .optimizers import (
    Optimizer,
    ProximalOptimizer,
    ProximalLSOptimizer,
    MetaOptimizer,
    GD,
    GDLS,
    PGD,
    PGDLS,
    FISTA,
    AugmentedLagrangian,
)

from .line_search import (
    Backtracker,
    MultiplicativeBacktracker,
    LSCondition,
    QuadraticBound,
    Armijo,
    StepSizeUpdater,
    KeepNew,
    KeepOld,
    ForwardTrack,
    Lassplore,
)

from .external_solver import LinearSolver

from .cvxpy_solvers import (
    CVXPYSolver,
    RelaxedMLPSolver,
    OrthantConstrainedMLPSolver,
    RelaxedLassoNetSolver,
    OrthantConstrainedLassoNetSolver,
)

from .termination_criteria import (
    GradientNorm,
    StepLength,
    ConstrainedOptimality,
)


__all__ = [
    "OptimizationProcedure",
    "IterativeOptimizationProcedure",
    "TorchLoop",
    "DoubleLoopProcedure",
    "ObservedSignPatterns",
    "ls",
    "gradient_step",
    "gd_ls",
    "proximal_gradient_step",
    "proximal_gradient_ls",
    "fista_step",
    "fista_ls",
    "update_multipliers",
    "acc_update_multipliers",
    "Optimizer",
    "ProximalOptimizer",
    "ProximalLSOptimizer",
    "MetaOptimizer",
    "GD",
    "GDLS",
    "PGD",
    "PGDLS",
    "FISTA",
    "AugmentedLagrangian",
    "LinearSolver",
    "CVXPYSolver",
    "RelaxedMLPSolver",
    "OrthantConstrainedMLPSolver",
    "RelaxedLassoNetSolver",
    "OrthantConstrainedLassoNetSolver",
    "Backtracker",
    "MultiplicativeBacktracker",
    "LSCondition",
    "QuadraticBound",
    "Armijo",
    "StepSizeUpdater",
    "KeepNew",
    "KeepOld",
    "ForwardTrack",
    "Lassplore",
    "GradientNorm",
    "StepLength",
    "ConstrainedOptimality",
]
