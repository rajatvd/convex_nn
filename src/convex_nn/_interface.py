"""
Interface between public objects in `convex_nn.public` module and private objects in `convex_nn.private`.
"""

from typing import Optional, Tuple, List

import lab

# Public Facing Objects

from convex_nn.public.optimizers import Optimizer, RFISTA, AL, ConeDecomposition
from convex_nn.public.regularizers import Regularizer, NeuronGL1, FeatureGL1, L2
from convex_nn.public.models import (
    Model,
    ConvexGatedReLU,
    ConvexReLU,
)
from convex_nn.public.metrics import Metrics

# Private Objects

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
)
from convex_nn.private.methods import Optimizer as InteralOpt
from convex_nn.private.prox import ProximalOperator, GroupL1, Identity

from convex_nn.private.models import (
    sign_patterns,
    ConvexMLP,
    AL_MLP,
    GroupL1Regularizer,
    L2Regularizer,
)

from convex_nn.private.models import Model as InternalModel
from convex_nn.private.models import Regularizer as InternalRegularizer


def build_prox_operator(regularizer: Optional[Regularizer] = None) -> ProximalOperator:
    """Convert public facing regularizer into proximal operator."""
    lam = 0.0
    prox: ProximalOperator

    if regularizer is not None:
        lam = regularizer.lam

    if isinstance(regularizer, NeuronGL1):
        prox = GroupL1(lam)
    elif isinstance(regularizer, FeatureGL1):
        prox = GroupL1(lam, group_by_feature=True)
    elif isinstance(regularizer, L2) or regularizer is None:
        prox = Identity()

    return prox


def build_fista(
    regularizer: Optional[Regularizer],
) -> FISTA:

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
    """Convert public facing optimizer objects into private implementations."""

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

    elif isinstance(optimizer, ConeDecomposition):
        raise NotImplementedError(
            "Optimization by Cone Decomposition is not supported yet!"
        )
    else:
        raise ValueError(f"Optimizer object {optimizer} not supported.")

    return opt_proc


def build_regularizer(regularizer: Optional[Regularizer] = None) -> InternalRegularizer:
    reg: InternalRegularizer

    lam = 0.0
    if regularizer is not None:
        lam = regularizer.lam

    if isinstance(regularizer, NeuronGL1):
        reg = GroupL1Regularizer(lam)
    elif isinstance(regularizer, FeatureGL1):
        reg = GroupL1Regularizer(lam, group_by_feature=True)
    elif isinstance(regularizer, L2) or regularizer is None:
        reg = L2Regularizer(lam)

    return reg


def build_model(model: Model, regularizer: Regularizer, X: lab.Tensor) -> InternalModel:
    """Convert public facing model objects into private implementations."""
    assert isinstance(model, (ConvexReLU, ConvexGatedReLU))

    internal_model: InternalModel
    d, c = model.d, model.p
    internal_reg = build_regularizer(regularizer)

    G = lab.tensor(model.G)
    D, U = sign_patterns.get_sign_patterns(X, None, U=G)

    if isinstance(model, ConvexReLU):
        internal_model = AL_MLP(
            d,
            D,
            U,
            "einsum",
            1000,
            regularizer=internal_reg,
            c=c,
        )
        internal_model.weights = lab.tensor(model.parameters[0])
    elif isinstance(model, ConvexGatedReLU):
        internal_model = ConvexMLP(d, D, U, "einsum", regularizer=internal_reg, c=c)
        internal_model.weights = lab.stack(
            [lab.tensor(model.parameters[0]), lab.tensor(model.parameters[1])]
        )
    else:
        raise ValueError(f"Model object {model} not supported.")

    return internal_model


def build_metrics_tuple(
    metrics: Metrics,
) -> Tuple[List[str], List[str], List[str]]:
    """Convert Metrics instance into tuple of metrics for internal use.

    Args:
        metrics: object specifying which metrics should be collected during optimization.

    Returns:
        (train_metrics, test_metrics, additional_metrics) --- tuple of list of strings specifying which metrics should be collected.
    """

    for key, value in metrics.metrics_to_collect.items():
        if key == "":
            pass
        elif key == "":
            pass
        elif key == "":
            pass
        elif key == "":
            pass


def update_ext_model(model: Model, internal_model: InternalModel) -> Model:
    if isinstance(model, ConvexGatedReLU):
        assert isinstance(internal_model, ConvexMLP)
        model.parameters = [lab.to_np(internal_model.weights)]
    elif isinstance(model, ConvexReLU):
        assert isinstance(internal_model, AL_MLP)
        model.parameters = [
            lab.to_np(internal_model.weights[0]),
            lab.to_np(internal_model.weights[1]),
        ]

    return model
