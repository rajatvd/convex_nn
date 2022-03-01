"""
Interface between public objects in `convex_nn.public` module and private objects in `convex_nn.private`.
"""

from typing import Optional, Tuple, List, Dict, Any
import logging

import numpy as np
import lab

# Public Facing Objects

from convex_nn.solvers import Optimizer, RFISTA, AL, ConeDecomposition
from convex_nn.regularizers import Regularizer, NeuronGL1, FeatureGL1, L2
from convex_nn.models import (
    Model,
    ConvexGatedReLU,
    NonConvexGatedReLU,
    ConvexReLU,
    NonConvexReLU,
)
from convex_nn.metrics import Metrics
from convex_nn.activations import compute_activation_patterns

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
    ConvexMLP,
    AL_MLP,
    ReLUMLP,
    GatedReLUMLP,
    GroupL1Regularizer,
    L2Regularizer,
)

from convex_nn.private.models import Model as InternalModel
from convex_nn.private.models import Regularizer as InternalRegularizer
from convex_nn.private.utils.data.transforms import unitize_columns


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
        reg = GroupL1Regularizer(lam, group_by_feature=False)
    elif isinstance(regularizer, FeatureGL1):
        reg = GroupL1Regularizer(lam, group_by_feature=True)
    elif isinstance(regularizer, L2) or regularizer is None:
        reg = L2Regularizer(lam)

    return reg


def build_model(model: Model, regularizer: Regularizer, X: lab.Tensor) -> InternalModel:
    """Convert public facing model objects into private implementations."""
    assert isinstance(model, (ConvexReLU, ConvexGatedReLU))

    internal_model: InternalModel
    d, c = model.d, model.c
    internal_reg = build_regularizer(regularizer)

    G = lab.tensor(model.G)
    D, G = lab.all_to_tensor(compute_activation_patterns(lab.to_np(X), lab.to_np(G)))

    if isinstance(model, ConvexReLU):
        internal_model = AL_MLP(
            d,
            D,
            G,
            "einsum",
            1000,
            regularizer=internal_reg,
            c=c,
        )
        internal_model.weights = lab.stack(
            [lab.tensor(model.parameters[0]), lab.tensor(model.parameters[1])]
        )
    elif isinstance(model, ConvexGatedReLU):
        internal_model = ConvexMLP(d, D, G, "einsum", regularizer=internal_reg, c=c)
        internal_model.weights = lab.tensor(model.parameters[0])
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
    train_metrics = []
    test_metrics = []
    additional_metrics = []

    for key, value in metrics.metrics_to_collect.items():
        if not value:
            continue

        if key == "objective":
            train_metrics.append("objective")
        elif key == "grad_norm":
            train_metrics.append("grad_norm")
        elif key == "time":
            continue
        elif key == "model_loss":
            train_metrics.append("base_objective")
        elif key == "lagrangian_grad":
            train_metrics.append("lagrangian_grad")
        # use convex model for training (same as non-convex)
        elif key == "train_accuracy":
            train_metrics.append("accuracy")
        # use convex model for training (same as non-convex)
        elif key == "train_mse":
            train_metrics.append("nc_squared_error")
        # use non-convex model for testing
        elif key == "test_accuracy":
            test_metrics.append("nc_accuracy")
        # use non-convex model for testing
        elif key == "test_mse":
            test_metrics.append("squared_error")
        elif key == "total_neurons":
            additional_metrics.append("total_neurons")
        elif key == "neuron_sparsity":
            additional_metrics.append("group_sparsity")
        elif key == "total_features":
            additional_metrics.append("total_features")
        elif key == "feature_sparsity":
            additional_metrics.append("feature_sparsity")
        elif key == "total_weights":
            additional_metrics.append("total_weights")
        elif key == "weight_sparsity":
            additional_metrics.append("weight_sparsity")

    return train_metrics, test_metrics, additional_metrics


def update_ext_metrics(metrics: Metrics, internal_metrics: Dict[str, Any]) -> Metrics:
    for key, value in internal_metrics.items():
        if key == "train_objective":
            metrics.objective = np.array(value)
        elif key == "train_grad_norm":
            metrics.grad_norm = np.array(value)
        elif key == "time":
            metrics.time = np.cumsum(np.array(value))
        elif key == "train_base_objective":
            metrics.model_loss = value
        elif key == "train_lagrangian_grad":
            metrics.lagrangian_grad = value
        elif key == "train_accuracy":
            metrics.train_accuracy = value
        # use convex model for training (same as non-convex)
        elif key == "train_squared_error":
            metrics.train_mse = value
        # use non-convex model for testing
        elif key == "test_nc_accuracy":
            metrics.test_accuracy = value
        # use non-convex model for testing
        elif key == "test_nc_squared_error":
            metrics.test_mse = value
        elif key == "total_neurons":
            metrics.total_neurons = value
        elif key == "neuron_sparsity":
            metrics.neuron_sparsity = value
        elif key == "total_features":
            metrics.total_features = value
        elif key == "feature_sparsity":
            metrics.feature_sparsity = value
        elif key == "total_weights":
            metrics.total_weights = value
        elif key == "weight_sparsity":
            metrics.weight_sparsity = value

    return metrics


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


def build_ext_nc_model(internal_nc_model: InternalModel) -> Model:
    model: Model

    if isinstance(internal_nc_model, GatedReLUMLP):
        U = lab.to_np(internal_nc_model.U)
        model = NonConvexGatedReLU(U, internal_nc_model.c)
        w1, w2 = internal_nc_model._split_weights(internal_nc_model.weights)

        model.set_parameters([lab.to_np(w1), lab.to_np(w2)])

    elif isinstance(internal_nc_model, ReLUMLP):
        model = NonConvexReLU(
            internal_nc_model.d, internal_nc_model.p, internal_nc_model.c
        )
        w1, w2 = internal_nc_model._split_weights(internal_nc_model.weights)

        model.set_parameters([lab.to_np(w1), lab.to_np(w2)])
    else:
        raise ValueError(f"Non-convex model {internal_nc_model} not recognized.")

    return model


def transform_weights(model_weights, column_norms):
    return model_weights / column_norms


def untransform_weights(model_weights, column_norms):
    return model_weights * column_norms


def process_data(X_train, y_train, X_test, y_test):

    # add extra target dimension if necessary
    if len(y_train.shape) == 1:
        y_train = lab.expand_dims(y_train, axis=1)
        y_test = lab.expand_dims(y_test, axis=1)

    train_set = (
        lab.tensor(X_train.tolist(), dtype=lab.get_dtype()),
        lab.tensor(y_train.tolist(), dtype=lab.get_dtype()),
    )

    test_set = (
        (
            lab.tensor(X_test.tolist(), dtype=lab.get_dtype()),
            lab.tensor(y_test.tolist(), dtype=lab.get_dtype()),
        )
        if X_test is not None
        else train_set
    )

    return unitize_columns(train_set, test_set)


def get_logger(
    name: str, verbose: bool = False, debug: bool = False, log_file: str = None
) -> logging.Logger:
    """Construct a logging.Logger instance with an appropriate configuration.

    Args:
        name: name for the Logger instance.
        verbose: (optional) whether or not the logger should print verbosely (ie. at the INFO level).
            Defaults to False.
        debug: (optional) whether or not the logger should print in debug mode (ie. at the DEBUG level).
            Defaults to False.
        log_file: (optional) path to a file where the log should be stored. The log is printed to stdout when 'None'.
    Returns:
         Instance of logging.Logger.
    """

    level = logging.WARNING
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO

    logging.basicConfig(level=level, filename=log_file)
    logger = logging.getLogger(name)
    logging.root.setLevel(level)
    logger.setLevel(level)
    return logger
