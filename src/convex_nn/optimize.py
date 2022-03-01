"""
Optimize neural networks using convex reformulations.
"""
from typing import Optional, Tuple, Literal
import math

import numpy as np

import lab

from convex_nn.models import Model, ConvexGatedReLU, ConvexReLU
from convex_nn.solvers import Optimizer, RFISTA, AL
from convex_nn.regularizers import Regularizer, NeuronGL1
from convex_nn.metrics import Metrics
from convex_nn.activations import sample_gate_vectors
from convex_nn.private.models.solution_mappings import get_nc_formulation

from convex_nn.private.interface import (
    build_model,
    build_optimizer,
    build_metrics_tuple,
    update_ext_model,
    update_ext_metrics,
    build_ext_nc_model,
    get_logger,
    transform_weights,
    process_data,
)

# Types

Formulation = Literal["gated_relu", "relu"]


def optimize(
    formulation: Formulation,
    max_neurons: int,
    lam: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    return_convex: bool = False,
    verbose: bool = False,
    log_file: str = None,
    backend: str = "numpy",
    device: str = "cpu",
    dtype: str = "float32",
    seed: int = 778,
) -> Tuple[Model, Metrics]:
    """Convenience function for training neural networks by convex reformulation.

    Args:
        formulation: the convex reformulation to solve. Must be one of
            `gated_relu` - train a network with Gated ReLU activations.
            `relu` - train a network with ReLU activations.
        max_neurons: the maximum number of neurons in the convex reformulation.
            The final model will be neuron-sparse when `lam` is moderate and so `max_neurons` should typically be as large as computationally possible.
        lam: the regularization strength.
        X_train: an :math:`n \\times d` matrix of training examples.
        y_train: an :math:`n \\times c` or vector matrix of training targets.
        X_test: an :math:`m \\times d` matrix of test examples.
        y_test: an :math:`n \\times c` or vector matrix of test targets.
        return_convex: whether or not to return the convex reformulation instead of the final non-convex model.
        verbose: whether or not the solver should print verbosely during optimization.
        log_file: a path to an optional log file.

    Returns:
        (Model, Metrics): the optimized model and metrics collected during optimization.
    """

    model: Model
    solver: Optimizer

    n, d = X_train.shape
    c = 1 if len(y_train.shape) == 1 else y_train.shape[1]

    # Instantiate convex model and other options.
    if formulation == "gated_relu":
        G = sample_gate_vectors(np.random.default_rng(seed), d, max_neurons)
        model = ConvexGatedReLU(G, c=c)
        solver = RFISTA(model)
    elif formulation == "relu":
        G = sample_gate_vectors(
            np.random.default_rng(seed), d, math.floor(max_neurons / 2)
        )
        model = ConvexReLU(G, c=c)
        solver = AL(model)
    else:
        raise ValueError(f"Convex formulation {formulation} not recognized!")

    regularizer = NeuronGL1(lam)
    metrics = Metrics(
        metric_freq=1,
        model_loss=True,
        train_accuracy=True,
        test_accuracy=True,
        neuron_sparsity=True,
    )

    return optimize_model(
        model,
        solver,
        metrics,
        X_train,
        y_train,
        X_test,
        y_test,
        regularizer,
        return_convex,
        verbose,
        log_file,
        backend,
        device,
        dtype,
        seed,
    )


def optimize_model(
    model: Model,
    solver: Optimizer,
    metrics: Metrics,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    regularizer: Optional[Regularizer] = None,
    return_convex: bool = False,
    verbose: bool = False,
    log_file: str = None,
    backend: str = "numpy",
    device: str = "cpu",
    dtype: str = "float32",
    seed: int = 778,
) -> Tuple[Model, Metrics]:
    """Train a neural network by convex reformulation.

    Args:
        model: a convex reformulation of a neural network model.
        solver: the optimizer to use when solving the reformulation.
        metrics: a object specifying which metrics to collect during optimization.
        X_train: an :math:`n \\times d` matrix of training examples.
        y_train: an :math:`n \\times c` or vector matrix of training targets.
        X_test: an :math:`m \\times d` matrix of test examples.
        y_test: an :math:`n \\times c` or vector matrix of test targets.
        regularizer: an optional regularizer for the convex reformulation.
        return_convex: whether or not to return the convex reformulation instead of the final non-convex model.
        verbose: whether or not the solver should print verbosely during optimization.
        log_file: a path to an optional log file.

    Returns:
        (Model, Metrics): the optimized model and metrics collected during optimization.
    """

    # set backend settings.
    lab.set_backend(backend)
    lab.set_device(device)
    lab.set_dtype(dtype)
    lab.set_seeds(seed)

    # Note: this unitizes columns of data matrix.
    (X_train, y_train), (X_test, y_test), column_norms = process_data(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    internal_model = build_model(model, regularizer, X_train)
    opt_procedure = build_optimizer(solver, regularizer, metrics)
    metrics_tuple = build_metrics_tuple(metrics)

    logger = get_logger("convex_nn", verbose, False, log_file)

    initializer = lambda model: model

    exit_status, internal_model, internal_metrics = opt_procedure(
        logger,
        internal_model,
        initializer,
        (X_train, y_train),
        (X_test, y_test),
        metrics_tuple,
    )

    metrics = update_ext_metrics(metrics, internal_metrics)

    # convert internal metrics

    # transform model back to original data space.
    internal_model.weights = transform_weights(internal_model.weights, column_norms)

    # update public-facing model
    update_ext_model(model, internal_model)

    if return_convex:
        return update_ext_model(model, internal_model), Metrics

    # convert into internal non-convex model
    nc_internal_model = get_nc_formulation(
        internal_model, implementation="manual", remove_sparse=True
    )

    # create non-convex model
    return build_ext_nc_model(nc_internal_model), metrics


def optimize_path():
    pass
