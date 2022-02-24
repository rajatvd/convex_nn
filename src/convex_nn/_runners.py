"""
Interface between `convex_nn.public` module and optimization procedures in `convex_nn.private`.
"""
from typing import Optional, Any
import logging

import numpy as np

import lab

from convex_nn.public.models import Model
from convex_nn.public.optimizers import Optimizer
from convex_nn.public.regularizers import Regularizer
from convex_nn.public.metrics import Metrics
from convex_nn.private.utils.data.transforms import unitize_columns
from convex_nn.private.models.solution_mappings import get_nc_formulation

from ._interface import (
    build_model,
    build_optimizer,
    update_ext_model,
    update_ext_metrics,
    build_ext_nc_model,
)


def optimize_model(
    model: Model,
    regularizer: Regularizer,
    optimizer: Optimizer,
    metrics: Metrics,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    return_convex: bool = False,
    verbose: bool = False,
    log_file: Optional[str] = None,
) -> Model:

    # Note: this unitizes columns of data matrix.
    X_train, y_train, test_set, column_norms = _process_data(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # TODO: what about warm starting the dual parameters?
    # Solution: Implement a separate loop for the full solution path.

    internal_model = build_model(model, regularizer, X_train)
    opt_procedure = build_optimizer(optimizer, regularizer, metrics)

    logger = _get_logger("convex_nn", verbose, False, log_file)

    exit_status, internal_model, internal_metrics = opt_procedure(
        logger,
        internal_model,
        None,  # TODO: handle initializers
        (X_train, y_train),
        (X_test, y_test),
        (["objective", "grad_norm"], [], []),
    )

    metrics = update_ext_metrics(metrics, internal_metrics)

    # convert internal metrics

    # transform model back to original data space.
    internal_model.weights = _transform_weights(internal_model.weights, column_norms)

    # update public-facing model
    update_ext_model(model, internal_model)

    # TODO: update modify get_nc_formulation to return public models.
    # TODO should probably move the get_nc_formulation into public?

    if return_convex:
        return update_ext_model(model, internal_model)

    # convert into internal non-convex model
    nc_internal_model = get_nc_formulation(
        internal_model, implementation="manual", remove_sparse=True
    )

    # create non-convex model
    return build_ext_nc_model(nc_internal_model)


# TODO: implement `optimize_path`.
def optimize_path():
    pass


# Helpers

# probably move this stuff to private?


def _transform_weights(model_weights, column_norms):
    return model_weights / column_norms


def _untransform_weights(model_weights, column_norms):
    return model_weights * column_norms


def _process_data(X_train, y_train, X_test, y_test):

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

    column_norms = None
    train_set, test_set, column_norms = unitize_columns(train_set, test_set)

    return train_set[0], train_set[1], (X_test, y_test), column_norms


def _get_logger(
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
