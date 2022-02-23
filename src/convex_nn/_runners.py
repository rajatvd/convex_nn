"""
Interface between `convex_nn.public` module and optimization procedures in `convex_nn.private`.
"""
from typing import Optional, Any

import numpy as np

import lab

from convex_nn.public.models import Model
from convex_nn.public.optimizers import Optimizer
from convex_nn.public.regularizers import Regularizer
from convex_nn.private.utils.data.transforms import unitize_columns

from ._interface import build_model, build_optimizer, update_ext_model

# TODO: implement `Metrics` class which takes a boolean for each possible metric?


def optimize_model(
    model: Model,
    regularizer: Regularizer,
    optimizer: Optimizer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    metrics: Any = None,  # TODO
    log_freq: int = 25,
    return_convex: bool = False,
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
    opt_procedure = build_optimizer(optimizer, regularizer, log_freq)

    # TODO: initializers are a pain...
    # We don't support for now and then add support later.

    exit_status, internal_model, metrics = opt_procedure(
        logger,  # TODO: handle loggers
        internal_model,
        None,  # TODO: handle initializers
        (X_train, y_train),
        (X_test, y_test),
        (["objective", "grad_norm"], [], []),
    )

    # transform model back to original data space.
    internal_model.weights = _transform_weights(internal_model.weights, column_norms)

    # update public-facing model
    model = update_ext_model(model, internal_model)

    # TODO: update modify get_nc_formulation to return public models.
    # TODO should probably move the get_nc_formulation into public?

    if return_convex:
        return model

    # convert into non-convex model
    return get_nc_formulation(convex_model, remove_sparse=True)


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
