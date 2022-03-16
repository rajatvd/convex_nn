"""
Callback functions to be executed after each iteration of optimization.
"""
from typing import Tuple, Dict, Optional, Callable

import numpy as np

import lab

from convex_nn.private.models import Model
from convex_nn.private.methods.optimizers.pgd import PGDLS
from convex_nn.private.methods.line_search import (
    QuadraticBound,
    MultiplicativeBacktracker,
    Lassplore,
)
from convex_nn.private.prox import ProximalOperator


class ObservedSignPatterns:

    """Updates the set of active hyperplane arrangements by simultaneously
    running (S)GD on a ReLU MLP of fixed size."""

    def __init__(self):
        """
        :param relu_mlp: a ReLU MLP to be optimized concurrently with the convex model.
        :param optimizer: an optimizer for the ReLU MLP.
        """

        self.observed_patterns = {}
        self.hasher = hash

    def _get_hashes(self, D: lab.Tensor):
        return np.apply_along_axis(
            lambda z: self.hasher(np.array2string(z)), axis=0, arr=lab.to_np(D)
        )

    def _check_and_store_pattern(self, pattern) -> bool:
        if pattern in self.observed_patterns:
            return False
        else:
            self.observed_patterns[pattern] = True
            return True

    def __call__(self, model: Model, X: lab.Tensor, y: lab.Tensor) -> Model:

        # compute new sign patterns
        patterns, weights = model.sign_patterns(X, y)

        indices_to_keep = lab.tensor(
            [
                self._check_and_store_pattern(pattern)
                for pattern in self._get_hashes(patterns)
            ]
        )
        new_patterns = patterns[:, indices_to_keep]
        new_weights = weights[indices_to_keep]

        if model.activation_history is None or not hasattr(model, "activation_history"):
            model.activation_history = new_patterns
            model.weight_history = new_weights
        else:
            model.activation_history = lab.concatenate(
                [model.activation_history, new_patterns], axis=1
            )
            model.weight_history = lab.concatenate(
                [model.weight_history, new_weights], axis=0
            )

        return model


class ConeDecomposition:

    """Convert a gated ReLU model into a ReLU model by solving the cone decomposition problem."""

    def __init__(self, solver: Callable):
        """
        :param solver: solver to use when computing the cone decomposition.
        """

        self.solver = solver

    def __call__(self, model: Model, X: lab.Tensor, y: lab.Tensor) -> Model:

        decomposed_model, _ = self.solver(model, X, y)

        return decomposed_model


class ProximalCleanup:
    """Cleanup the solution to an optimization problem by taking one proximal-gradient step."""

    def __init__(self, prox: ProximalOperator):
        """
        :param prox: the proximal-operator to use when cleaning-up the model solution.
        """
        self.prox = prox
        # create optimizer to use
        self.optimizer = PGDLS(
            1.0,
            QuadraticBound(),
            MultiplicativeBacktracker(beta=0.8),
            Lassplore(alpha=1.25, threshold=5.0),
            prox=self.prox,
        )

    def __call__(self, model: Model, X: lab.Tensor, y: lab.Tensor) -> Model:
        cleaned_model, _, exit_state = self.optimizer.step(model, X, y)

        return cleaned_model
