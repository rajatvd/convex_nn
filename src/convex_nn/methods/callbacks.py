"""
Callback functions to be executed after each iteration of optimization.
"""
from typing import Tuple, Dict, Optional

import numpy as np

import lab

from convex_nn.models import Model
from convex_nn.methods.optimizers import Optimizer


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

    def __call__(
        self, model: Model, X: lab.Tensor, y: lab.Tensor
    ) -> Model:

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
