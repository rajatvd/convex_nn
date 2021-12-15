"""
Implementation of one-layer LassoNet with Gated ReLU activations.
"""

from typing import Optional, Tuple, List

import lab

from .relu_mlp import ReLUMLP
from cvx_nn.models.model import Model
from cvx_nn.models.regularizers import Regularizer
from cvx_nn.loss_functions import squared_error, relu


class ReLULassoNet(Model):

    """One-layer ReLU LassoNet with squared-error objective."""

    def __init__(
        self,
        d: int,
        p: int,
        regularizer: Optional[Regularizer] = None,
        c: int = 1,
    ):
        """
        :param d: the dimensionality of the dataset (ie.number of features).
        :param U: the gate vectors associated with gated ReLU activations.
        :param regularizer: (optional) a penalty function controlling the flexibility of the model.
        :param c: (optional) the number of targets.
        """
        super().__init__(regularizer)
        self.c = c
        self.d = d
        self.p = p

        self.total_weights = d * self.p + self.p * c + d * c
        # use random initialization by default.
        self.weights = lab.tensor(
            lab.np_rng.standard_normal((self.total_weights)),
            dtype=lab.get_dtype(),
        )
        self.nl_model = ReLUMLP(d, p, None, c)
        # free memory of sub-model.
        self.nl_model.weights = None

    def set_weights(self, weights: lab.Tensor):
        # weights include second layer
        if weights.shape == (self.total_weights,):
            self.weights = weights
        # weights consist only of first-layer weights.
        elif weights.shape == (self.d * self.p,):
            self.weights[: self.p * self.d] = weights
        # weights consist only of second-layer weights.
        elif weights.shape == (self.p * self.c,):
            self.weights[self.p * self.d : self.p * self.c] = weights
        # weights consist of skip-layer weights
        elif weights.shape == (self.d * self.c,):
            self.weights[self.p * self.c :] = weights
        # weights need to be flattened.
        elif lab.size(weights) == (self.total_weights,):
            self.weights = lab.ravel(weights)
        else:
            raise ValueError(
                f"Weights with shape {weights.shape} cannot be set to ReluMLP with weight shape {self.d * self.p + p}."
            )

    def get_weights(self) -> List[lab.Tensor]:
        """Get model weights in an interpretable format.
        :returns: list of tensors -- [network_1, network_2, skip weights].
        """
        network_weights, skip_weights = self._split_weights(self.weights)

        w1, w2 = self.nl_model._split_weights(network_weights)

        return [w1, w2, skip_weights]

    def _split_weights(self, w: lab.Tensor) -> Tuple[lab.Tensor, lab.Tensor]:

        return (
            w[: self.d * self.p + self.p * self.c],
            w[self.d * self.p + self.p * self.c :].reshape(self.c, self.d),
        )

    def _join_weights(
        self, network_weights: lab.Tensor, skip_weights: lab.Tensor
    ) -> lab.Tensor:

        return lab.concatenate([lab.ravel(network_weights), lab.ravel(skip_weights)])

    def _forward(self, X: lab.Tensor, w: lab.Tensor, **kwargs) -> lab.Tensor:
        """Compute forward pass.
        :param X: (n,d) array containing the data examples.
        :param w: parameter at which to compute the forward pass.
        """
        network_weights, skip_weights = self._split_weights(w)

        return self.nl_model._forward(X, network_weights) + X @ skip_weights.T

    def _objective(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> float:
        """Compute objective associated with examples X and targets y.
        :param X: (n,d) array containing the data examples.
        :param y: (n) array containing the data targets.
        :param w: parameter at which to compute the objective.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: objective L(f, (X, y)).
        """
        return squared_error(self._forward(X, w), y) / self._scaling(y, scaling)

    def _grad(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute the gradient of the l2 objective with respect to the model parameters.
        :param X: (n,d) array containing the data examples.
        :param y: (n) array containing the data targets.
        :param w: parameter at which to compute the gradient pass.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: the gradient
        """
        network_weights, skip_weights = self._split_weights(w)

        w1, w2 = self.nl_model._split_weights(network_weights)
        Z = relu(X @ w1.T)
        D = lab.sign(Z)

        residuals = X @ skip_weights.T + Z @ w2.T - y

        network_grad = self.nl_model._grad_helper(X, residuals, w1, w2, D, Z)

        skip_grad = residuals.T @ X

        return self._join_weights(network_grad, skip_grad) / self._scaling(y, scaling)
