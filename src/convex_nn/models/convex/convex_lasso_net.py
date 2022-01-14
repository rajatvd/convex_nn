"""
Convex formulation of LassoNet model.
"""

from typing import Optional, Tuple, List

import lab

from convex_nn.models.regularizers import Regularizer
from convex_nn.models.convex import operators
from convex_nn.models.convex.convex_mlp import ConvexMLP
from convex_nn.loss_functions import squared_error


class ConvexLassoNet(ConvexMLP):
    """Convex formulation of a two-layer LassoNet (https://lassonet.ml)."""

    def __init__(
        self,
        d: int,
        D: lab.Tensor,
        U: lab.Tensor,
        kernel: str = operators.EINSUM,
        regularizer: Optional[Regularizer] = None,
        gamma: float = 0.0,
        c: int = 1,
    ) -> None:
        """
        :param d: the dimensionality of the dataset (ie. number of features).
        :param D: array of possible sign patterns.
        :param U: array of hyperplanes creating the sign patterns.
        :param kernel: the kernel to drive the matrix-vector operations.
        :param regularizer: (optional) a penalty function controlling the flexibility of the model.
        :param gamma: (optional) strength of the L1 penalty on the linear skip connections.
        """
        super().__init__(d, D, U, kernel, regularizer, c)

        # include positive and negative skip connections.
        self.weights = lab.zeros((self.c, self.p + 2, self.d))
        self.gamma = gamma

    def set_weights(self, weights: lab.Tensor):
        if weights.shape == (self.c, self.p + 2, self.d):
            self.weights = weights
        elif weights.shape == (self.c, self.p, self.d):
            self.weights[:, :-2] = weights
        elif weights.shape == (self.c, 2, self.d):
            self.weights[:, -2:] = weights
        else:
            raise ValueError(
                f"Weights with shape {weights.shape} cannot be set to ConvexLassoNet with weight shape {(self.p + 2, self.d)}."
            )

    def _split_weights(self, w: lab.Tensor) -> Tuple[lab.Tensor, lab.Tensor]:

        # separate out positive and negative skip weights.
        return w[:, : self.p], w[:, self.p :]

    def _join_weights(self, network_w: lab.Tensor, skip_w: lab.Tensor) -> lab.Tensor:

        return lab.concatenate([network_w, skip_w], axis=1)

    def get_weights(self) -> List[lab.Tensor]:
        """Get model weights in an interpretable format.
        :returns: list of tensors -- [network weights, skip weights].
        """

        network_w, skip_w = self._split_weights(self.weights)

        # combine positive and negative components of skip weights.
        skip_w = skip_w[:, 0] - skip_w[:, 1]

        return [network_w, skip_w]

    def get_reduced_weights(self) -> lab.Tensor:
        return self.weights

    def _forward(
        self,
        X: lab.Tensor,
        w: lab.Tensor,
        D: Optional[lab.Tensor] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute forward pass.
        :param X: (n,d) array containing the data examples.
        :param w: parameter at which to compute the forward pass.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
            Defaults to self.D or manual computation depending on the value of self._train.
        :returns: predictions for X.
        """
        network_w, skip_w = self._split_weights(w.reshape(self.c, self.p + 2, self.d))

        # combine positive and negative components
        skip_w = skip_w[:, 0] - skip_w[:, 1]

        return super()._forward(X, network_w, self._signs(X, D)) + X @ skip_w.T

    def _objective(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        D: Optional[lab.Tensor] = None,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> float:
        """Compute the l2 objective with respect to the model weights *and* the L1 penalty on the skip connections.
        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: parameter at which to compute the objective.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
            Defaults to self.D or manual computation depending on the value of self._train.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: the objective
        """
        w = w.reshape(self.c, self.p + 2, self.d)

        skip_weights_penalty = self.gamma * lab.sum(w[:, -2:])
        return (
            squared_error(self._forward(X, w, D), y) / self._scaling(y, scaling)
            + skip_weights_penalty
        )

    def _grad(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        D: Optional[lab.Tensor] = None,
        flatten: bool = False,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute the gradient of the l2 objective with respect to the model weights.
        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: parameter at which to compute the gradient.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
            Defaults to self.D or manual computation depending on the value of self._train.
        :param flatten: whether or not to flatten the blocks of the gradient into a single vector.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: the gradient
        """
        w = w.reshape(self.c, self.p + 2, self.d)
        D = self._signs(X, D)
        residual = self._forward(X, w, D) - y
        network_grad = lab.einsum("ij, il, ik->ljk", D, residual, X) / self._scaling(
            y, scaling
        )
        skip_grad = X.T @ residual / self._scaling(y, scaling)
        skip_grad = lab.expand_dims(skip_grad.T, axis=1)

        grad = self._join_weights(
            network_grad,
            lab.concatenate([skip_grad, -skip_grad], axis=1)
            + self.gamma,  # apply L1 penalty
        )

        if flatten:
            grad = lab.ravel(grad)

        return grad
