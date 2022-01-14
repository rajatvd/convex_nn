"""
Augmented Lagrangian for LassoNet.
"""
from typing import Tuple, Optional, Union, Any, List

import lab

from .al_mlp import AL_MLP
from convex_nn.models.regularizers import Regularizer
from convex_nn.models.convex import operators
from convex_nn.loss_functions import squared_error


class AL_LassoNet(AL_MLP):

    """Base class for augmented Lagrangian of the convex formulation of a LassoNet."""

    def __init__(
        self,
        d: int,
        D: lab.Tensor,
        U: lab.Tensor,
        kernel: str = operators.EINSUM,
        delta: float = 100.0,
        regularizer: Optional[Regularizer] = None,
        c: int = 1,
        gamma: float = 0.0,
    ) -> None:
        """
        :param d: the dimensionality of the dataset (ie. number of features).
        :param D: array of possible sign patterns.
        :param U: array of hyperplanes creating the sign patterns.
        :param kernel: the kernel to drive the matrix-vector operations.
        :param delta: parameter controlling the strength of the quadratic penalty
        in the augmented Lagrangian.
        :param regularizer: (optional) a penalty function controlling the flexibility of the model.
        :param lam: (optional) strength of the L1 penalty on the linear skip connections.
        """

        super().__init__(d, D, U, kernel, regularizer=regularizer, c=c)
        self.n = self.D.shape[0]

        self.weights = lab.zeros((2, self.c, self.p + 1, self.d))

        # estimates of the optimal dual variables
        self.e_multipliers = lab.tensor([0.0])  # NOT USED.
        self.i_multipliers = lab.zeros((2, self.c, self.p, self.n))

        self.delta = delta
        self.orthant = 2 * self.D - lab.ones_like(self.D)

        self.gamma = gamma

    def set_weights(self, weights: lab.Tensor):
        if weights.shape == (2, self.c, self.p + 1, self.d):
            self.weights = weights
        elif weights.shape == (2, self.c, self.p, self.d):
            self.weights[:, :, :-1] = weights
        elif weights.shape == (2, self.c, 1, self.d):
            self.weights[:, :, :-1] = weights
        elif weights.shape == (self.c, self.p, self.d):
            # "split" the weights and assign to the model
            self.weights[:, :, :-1, :] = lab.stack([weights / 2, -weights / 2])
        else:
            raise ValueError(
                f"Weights with shape {weights.shape} cannot be set to IneqLassoNet with weight shape {(2, self.p + 1, self.d)}."
            )

    def _split_weights(self, w: lab.Tensor) -> Tuple[lab.Tensor, lab.Tensor]:

        # separate out positive and negative skip weights.
        return w[:, :, : self.p], w[:, :, self.p]

    def _join_weights(self, network_w: lab.Tensor, skip_w: lab.Tensor) -> lab.Tensor:

        return lab.concatenate([network_w, skip_w], axis=2)

    def update_model_weights(self, w: lab.Tensor):
        self.weights = w

    def get_reduced_weights(self) -> lab.Tensor:
        network_w, skip_w = self._split_weights(self.weights)

        reduced = lab.concatenate(
            [network_w[0] - network_w[1], skip_w.reshape(self.c, 2, self.d)], axis=1
        )
        return reduced

    def get_weights(self) -> List[lab.Tensor]:
        """Get model weights in an interpretable format.
        :returns: list of tensors -- [network weights, skip weights].
        """

        network_w, skip_w = self._split_weights(self.weights)

        # combine positive and negative components of skip weights.
        skip_w = skip_w[0] - skip_w[1]

        return [network_w, skip_w]

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
        network_w, skip_w = self._split_weights(w)

        return super()._forward(X, network_w, D) + X @ (skip_w[0] - skip_w[1]).T

    def _objective(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        D: Optional[lab.Tensor] = None,
        index_range: Optional[Tuple[int, int]] = None,
        scaling: Optional[float] = None,
        ignore_lagrange_penalty: bool = False,
        **kwargs,
    ) -> float:
        """Compute the l2 objective with respect to the model weights.
        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: specific parameter at which to compute the objective.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
        Defaults to self.D or manual computation depending on the value of self._train.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: the objective
        """
        network_w, skip_w = self._split_weights(
            w.reshape(2, self.c, self.p + 1, self.d)
        )
        skip_weights_penalty = self.gamma * lab.sum(skip_w)

        obj = (
            squared_error(self._forward(X, w, D), y) / self._scaling(y, scaling)
            + skip_weights_penalty
        )

        if not ignore_lagrange_penalty:
            obj += self.lagrange_penalty_objective(X, network_w, index_range, scaling)

        return obj

    def _grad(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        D: Optional[lab.Tensor] = None,
        index_range: Optional[Tuple[int, int]] = None,
        scaling: Optional[float] = None,
        flatten: bool = False,
        **kwargs,
    ) -> lab.Tensor:
        """Compute the gradient of augmented Lagrangian objective with respect to the model weights.
        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param w: parameter at which to compute the gradient.
        :param D: (optional) specific activation matrix at which to compute the forward pass.
        Defaults to self.D or manual computation depending on the value of self._train.
        :param flatten: whether or not to flatten the blocks of the gradient into a single vector.
        :returns: the gradient
        """
        w = w.reshape(2, self.c, self.p + 1, self.d)
        network_w, _ = self._split_weights(w)
        D = self._signs(X, D)
        residual = self._forward(X, w, D) - y
        network_grad = lab.einsum("ij, il, ik->ljk", D, residual, X) / self._scaling(
            y, scaling
        )
        skip_grad = X.T @ residual / self._scaling(y, scaling)
        skip_grad = lab.expand_dims(skip_grad.T, axis=1)

        split_network_grad = lab.stack([network_grad, -network_grad])
        split_network_grad += self.lagrange_penalty_grad(
            X, network_w, D=D, index_range=index_range, scaling=scaling
        )

        split_skip_grad = lab.stack([skip_grad, -skip_grad])
        # apply L1 penalty
        split_skip_grad += self.gamma

        grad = lab.concatenate([split_network_grad, split_skip_grad], axis=2)

        if flatten:
            return lab.ravel(grad)

        return grad

    def lagrangian_grad(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: Optional[lab.Tensor] = None,
        D: Optional[lab.Tensor] = None,
        index_range: Optional[Tuple[int, int]] = None,
        scaling: Optional[float] = None,
        flatten: bool = False,
    ) -> lab.Tensor:

        network_w, _ = self._split_weights(self._weights(w))
        return super().lagrangian_grad(
            X, y, network_w, D, index_range, scaling, flatten
        )

    def i_constraint_gap(
        self,
        X: lab.Tensor,
        w: Optional[lab.Tensor] = None,
        index_range: Optional[Tuple[int, int]] = None,
    ):
        """Compute violation of the linear constraints
        .. math::

            X v_i - a_i

            X w_i - b_i,

        where a_i, b_i are the slack variables.
        :param X: (n,d) array containing the data examples.
        :param w: (optional) specific parameter at which to compute the gradient.
        Defaults to 'None', in which case the current model state is used.
        :returns: the constraint gaps.
        """
        if w is None or w.shape[2] == self.p + 1:
            w, _ = self._split_weights(
                self._weights(w).reshape(2, self.c, self.p + 1, self.d)
            )

        return super().i_constraint_gap(X, w, index_range)
