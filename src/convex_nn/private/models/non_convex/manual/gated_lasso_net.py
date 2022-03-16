"""Implementation of one-layer LassoNet with Gated ReLU activations."""

from typing import Optional, Tuple

import lab

from convex_nn.private.models.model import Model
from .gated_relu_mlp import GatedReLUMLP
from .relu_lasso_net import ReLULassoNet
from convex_nn.private.models.regularizers import Regularizer
from convex_nn.private.loss_functions import squared_error, relu


class GatedLassoNet(ReLULassoNet):

    """One-layer Gated ReLU LassoNet with squared-error objective."""

    def __init__(
        self,
        d: int,
        U: lab.Tensor,
        regularizer: Optional[Regularizer] = None,
        c: int = 1,
    ):
        """
        :param d: the dimensionality of the dataset (ie.number of features).
        :param U: the gate vectors associated with gated ReLU activations.
        :param regularizer: (optional) a penalty function controlling the flexibility of the model.
        :param c: (optional) the number of targets.
        """
        super().__init__(d, U.shape[1], regularizer, c)
        self.U = U

        self.nl_model = GatedReLUMLP(d, U, None, c)
        # free memory of sub-model.
        self.nl_model.weights = None

    def _grad(
        self,
        X: lab.Tensor,
        y: lab.Tensor,
        w: lab.Tensor,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute the gradient of the l2 objective with respect to the model
        parameters.

        :param X: (n,d) array containing the data examples.
        :param y: (n) array containing the data targets.
        :param w: parameter at which to compute the gradient pass.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: the gradient
        """
        network_weights, skip_weights = self._split_weights(w)

        w1, w2 = self.nl_model._split_weights(network_weights)
        D = lab.sign(relu(X @ self.U))
        Z = lab.multiply(D, X @ w1.T)
        residuals = X @ skip_weights.T + Z @ w2.T - y

        network_grad = self.nl_model._grad_helper(X, residuals, w1, w2, D, Z)

        skip_grad = residuals.T @ X

        return self._join_weights(network_grad, skip_grad) / self._scaling(y, scaling)
