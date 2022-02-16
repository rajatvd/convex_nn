"""
Implementation of linear regression with the least-squares objective.
"""
from typing import Optional

import lab

from convex_nn.private.models.model import Model
from convex_nn.private.models.regularizers import Regularizer
import convex_nn.private.loss_functions as loss_fns


class L2Regression(Model):

    """Linear regression with least-squares (l2) objective."""

    def __init__(self, d: int, regularizer: Optional[Regularizer] = None):
        """
        :param d: the dimensionality of the dataset (ie.number of features).
        :param regularizer: (optional) a penalty function controlling the flexibility of the model.
        """
        super().__init__(regularizer)
        self.d = d
        self.weights = lab.zeros(d)

    def _forward(self, X: lab.Tensor, w: lab.Tensor, **kwargs) -> lab.Tensor:
        """Compute forward pass.
        :param X: (n,d) array containing the data examples.
        :param w: parameter at which to compute the forward pass.
        """
        return X @ w

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
        :param y: (n,d) array containing the data targets.
        :param w: specific parameter at which to compute the forward pass.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: objective L(f, (X, y)).
        """
        return loss_fns.squared_error(self._forward(X, w), y) / self._scaling(
            y, scaling
        )

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
        :param y: (n,d) array containing the data targets.
        :param w: parameter at which to compute the forward pass.
        :param scaling: (optional) scaling parameter for the objective. Defaults to `n * c`.
        :returns: the gradient
        """

        res = self._forward(X, w) - y
        return lab.matmul(X.T, res) / self._scaling(y, scaling)
