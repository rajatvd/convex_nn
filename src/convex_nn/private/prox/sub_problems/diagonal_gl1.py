"""Sub-problem for evaluating the proximal operator associated with the group
L1 regularizer with a diagonal preconditioner."""
from typing import Optional, Tuple, Callable

import lab


class GL1DiagonalSubproblem:

    """Model for the root-finding problem that must be solved when computing
    the proximal operator for a group L1 regularizer with diagonal
    preconditioning."""

    def __init__(self, w: lab.Tensor, H: lab.Tensor):
        """
        :param w: the initial weights of the model. This is only required for
            to initialize the weight of the root-finding problem and any
            matrix with the correct shape is acceptable.
        :param H: a diagonal preconditioner matrix.
        """
        # preconditioner should be represented as a matrix of column vectors.
        assert len(H.shape) == 2

        self.base_H = H
        self.H = self.base_H

        if len(w.shape) == 2:
            self.alpha = lab.ones((self.base_H.shape[0]))
            self.sum_axis = 1
        else:
            self.sum_axis = 2
            self.alpha = lab.ones((2, self.base_H.shape[0]))

    def subset(self, indices: lab.Tensor):
        self.H = self.base_H[indices]

    def _alpha(self, alpha: Optional[lab.Tensor] = None) -> lab.Tensor:
        return self.alpha if alpha is None else alpha

    def get_alpha(self, indices: lab.Tensor) -> lab.Tensor:
        if self.sum_axis == 1:
            return self.alpha[indices]
        else:
            return self.alpha[:, indices]

    def set_alpha(self, alpha: lab.Tensor, indices: lab.Tensor):
        if self.sum_axis == 1:
            self.alpha[indices] = alpha
        else:
            self.alpha[:, indices] = alpha

    def w_alpha(self, x: lab.Tensor, rho: float, alpha: Optional[lab.Tensor] = None):
        """"""
        alpha = self._alpha(alpha)
        alpha_v = lab.expand_dims(alpha, axis=-1)
        return alpha_v * x / (self.H * alpha_v + rho)

    def objective(
        self,
        x: lab.Tensor,
        rho: float,
        alpha: Optional[lab.Tensor] = None,
        **kwargs,
    ):
        """"""
        alpha = self._alpha(alpha)
        x_alpha = x / (self.H * lab.expand_dims(alpha, axis=-1) + rho)
        return (lab.sum(x_alpha ** 2, axis=self.sum_axis) - 1) / 2

    def grad(
        self,
        x: lab.Tensor,
        rho: float,
        alpha: Optional[lab.Tensor] = None,
        **kwargs,
    ):
        """"""
        alpha = self._alpha(alpha)
        alpha_v = lab.expand_dims(alpha, axis=-1)
        precon = (self.H * alpha_v + rho)
        x_alpha = x / precon
        z_alpha = x_alpha / alpha_v
        z_alpha_prime = -z_alpha * (self.H / precon)

        return lab.sum(z_alpha_prime * x_alpha, axis=self.sum_axis)

    def get_closures(
        self,
        x: lab.Tensor,
        rho: float,
    ) -> Tuple[Callable, Callable]:
        """Returns closures for computing the objective, gradient, and Hessian given (X, y).
            Warning: this closure will retain references to X, y and so can prevent garbage collection of
            these objects.
        :param X: (n,d) array containing the data examples.
        :param y: (n,d) array containing the data targets.
        :param hessian_as_op: return the Hessian as a LinearOperator rather than matrix.
        :returns: (objective_fn, grad_fn, hessian_fn).
        """

        def objective_fn(alpha: Optional[lab.Tensor] = None, **kwargs):
            return self.objective(x, rho, alpha=alpha, **kwargs)

        def grad_fn(alpha: Optional[lab.Tensor] = None, **kwargs):
            return self.grad(x, rho, alpha=alpha, **kwargs)

        return objective_fn, grad_fn
