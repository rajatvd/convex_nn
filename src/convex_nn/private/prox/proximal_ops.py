"""
Proximal operators. This module provides functions for solving minimization problems of the form
    $argmin_x { d(x,w) + beta * g(x) }$,
where d is a metric, g is a "simple" function, and beta is a parameter controlling the trade-off between d and g.

TODO: 
    - Add proximal operator for L2-squared penalty so that we can support this using R-FISTA.

"""
from typing import Optional

import lab


class ProximalOperator:

    """Base class for proximal operators."""

    def __call__(self, w: lab.Tensor, beta: Optional[float] = None) -> lab.Tensor:
        """Evaluate the proximal_operator.
        :param w: parameters to which apply the operator will be applied.
        :param beta: the coefficient in the proximal operator. This is usually a step-size.
        :returns: prox(w)
        """

        raise NotImplementedError("A proximal operator must implement '__call__'!")


class Identity(ProximalOperator):
    def __call__(self, w: lab.Tensor, beta: Optional[float] = None) -> lab.Tensor:
        """The Identity operator which returns w.
        :param w: parameters to which apply the operator will be applied.
        :param beta: (NOT USED) the coefficient in the proximal operator. This is usually a step-size.
        :returns: w, the parameters exactly as they were supplied.
        """

        return w


class Regularizer(ProximalOperator):

    """Base class for proximal operators based on regularizers."""

    def __init__(self, lam: float):
        """
        :param lam: parameter controlling the strength of the regularization.
        """
        self.lam = lam


class L1(Regularizer):

    """The proximal operator for the l1-norm, sometimes known as the
    soft-thresholding operator. In math, this is
        $argmin_x {||x - w||_2^2 + beta * lambda ||x||_1}$.
    """

    def __call__(self, w: lab.Tensor, beta: float) -> lab.Tensor:
        """Compute the proximal operator.
        :param w: parameters to which apply the operator will be applied.
        :param beta: the coefficient in the proximal operator. This is usually a step-size.
        :returns: updated parameters.
        """

        return lab.sign(w) * lab.smax(lab.abs(w) - beta * self.lam, 0.0)


class GroupL1(Regularizer):

    """Compute the proximal operator for the group l1 regularizer, which has the form
    $argmin_x {||x - w||_2^2 + beta * sum_{i=1}^n lambda_i ||x_i||_1}$.
    """

    def __init__(self, lam: float, group_by_feature: bool = False):
        """
        :param lam: parameter controlling the strength of the regularization.
        :param group_by_feature: whether or not weights should be grouped by feature, rather
            than neuron.
        """
        self.lam = lam
        self.group_by_feature = group_by_feature

    def __call__(self, w: lab.Tensor, beta: float) -> lab.Tensor:
        """
        :param w: parameters to which apply the operator will be applied.
        :param beta: the coefficient in the proximal operator. This is usually a step-size.
        :returns: updated parameters.
        """
        if self.group_by_feature:
            w = lab.transpose(w, -1, -2)

        # compute the squared norms of each group.
        norms = lab.sqrt(lab.sum(w ** 2, axis=-1, keepdims=True))

        w_plus = lab.multiply(
            lab.safe_divide(w, norms), lab.smax(norms - self.lam * beta, 0)
        )

        if self.group_by_feature:
            w_plus = lab.transpose(w_plus, -1, -2)

        return w_plus


class Orthant(ProximalOperator):

    """The projection operator for the orthant constrain,
        A_i x >= 0,
    where A_i is a diagonal matrix with (A_i)_jk in {-1, 1}.
    """

    def __init__(self, A: lab.Tensor):
        """
        :param A: a matrix of sign patterns defining orthants on which to project.
            The diagonal A_i is stored as the i'th column of A.
        """
        self.A = A
        if len(A.shape) == 3:
            self.sum_string = "ikj,imjk->imjk"
        else:
            self.sum_string = "kj,imjk->imjk"

    def __call__(self, w: lab.Tensor, beta: Optional[float] = None) -> lab.Tensor:
        """
        :param w: parameters to which the projection will be applied.
            This should be a (k x c x p x d) array, where each element of axis -1 corresponds to one column of A.
        :param beta: NOT USED. The coefficient in the proximal operator. This is usually a step-size.
        :returns: updated parameters.
        """

        return lab.where(
            lab.einsum(self.sum_string, self.A, w) >= 0, w, lab.zeros_like(w)
        )


class GroupL1Orthant(Regularizer):

    """Mixed group L1 penalty with orthant constraint."""

    def __init__(self, d: int, lam: float, A: lab.Tensor):
        """
        :param d: the dimensionality of each group for the regularizer.
        :param lam: the strength of the group-L1 regularizer.
        :param A: a matrix of sign patterns defining orthants on which to project.
            The diagonal A_i is stored as the i'th column of A.
        """
        self.d = d
        self.A = A
        self.lam = lam

        self.group_prox = GroupL1(lam)
        self.orthant_proj = Orthant(A)

    def __call__(self, w: lab.Tensor, beta: float) -> lab.Tensor:
        """
        :param w: parameters to which the projection will be applied.
            This should be a (k x c x p x d + n) array.
            The first 'd' entries in element of axis -1 correspond to the model weights (with group L1 regularizer)
            and the remaining 'n' entries correspond to the slack variables (with orthant constraint).
        :param beta: The coefficient in the proximal operator. This is usually a step-size.
        :returns: updated parameters.
        """
        model_weights, slacks = w[:, :, :, : self.d], w[:, :, :, self.d :]
        w_plus, s_plus = self.group_prox(model_weights, beta), self.orthant_proj(
            slacks, beta
        )

        return lab.concatenate([w_plus, s_plus], axis=-1)
