"""Group L1 Regularizer."""
from typing import Optional

import lab

from convex_nn.private.loss_functions import group_l1_penalty
from convex_nn.private.models.regularizers.regularizer import Regularizer


class DiagonalGL1Regularizer(Regularizer):

    """The Group L1-regularizer for with diagonal weighting.

        $f(w) = sum_i lambda_i ||w_i||_2$,

    where {w_i : i [p]} are the parameter groups.
    """

    def __init__(
        self, lam: float, A: lab.Tensor, group_by_feature: bool = False
    ):
        """
        :param base_model: the Model instance to regularize.
        :param lam: the tuning parameter controlling the strength of regularization.
        :param group_by_feature: whether or not weights should be grouped by feature, rather
            than neuron.
        """

        self.lam = lam
        self.A = A
        self.group_by_feature = group_by_feature

    def penalty(
        self,
        w: lab.Tensor,
        **kwargs,
    ) -> float:
        """Compute the penalty associated with the regularizer.

        :param w: parameter at which to compute the penalty.
        :returns: penalty value
        """
        if self.group_by_feature:
            w = lab.transpose(w, -1, -2)

        z = lab.multiply(self.A, w)

        return self.lam * lab.sum(
            lab.sqrt(lab.sum(lab.multiply(w, z), axis=-1))
        )

    def grad(
        self,
        w: lab.Tensor,
        base_grad: Optional[lab.Tensor] = None,
        **kwargs,
    ) -> lab.Tensor:
        """Compute the minimum-norm subgradient (aka, the pseudo-gradient).

        :param w: parameter at which to compute the penalty gradient.
        :param base_grad: the gradient of the un-regularized objective. This is required
            to compute the minimum-norm subgradient.
        :returns: minimum-norm subgradient.
        """
        # requires base_grad to compute minimum-norm subgradient
        assert base_grad is not None

        if self.group_by_feature:
            w = lab.transpose(w, -1, -2)
            base_grad = lab.transpose(base_grad, -1, -2)

        z = lab.multiply(self.A, w)
        weight_norms = lab.sqrt(lab.sum(z * w, axis=-1, keepdims=True))

        A_inv_grad = lab.divide(base_grad, self.A)
        grad_norms = lab.sqrt(
            lab.sum(base_grad * A_inv_grad, axis=-1, keepdims=True)
        )

        non_smooth_term = (
            base_grad
            * lab.smin(self.lam / grad_norms, 1)
            * (weight_norms == 0)
        )
        smooth_term = self.lam * lab.safe_divide(z, weight_norms)

        # match input shape
        subgrad = smooth_term - non_smooth_term

        if self.group_by_feature:
            subgrad = lab.transpose(subgrad, -1, -2)

        return subgrad
