"""
Group L1 Regularizer.
"""
from typing import Optional

import lab

from cvx_nn.loss_functions import group_l1_penalty
from cvx_nn.models.regularizers.regularizer import Regularizer


class GroupL1Regularizer(Regularizer):

    """The Group L1-regularizer, which has the mathematical form
        $f(w) = sum_i lambda_i ||w_i||_2$,
    where {w_i : i [p]} are the parameter groups.
    """

    def __init__(self, lam: float, group_by_feature: bool = False):
        """
        :param base_model: the Model instance to regularize.
        :param lam: the tuning parameter controlling the strength of regularization.
        :param group_by_feature: whether or not weights should be grouped by feature, rather
            than neuron.
        """

        self.lam = lam
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

        return group_l1_penalty(w, self.lam)

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

        weight_norms = lab.sqrt(lab.sum(w ** 2, axis=-1, keepdims=True))
        grad_norms = lab.sqrt(lab.sum(base_grad ** 2, axis=-1, keepdims=True))

        non_smooth_term = (
            base_grad * lab.smin(self.lam / grad_norms, 1) * (weight_norms == 0)
        )
        smooth_term = self.lam * lab.safe_divide(w, weight_norms)

        # match input shape
        subgrad = smooth_term - non_smooth_term

        if self.group_by_feature:
            subgrad = lab.transpose(subgrad, -1, -2)

        return subgrad
