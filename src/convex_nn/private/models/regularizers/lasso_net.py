"""
Orthant constraint.
"""
from convex_nn.private.models.regularizers.constraint import Constraint

from convex_nn.private.prox import HierProx


class LassoNetConstraint(Constraint):

    """Representation of the convex LassoNet constraint,
       |W_ij| <= M * (beta_plus + beta_minus),
    where W is the weight matrix for the convex neural network, (beta_plus, beta_minus)
    are the weights for the (split) skip connections, and M is a tuning parameter.
    """

    def __init__(self, M: float):
        """
        :param M: a tuning parameter controlling the "strength" of the constraint.
        """
        self.M = M
        self.projection_op = HierProx(M)
