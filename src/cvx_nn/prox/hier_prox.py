"""
Proximal operators for LassoNet models.
"""

from typing import Optional, Union, Tuple

import lab

from cvx_nn.prox.proximal_ops import ProximalOperator


class HierProx(ProximalOperator):
    """Implementation of the hierarchical proximal operator *without* group L1 regularization."""

    def __init__(self, M: float):
        """
        :param M: parameter for the magnitude constraints on the network weights.
            Note that M > 0 is required; taking M = 0 will result in undefined behavior.
            It is possible to simplify the operator's computation for M = 1. This is future work.
        """

        self.M = M

    def __call__(
        self,
        w: lab.Tensor,
        beta: Optional[float] = None,
        return_slacks: bool = False,
    ) -> Union[lab.Tensor, Tuple[lab.Tensor, lab.Tensor]]:
        """Compute the proximal operator.
        :param w: parameters to which apply the operator will be applied.
        :param beta: (NOT USED) the coefficient in the proximal operator.
            Note that it has no effect for projections such as this operator.
        :param return_slacks: whether or not to also compute and return slack variables
            for the associated dual problem.
        :returns: (weights) or (weights, slacks) if 'return_slacks' is True.
        """

        if len(w.shape) == 4 and w.shape[1] > 1 or len(w.shape) == 3 and w.shape[0] > 1:
            raise ValueError("HierProx only supports scalar targets!")

        c_axis = 0
        if len(w.shape) == 4:
            c_axis = 1

        w = lab.squeeze(w)

        # split off skip weights
        if len(w.shape) == 3:
            split_network_w, skip_w = w[:, :-1], w[:, -1:]

            network_w = split_network_w[0] - split_network_w[1]
            skip_w = lab.squeeze(skip_w)
        else:
            network_w, skip_w = w[:-2], w[-2:]

        # reparameterize problem
        skip_sum, skip_diff = skip_w[0] + skip_w[1], skip_w[0] - skip_w[1]

        # solve proximal operator
        network_plus, skip_plus, slacks = self._prox_helper(
            network_w, skip_sum, skip_diff, return_slacks
        )

        # undo reparameterization
        if len(w.shape) == 3:
            network_z = split_network_w[0] + split_network_w[1]
            network_plus = 0.5 * lab.stack(
                [network_plus + network_z, network_z - network_plus]
            )

            weights = lab.concatenate(
                [network_plus, lab.expand_dims(skip_plus, axis=1)], axis=1
            )
        else:
            weights = lab.concatenate([network_plus, skip_plus])

        weights = lab.expand_dims(weights, axis=c_axis)

        if return_slacks:
            return weights, slacks

        return weights

    def _prox_helper(
        self,
        network_w: lab.Tensor,
        skip_sum: lab.Tensor,
        skip_diff: lab.Tensor,
        return_slacks: bool = False,
    ) -> Tuple[lab.Tensor, lab.Tensor, lab.Tensor]:

        p, d = network_w.shape
        abs_skip_diff = lab.abs(skip_diff)
        abs_network_w = lab.abs(network_w).T

        # compute non-smooth "breakpoints" of test function.
        breakpoints = lab.sort(
            lab.concatenate(
                [
                    lab.zeros((d, 1)),
                    abs_network_w / self.M,
                    lab.expand_dims(abs_skip_diff, axis=-1),
                ],
                axis=-1,
            ),
            axis=-1,
        )

        # evaluate test function at breakpoints
        f_vals = (
            breakpoints
            - lab.expand_dims(skip_sum, axis=-1)
            - lab.smax(lab.expand_dims(abs_skip_diff, axis=-1) - breakpoints, 0)
            - self.M
            * lab.sum(
                lab.smax(
                    lab.expand_dims(abs_network_w.T, axis=2) - self.M * breakpoints, 0
                ),
                axis=0,
            )
        )

        # compute signs of test function
        f_signs = lab.concatenate([lab.sign(f_vals), lab.ones((d, 1))], axis=-1)
        # zeros can be treated as positive or negative points.
        f_signs[f_signs == 0.0] = -1

        # find breakpoints where the sign transition *starts*.
        zero_indices = lab.argmin(lab.abs(f_signs[:, :-1] + f_signs[:, 1:]), axis=1)
        row_indices = lab.arange(len(zero_indices))
        selected_points = breakpoints[row_indices, zero_indices]

        network_w_mask = abs_network_w > self.M * lab.expand_dims(
            selected_points, axis=1
        )
        skip_diff_mask = abs_skip_diff > selected_points

        # compute number of non-zero elements in the sum.
        n_vals = lab.sum(network_w_mask, axis=-1)

        network_w_sums = lab.sum(
            abs_network_w * network_w_mask,
            axis=-1,
        )

        beta_1 = (
            skip_sum
            + lab.multiply(abs_skip_diff, skip_diff_mask)
            + self.M * network_w_sums
        ) / (1 + (self.M ** 2) * n_vals + skip_diff_mask)

        # threshold features when f(0) >= 0. Note that beta_1 as computed above is
        # not well-defined for these entries.
        beta_1[f_vals[:, 0] >= 0.0] = 0.0

        # recover beta_2 and updated network weights.
        beta_2 = lab.sign(skip_diff) * lab.minimum(abs_skip_diff, beta_1)
        network_plus = lab.sign(network_w) * lab.minimum(
            abs_network_w.T, self.M * beta_1
        )
        # recover skip weights by undoing change-of-variables.
        skip_plus = 0.5 * lab.stack([(beta_1 + beta_2), (beta_1 - beta_2)])

        # Compute slack variables for debugging/testing.
        slacks = 0
        if return_slacks:
            slacks = (
                beta_1
                - skip_sum
                - lab.smax(abs_skip_diff - beta_1, 0)
                - self.M
                * lab.sum(
                    lab.smax(
                        abs_network_w - self.M * lab.expand_dims(beta_1, axis=-1), 0
                    ),
                    axis=-1,
                )
            )

        return network_plus, skip_plus, slacks
