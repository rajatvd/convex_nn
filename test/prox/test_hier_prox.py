"""
Tests for proximal operators.
"""

import unittest

from parameterized import parameterized_class  # type: ignore
import lab

from convex_nn.prox import HierProx


@parameterized_class(lab.TEST_GRID)
class TestHierProx(unittest.TestCase):
    """Test proximal operators."""

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)

    def check_kkt_conditions(self, weights_plus, slacks):
        # sanity check
        weights_plus = lab.squeeze(weights_plus)
        self.assertTrue(
            lab.all(weights_plus[-2:] >= 0.0), "Skip weights were not non-negative!"
        )

        # compute beta_1 from the slack variables:
        beta_1 = weights_plus[-2] + weights_plus[-1]

        # KKT Conditions: note that we cannot check stationarity of the Lagrangian since we infer
        # 'slacks' from this condition. These checks do not guarantee correctness but can catch some errors.

        # positivity
        self.assertTrue(
            lab.all(beta_1 >= 0.0), "Primal variables were not non-negative!"
        )
        # positivity: we allow slacks to be very slightly negative due to rounding errors.
        self.assertTrue(lab.all(slacks >= -1e-8), "Slacks were not non-negative!")

        # complementary slackness
        self.assertTrue(
            lab.allclose(slacks * beta_1, lab.tensor(0.0)),
            "Complementary slackness failed!",
        )

    def test_interior_point(self):
        """Check that projecting an interior point onto the constraint set is the identity map."""
        matrix = lab.tensor(
            [
                [
                    # network weights
                    [1.0, 10, 100, 1000],
                    [1, 10, 100, 1000],
                    [1, 10, 100, 1000],
                    # skip connections
                    [1, 2, 3, 4],
                    [0, 1, 1, 1],
                ]
            ]
        )

        prox = HierProx(M=1000)

        # the proximal operator should be the identity map when M -> infty and skip weights are positive.
        res, slacks = prox(matrix, return_slacks=True)

        self.assertTrue(
            lab.allclose(res, matrix),
            "Prox with M -> infty and positive skip weights didn't reduce to the identity map.",
        )

        self.check_kkt_conditions(res, slacks)

    def test_negative_skip_weights(self):
        """Test proximal operator with an exterior point with negative skip weights."""
        matrix = lab.tensor(
            [
                [
                    # network weights
                    [1.0, 10, 100, 1000],
                    [1, 10, 100, 1000],
                    [1, 10, 100, 1000],
                    # skip connections
                    [-1, 2, -3, 4],
                    [0, 1, 1, 1],
                ]
            ]
        )

        prox = HierProx(M=1000)

        # the proximal operator should project the skip weights onto the non-negative orthant.
        res, slacks = prox(matrix, return_slacks=True)

        self.assertTrue(
            lab.allclose(res[0, -2:, [1, 3]], matrix[0, -2:, [1, 3]]),
            "Prox with M -> infty shouldn't change positive skip weights.",
        )

        self.assertTrue(
            lab.allclose(res[0, 0:3], matrix[0, 0:3]),
            "Prox with M -> infty did not reduce to the identity map for the network weights.",
        )

        self.check_kkt_conditions(res, slacks)

    def test_thresholding(self):
        """Test proximal operator with an exterior point leading to feature sparsity."""
        matrix = lab.tensor(
            [
                [
                    # network weights
                    [1.0, 10, -100, 1000],
                    [3, -10, -100, 2000],
                    [2, 20, -100, 1000],
                    # skip connections
                    [-10, 2, -3, 4],
                    [-10, 1, 1, 1],
                ]
            ]
        )

        prox = HierProx(M=1)

        # the proximal operator should project the skip weights onto the non-negative orthant.
        res, slacks = prox(matrix, return_slacks=True)

        self.assertTrue(
            lab.all(res[0, :, 0] == 0.0),
            "The first column of network and skip weights was not zeroed.",
        )

        self.check_kkt_conditions(res, slacks)


if __name__ == "__main__":
    unittest.main()
