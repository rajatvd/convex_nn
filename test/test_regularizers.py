"""End-to-end tests for regularizers.

TODO:
    - Test solving ReLU and Gated ReLU problems with different regularizers and with appropriate solvers.
    - Should we solve both the ReLU and Gated ReLU problems or just the ReLU?
    - L2-squared and L1.
"""

import unittest

from convex_nn.private.utils.data.synthetic import gen_regression_data
from convex_nn.regularizers import NeuronGL1, FeatureGL1, L2
from convex_nn.optimize import optimize


class TestRegularizers(unittest.TestCase):
    """Test solving convex reformulations with different regularizers."""

    def setUp(self):
        # Generate realizable synthetic classification problem (ie. Figure 1)
        n_train = 100
        n_test = 100
        d = 25
        c = 2
        kappa = 1
        self.max_neurons = 100

        (
            (self.X_train, self.y_train),
            (self.X_test, self.y_test),
            _,
        ) = gen_regression_data(
            123, n_train, n_test, d, c, kappa=kappa, unitize_data_cols=False
        )
        self.lam = 0.0001

    def test_l2_squared(self):
        """"""

    def test_l1(self):
        """"""

    def test_neuron_gl1(self):
        """"""
        cvx_model, metrics = optimize(
            "gated_relu",
            self.max_neurons,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            NeuronGL1(self.lam),
        )

    def test_feature_gl1(self):
        """"""
        cvx_model, metrics = optimize(
            "gated_relu",
            self.max_neurons,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            FeatureGL1(self.lam),
        )


if __name__ == "__main__":
    unittest.main()
