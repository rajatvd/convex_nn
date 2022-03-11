"""
Tests for convex neural networks.
"""

import unittest

import numpy as np
from parameterized import parameterized_class  # type: ignore
from scipy.optimize import check_grad  # type: ignore

import lab

from convex_nn import activations
from convex_nn.private.models import ConvexLassoNet
from convex_nn.private.utils.data import gen_regression_data


@parameterized_class(lab.TEST_GRID)
class TestConvexLassoNet(unittest.TestCase):
    """Test convex formulation of two-layer LassoNet model."""

    d: int = 2
    n: int = 3
    c: int = 2
    rng: np.random.Generator = np.random.default_rng(778)

    tries: int = 10

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)

        train_set, _, _ = gen_regression_data(
            self.rng,
            self.n,
            0,
            self.d,
            c=self.c,
        )
        self.U = activations.sample_gate_vectors(self.rng, self.d, 100)
        self.D, self.U = lab.all_to_tensor(
            activations.compute_activation_patterns(train_set[0], self.U)
        )
        self.X, self.y = lab.all_to_tensor(train_set)

        self.p = self.D.shape[1]

        # instantiate models with all available kernels.
        self.lasso_net = ConvexLassoNet(
            self.d, self.D, self.U, kernel="einsum", c=self.c
        )

        self.objective, self.grad = self.lasso_net.get_closures(self.X, self.y)

        def objective_fn(v):
            return self.objective(lab.tensor(v))

        def grad_fn(v):
            return lab.to_np(self.grad(lab.tensor(v), flatten=True))

        self.objective_fn = objective_fn
        self.grad_fn = grad_fn

    def test_grad(self):
        """Check that the gradient is computed properly."""

        # test the gradient against finite differences
        for i in range(self.tries):
            v = self.rng.standard_normal(
                self.c * self.d * (self.p + 2), dtype=self.dtype
            )

            self.assertTrue(
                np.isclose(
                    check_grad(self.objective_fn, self.grad_fn, v),
                    0,
                    rtol=1e-5,
                    atol=1e-5,
                ),
                "Gradient does not match finite differences approximation.",
            )


if __name__ == "__main__":
    unittest.main()
