"""
Tests for one-layer ReLU MLPs.
"""

import unittest

import numpy as np
from scipy.optimize import check_grad  # type: ignore
from parameterized import parameterized_class  # type: ignore

import lab

from convex_nn.private.models import sign_patterns, GatedLassoNet
from convex_nn.private.utils.data import gen_regression_data


@parameterized_class(lab.TEST_GRID)
class TestGatedLassoNet(unittest.TestCase):
    """Tests for one-layer LassoNet models with Gated ReLU activations."""

    d: int = 5
    p: int = 3
    n: int = 10
    c: int = 2
    rng: np.random.Generator = np.random.default_rng(778)
    tries: int = 10

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)
        # generate dataset
        train_set, _, self.wopt = gen_regression_data(
            self.rng, self.n, 0, self.d, c=self.c
        )
        self.X, self.y = lab.all_to_tensor(train_set)
        self.wopt = lab.tensor(self.wopt)

        self.U = sign_patterns.sample_gate_vectors(self.rng, self.d, self.p)
        self.D, self.U = sign_patterns.compute_sign_patterns(self.X, self.U)

        # initialize model
        self.gated_ln = GatedLassoNet(self.d, self.U, c=self.c)

        self.objective, self.grad = self.gated_ln.get_closures(self.X, self.y)

        def objective_fn(v):
            return self.objective(lab.tensor(v))

        def grad_fn(v):
            return lab.to_np(self.grad(lab.tensor(v)))

        self.objective_fn = objective_fn
        self.grad_fn = grad_fn

    def test_grad(self):
        """Check that the gradient is computed properly."""
        # test the gradient against finite differences
        for i in range(self.tries):
            v = self.rng.standard_normal(
                self.d * self.p + self.c * self.p + self.d * self.c, dtype=self.dtype
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
