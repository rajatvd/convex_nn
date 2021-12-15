"""
Tests for least-squares regression.
"""

import unittest

import numpy as np
from scipy.optimize import check_grad  # type: ignore
from parameterized import parameterized_class  # type: ignore

import lab

from cvx_nn.models import L2Regression
from cvx_nn.utils.linear import direct_solvers
from cvx_nn import datasets


@parameterized_class(lab.TEST_GRID)
class TestL2Regression(unittest.TestCase):
    """Tests for least squares regression."""

    d: int = 5
    n: int = 10
    rng: np.random.Generator = np.random.default_rng(778)
    tries: int = 10

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)
        # generate dataset
        (self.X, self.y), _, self.wopt = datasets.generate_synthetic_regression(
            self.rng, self.n, 0, self.d
        )

        # initialize model
        self.lr = L2Regression(self.d)

        self.objective, self.grad = self.lr.get_closures(self.X, self.y)

        def objective_fn(v):
            return self.objective(lab.tensor(v))

        def grad_fn(v):
            return lab.to_np(self.grad(lab.tensor(v)))

        self.objective_fn = objective_fn
        self.grad_fn = grad_fn

    def test_grad(self):
        """Check that the gradient is computed properly."""
        # test the gradient against brute computation
        random_weights = lab.tensor(self.rng.standard_normal(self.d, dtype=self.dtype))
        self.lr.weights = random_weights

        model_grad = self.lr.grad(self.X, self.y)
        manual_grad = (self.X.T @ ((self.X @ random_weights) - self.y)) / len(self.y)

        self.assertTrue(
            np.allclose(model_grad, manual_grad),
            "Gradient does not match brute computation.",
        )

        # check that the gradient is zero after the normal equations.
        wopt = direct_solvers.solve_ne(self.X, self.y, lam=0)
        grad = self.lr.grad(self.X, self.y, wopt)
        self.assertTrue(lab.allclose(grad, lab.zeros_like(grad)))

        # test the gradient against finite differences
        for i in range(self.tries):
            v = self.rng.standard_normal(self.d, dtype=self.dtype)

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
