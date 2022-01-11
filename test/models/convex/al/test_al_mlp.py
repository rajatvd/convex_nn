"""
Tests for convex formulation of two-layer MLP augmented with slack variables to support orthant constraints.
"""

import unittest

import numpy as np
from scipy.optimize import check_grad  # type: ignore
from parameterized import parameterized_class  # type: ignore

import lab

from cvx_nn.models import sign_patterns, AL_MLP
from cvx_nn.models.convex import operators
from cvx_nn import datasets


@parameterized_class(lab.TEST_GRID)
class TestIneqAugmentedConvexMLP(unittest.TestCase):
    """Test convex formulation of two-layer ReLU network with inequality constraints."""

    d: int = 2
    n: int = 4
    c: int = 5
    rng: np.random.Generator = np.random.default_rng(778)

    tries: int = 10

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)

        (self.X, self.y), _, _ = datasets.generate_synthetic_regression(
            self.rng, self.n, 0, self.d, c=self.c, vector_output=True
        )

        self.D, self.U = sign_patterns.approximate_sign_patterns(
            self.rng, self.X, n_samples=100
        )
        self.P = self.D.shape[1]

        self.nn = AL_MLP(self.d, self.D, self.U, delta=2, c=self.c)

    def test_forward(self):
        """Test network predictions."""

        weights = lab.tensor(
            self.rng.standard_normal((2, self.c, self.d, self.P), dtype=self.dtype)
        )
        self.nn.weights = weights

        expanded_X = operators.expanded_data_matrix(self.X, self.D)
        network_preds = self.nn(self.X)
        direct_preds = lab.matmul(
            expanded_X,
            weights[0].reshape(self.c, -1).T - weights[1].reshape(self.c, -1).T,
        )

        self.assertTrue(
            lab.allclose(network_preds, direct_preds),
            "Network predictions did not match direct calculation.",
        )

    def test_lagrange_penalty(self):
        """Test implementation of penalty functions + their gradients for the augmented Lagrangian."""

        def obj_fn(w):
            return self.nn.lagrange_penalty_objective(self.X, lab.tensor(w))

        def grad_fn(w):
            return lab.to_np(
                self.nn.lagrange_penalty_grad(self.X, lab.tensor(w), flatten=True)
            )

        for tr in range(self.tries):
            weights = self.rng.standard_normal(
                (2, self.c, self.P, self.d), dtype=self.dtype
            )

            self.assertTrue(
                np.allclose(
                    check_grad(obj_fn, grad_fn, weights.reshape(-1)),
                    0.0,
                    atol=1e-4,
                ),
                "The gradient of the penalty functions do not match the finite-difference approximation.",
            )

    def test_weights_obj_grad(self):
        """Test implementation of objective and gradient for the augmented Lagrangian."""

        def obj_fn(w):
            return self.nn.objective(
                self.X, self.y, lab.tensor(w).reshape(2, self.c, self.P, self.d)
            )

        def grad_fn(w):
            return lab.to_np(
                self.nn.grad(
                    self.X,
                    self.y,
                    lab.tensor(w),
                    flatten=True,
                )
            )

        for tr in range(self.tries):
            weights = self.rng.standard_normal(
                (2, self.c, self.P, self.d), dtype=self.dtype
            )

            self.assertTrue(
                np.allclose(
                    check_grad(obj_fn, grad_fn, weights.reshape(-1)),
                    0.0,
                    atol=1e-4,
                ),
                "The gradient of the objective does not match the finite-difference approximation.",
            )


if __name__ == "__main__":
    unittest.main()
