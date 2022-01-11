"""
Tests for group-l1 regularization.
"""

import unittest

import numpy as np
from scipy.optimize import check_grad  # type: ignore
from parameterized import parameterized_class  # type: ignore

import lab

from cvx_nn.models import ConvexMLP, sign_patterns
from cvx_nn.models.convex import operators
from cvx_nn.models.regularizers.group_l1 import GroupL1Regularizer
from cvx_nn import datasets


@parameterized_class(lab.TEST_GRID)
class TestGroupL1Regularizer(unittest.TestCase):
    """Tests for group L1 regularizers."""

    d: int = 2
    n: int = 4
    c: int = 3
    lam: float = 2.0
    rng: np.random.Generator = np.random.default_rng(778)
    tries: int = 10

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)

        # generate dataset
        (self.X, self.y), _, self.wopt = datasets.generate_synthetic_regression(
            self.rng, self.n, 0, self.d, c=self.c, vector_output=True
        )
        # initialize model
        self.D, self.U = sign_patterns.approximate_sign_patterns(
            self.rng, self.X, n_samples=100
        )
        self.p = self.D.shape[1]

        self.regularizer = GroupL1Regularizer(lam=self.lam)
        self.regularized_model = ConvexMLP(
            self.d, self.D, self.U, regularizer=self.regularizer, c=self.c
        )

        self.random_weights = self.rng.standard_normal(
            (self.c, self.p, self.d), dtype=self.dtype
        )
        self.regularized_model.weights = lab.tensor(self.random_weights * 100)
        self.expanded_X = operators.expanded_data_matrix(self.X, self.D)

        self.objective, self.grad = self.regularized_model.get_closures(self.X, self.y)

        def objective_fn(v=None):
            if v is not None:
                v = lab.tensor(v).reshape(self.c, self.p, self.d)

            return self.objective(v)

        def grad_fn(v=None):
            if v is not None:
                v = lab.tensor(v).reshape(self.c, self.p, self.d)

            return lab.to_np(lab.ravel(self.grad(v)))

        self.objective_fn = objective_fn
        self.grad_fn = grad_fn

    def test_objective(self):
        """Check that the regularized objective is computed properly"""
        random_weights = self.regularized_model.weights

        # the regularized model should have the same objective as the underlying model when lam = 0
        self.regularized_model.regularizer.lam = 0

        self.assertTrue(
            lab.allclose(
                self.regularized_model.objective(
                    self.X, self.y, ignore_regularizer=True
                ),
                self.objective_fn(),
            ),
            "The regularized model objective did not match the underlying model objective when lam = 0!",
        )

        self.regularized_model.regularizer.lam = self.lam
        regularized_loss = (
            lab.sum(
                ((self.expanded_X @ random_weights.reshape(self.c, -1).T) - self.y) ** 2
            )
        ) / (2 * lab.size(self.y)) + self.lam * lab.sum(
            lab.sqrt(lab.sum(random_weights ** 2, axis=-1))
        )

        self.assertTrue(
            lab.allclose(
                regularized_loss, self.regularized_model.objective(self.X, self.y)
            ),
            "The regularized model objective did not match direct computation!",
        )

    def test_grad(self):
        """Check that the gradient is computed properly."""
        # the gradient should be the same as the underlying model gradient when lam = 0.
        self.regularized_model.regularizer.lam = 0
        model_grad = self.regularized_model.grad(self.X, self.y)

        self.assertTrue(
            lab.allclose(
                model_grad,
                self.regularized_model.grad(self.X, self.y, ignore_regularizer=True),
            ),
            "Gradient does not match underlying model gradient when lambda = 0.",
        )

        self.regularized_model.regularizer.lam = 1.0
        model_grad = self.regularized_model.grad(self.X, self.y)
        self.assertFalse(
            lab.allclose(
                model_grad,
                self.regularized_model.grad(self.X, self.y, ignore_regularizer=True),
            ),
            "Gradient matched underlying model gradient when lambda != 0.",
        )

        # set some weights to zero and check that their gradient is zero when lambda is large enough.
        self.regularized_model.regularizer.lam = 100000
        self.regularized_model.weights[:, [1, 3]] = 0
        self.assertTrue(
            lab.all(self.regularized_model.grad(self.X, self.y)[:, [1, 3]] == 0),
            "The min-norm subgradient should be sparse.",
        )

        self.regularized_model.regularizer.lam = 100
        # test the gradient against finite differences when all weights are non-zero
        for i in range(self.tries):
            v = (
                self.rng.standard_normal((self.c * self.p * self.d), dtype=self.dtype)
                * 1
            )

            # we expect a lot of numerical error in finite-difference approximation.
            self.assertTrue(
                np.isclose(
                    check_grad(self.objective_fn, self.grad_fn, v),
                    0,
                    rtol=1e-3,
                    atol=1e-3,
                ),
                "Regularized gradient does not match finite differences approximation.",
            )


if __name__ == "__main__":
    unittest.main()
