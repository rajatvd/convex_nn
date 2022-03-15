"""
Test augmented Lagrangian for LassoNet models.
"""

import unittest

import numpy as np
from scipy.optimize import check_grad  # type: ignore
from parameterized import parameterized_class  # type: ignore

import lab

from convex_nn import activations
from convex_nn.private.models import AL_LassoNet
from convex_nn.private.models.convex import operators
from convex_nn.private.utils.data import gen_regression_data


@parameterized_class(lab.TEST_GRID)
class TestIneqLassoNet(unittest.TestCase):
    """Test convex formulation of two-layer LassoNet network with inequality constraints."""

    d: int = 2
    n: int = 4
    c: int = 3
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
        self.U = activations.sample_dense_gates(self.rng, self.d, 100)
        self.D, self.U = lab.all_to_tensor(
            activations.compute_activation_patterns(train_set[0], self.U)
        )
        self.X, self.y = lab.all_to_tensor(train_set)

        self.P = self.D.shape[1]

        self.nn = AL_LassoNet(self.d, self.D, self.U, delta=2, gamma=1.0, c=self.c)

    def test_forward(self):
        """Test network predictions."""

        weights = lab.tensor(
            self.rng.standard_normal((2, self.c, self.P + 1, self.d), dtype=self.dtype)
        )
        self.nn.weights = weights

        expanded_X = operators.expanded_data_matrix(self.X, self.D)
        network_preds = self.nn(self.X)
        direct_preds = (
            lab.matmul(
                expanded_X,
                (weights[0, :, : self.P] - weights[1, :, : self.P])
                .reshape(self.c, -1)
                .T,
            )
            + self.X @ (weights[0, :, self.P] - weights[1, :, self.P]).T
        )

        self.assertTrue(
            lab.allclose(network_preds, direct_preds),
            "Network predictions did not match direct calculation.",
        )

    def test_weights_obj_grad(self):
        """Test implementation of objective and gradient for the augmented Lagrangian."""

        def obj_fn(w):
            return self.nn.objective(
                self.X, self.y, lab.tensor(w).reshape(2, self.c, self.P + 1, self.d)
            )

        def grad_fn(w):
            return lab.to_np(
                self.nn.grad(
                    self.X,
                    self.y,
                    lab.tensor(w).reshape(2, self.c, self.P + 1, self.d),
                ).ravel()
            )

        for tr in range(self.tries):
            weights = self.rng.standard_normal(
                (2, self.c, self.P + 1, self.d), dtype=self.dtype
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
