"""
Tests for convex neural networks.
"""

import unittest

import numpy as np
from parameterized import parameterized_class  # type: ignore

import lab

from convex_nn.models import ConvexMLP, sign_patterns
from convex_nn.models.convex import operators
from convex_nn.utils.data import gen_regression_data


@parameterized_class(lab.TEST_GRID)
class TestConvexMLP(unittest.TestCase):
    """Test convex formulation of two-layer ReLU network."""

    d: int = 2
    n: int = 3
    c: int = 5
    rng: np.random.Generator = np.random.default_rng(778)

    tries: int = 10

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)

        train_set, _, _ = gen_regression_data(self.rng, self.n, 0, self.d, c=self.c)
        self.X, self.y = lab.all_to_tensor(train_set)
        self.y = lab.squeeze(self.y)

        self.U = sign_patterns.sample_gate_vectors(self.rng, self.d, 10)
        self.D, self.U = sign_patterns.compute_sign_patterns(self.X, self.U)
        self.P = self.D.shape[1]

        # instantiate models with all available kernels.
        self.networks = {}
        for kernel_name in operators.KERNELS:
            self.networks[kernel_name] = ConvexMLP(
                self.d, self.D, self.U, kernel=kernel_name, c=self.c
            )

    def test_forward(self):
        """Test network predictions."""
        for kernel_name in operators.KERNELS:
            nn = self.networks[kernel_name]

            weights = lab.tensor(
                self.rng.standard_normal((self.c, self.P, self.d), dtype=self.dtype)
            )
            nn.weights = weights

            expanded_X = operators.expanded_data_matrix(self.X, self.D)
            network_preds = nn(self.X)
            direct_preds = lab.matmul(expanded_X, weights.reshape(self.c, -1).T)

            self.assertTrue(
                lab.allclose(network_preds, direct_preds),
                f"Network predictions with kernel {kernel_name} failed to match direct calculation.",
            )

    def test_data_matrix_operator(self):
        """Test implementation of data matrix operators."""

        for kernel_name in operators.KERNELS:
            nn = self.networks[kernel_name]
            data_op = nn.data_operator(self.X)
            expanded_X = operators.expanded_data_matrix(self.X, self.D)

            # check forward operator

            # try an assortment of random vectors
            for i in range(self.tries):
                v = lab.tensor(
                    self.rng.standard_normal(
                        (self.c, self.d * self.P), dtype=self.dtype
                    )
                )

                self.assertTrue(
                    lab.allclose(expanded_X @ v.T, data_op.matvec(v)),
                    f"The forward data operator with kernel {kernel_name} did not match direct computation for the vector with shape (d*P,)!",
                )

                # the operator should support matrices with shape (d, P) as well.
                w = v.reshape(self.c, self.P, self.d)
                self.assertTrue(
                    lab.allclose(
                        expanded_X @ v.T, data_op.matvec(w).reshape(-1, self.c)
                    ),
                    f"The forward data operator with kernel {kernel_name} did not match direct computation for the vector with shape (d,P)!",
                )

                # check transpose operator
                w = lab.tensor(self.rng.standard_normal(self.n, dtype=self.dtype))

                self.assertTrue(
                    lab.allclose(expanded_X.T @ w, data_op.rmatvec(w)),
                    f"The transpose data operator with kernel {kernel_name} did not match direct computation for the vector with shape (d*P,)!",
                )


if __name__ == "__main__":
    unittest.main()
