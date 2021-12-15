"""
Tests for convex neural networks.
"""

import unittest

import numpy as np
from parameterized import parameterized_class  # type: ignore

import lab

from cvx_nn.models import ConvexMLP, sign_patterns, operators
from cvx_nn import datasets


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

        (self.X, self.y), _, _ = datasets.generate_synthetic_regression(
            self.rng, self.n, 0, self.d, c=self.c, vector_output=True
        )

        self.D, self.U = sign_patterns.approximate_sign_patterns(
            self.rng, self.X, n_samples=10
        )
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
                    self.rng.standard_normal((self.c, self.d * self.P), dtype=self.dtype)
                )

                self.assertTrue(
                    lab.allclose(expanded_X @ v.T, data_op.matvec(v)),
                    f"The forward data operator with kernel {kernel_name} did not match direct computation for the vector with shape (d*P,)!",
                )

                # the operator should support matrices with shape (d, P) as well.
                w = v.reshape(self.c, self.P, self.d)
                self.assertTrue(
                    lab.allclose(expanded_X @ v.T, data_op.matvec(w).reshape(-1, self.c)),
                    f"The forward data operator with kernel {kernel_name} did not match direct computation for the vector with shape (d,P)!",
                )

                # check transpose operator
                w = lab.tensor(self.rng.standard_normal(self.n, dtype=self.dtype))

                self.assertTrue(
                    lab.allclose(expanded_X.T @ w, data_op.rmatvec(w)),
                    f"The transpose data operator with kernel {kernel_name} did not match direct computation for the vector with shape (d*P,)!",
                )

    def test_hessian_operator(self):
        """Test implementation of Hessian-vector operators."""
        for kernel_name in operators.KERNELS:
            nn = self.networks[kernel_name]

            # full matrix product
            network_H = nn.hessian(self.X, self.y, flatten=True)
            implicit_H = nn.hessian_operator(self.X, self.y)

            # try an assortment of random vectors
            for i in range(self.tries):
                v = lab.tensor(
                    self.rng.standard_normal(self.d * self.P, dtype=self.dtype)
                )
                self.assertTrue(
                    lab.allclose(network_H @ v, implicit_H.matvec(v)),
                    f"The full-matrix matrix-vector product with kernel {kernel_name} did not match direct computation for the vector with shape (d*P,)!",
                )

                # the operator should support matrices with shape (d, P) as well.
                w = v.reshape(self.P, self.d)
                self.assertTrue(
                    lab.allclose(network_H @ v, implicit_H.matvec(v).reshape(-1)),
                    f"The full-matrix matrix-vector product with kernel {kernel_name} did not match direct computation for the vector with shape (d,P)!",
                )

            # block-diagonal approximation
            network_bd_H = nn.hessian(self.X, self.y, block_diagonal=True)
            implicit_bd_H = nn.hessian_operator(self.X, self.y, block_diagonal=True)

            # try an assortment of random vectors
            for i in range(self.tries):
                v = lab.tensor(
                    self.rng.standard_normal(self.d * self.P, dtype=self.dtype)
                )
                w = v.reshape((self.P, self.d))
                # compute block-diagonal product by looping over blocks
                results = []
                for i, block in enumerate(network_bd_H):
                    results.append(block @ w[i])

                brute_force = lab.stack(results).reshape(-1)

                self.assertTrue(
                    lab.allclose(brute_force, implicit_bd_H.matvec(v)),
                    f"The block-diagonal matrix-vector product with kernel {kernel_name} did not match direct computation with vectors of shape (d*P,)!",
                )
                # check (d, P) matrices.
                self.assertTrue(
                    lab.allclose(brute_force, implicit_bd_H.matvec(w).reshape(-1)),
                    f"The block-diagonal matrix-vector product with kernel {kernel_name} did not match direct computation with vectors of shape (d, P)!",
                )


if __name__ == "__main__":
    unittest.main()
