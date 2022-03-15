"""Test different methods for sampling activation patterns.

TODO:
    - 
"""

import unittest

import numpy as np

from convex_nn.private.utils.data.synthetic import gen_regression_data
from convex_nn.activations import (
    sample_gate_vectors,
    sample_sparse_gates,
    generate_index_lists,
)


class TestActivations(unittest.TestCase):
    """Test different methods for sampling activation patterns."""

    def setUp(self):
        # Generate realizable synthetic classification problem (ie. Figure 1)
        n_train = 100
        n_test = 100
        self.d = 25
        c = 2
        kappa = 1
        self.max_neurons = 100

        (
            (self.X_train, self.y_train),
            (self.X_test, self.y_test),
            _,
        ) = gen_regression_data(
            123, n_train, n_test, self.d, c, kappa=kappa, unitize_data_cols=False
        )
        self.seed = 123
        self.n_gates = 1000

    def test_dense_gates(self):
        """Test sampling dense gate vectors."""

        G = sample_gate_vectors(self.seed, self.d, self.n_gates, gate_type="dense")

        # check gate shape.
        self.assertTrue(G.shape == (self.d, self.n_gates))

        # check gates are dense
        self.assertTrue(np.all(G != 0.0))

    def test_feature_sparse_gates(self):
        """Test sampling feature-sparse gate vectors."""
        G = sample_sparse_gates(
            np.random.default_rng(self.seed),
            self.d,
            self.n_gates,
            sparsity_indices=np.arange(self.d).reshape(self.d, 1).tolist(),
        )

        # check gate shape.
        self.assertTrue(G.shape == (self.d, self.n_gates))

        # each gate should have exactly one sparse index
        self.assertTrue(np.all(np.sum(G == 0, axis=0) == 1))

        # try generating index lists
        order_one = generate_index_lists(self.d, 1)
        self.assertTrue(
            np.all(order_one == np.arange(self.d).reshape(self.d, 1).tolist())
        )

        order_two = generate_index_lists(self.d, 2)

        G = sample_sparse_gates(
            np.random.default_rng(self.seed),
            self.d,
            self.n_gates,
            sparsity_indices=order_two,
        )

        # each gate should have exactly one or two sparse indices
        n_sparse_indices = np.sum(G == 0, axis=0)
        self.assertTrue(
            np.all(np.logical_or(n_sparse_indices == 1, n_sparse_indices == 2))
        )


if __name__ == "__main__":
    unittest.main()
