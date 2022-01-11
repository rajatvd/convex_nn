"""
Tests for external-facing functions.
"""

import unittest

import torch
import numpy as np
from parameterized import parameterized_class  # type: ignore

import lab

from cvx_nn import datasets
from cvx_nn.models import sign_patterns, GatedReLULayer
from cvx_nn.wrappers import optimize, _transform_weights


@parameterized_class(lab.TEST_GRID)
class TestWrapper(unittest.TestCase):
    """Test implementation of proximal gradient descent with and without line-search."""

    # basic parameters
    d: int = 10
    n: int = 200
    rng: np.random.Generator = np.random.default_rng(seed=778)

    # proximal-gd parameters
    max_iters: int = 1000

    # line-search parameters
    beta: float = 0.8
    init_step_size: float = 10

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)

        # generate random dataset
        ((self.X, self.y,), _, self.wopt,) = datasets.generate_synthetic_regression(
            778, self.n, 0, self.d, sparse_opt=False, unitize_data_cols=False
        )

        self.D, self.U = sign_patterns.approximate_sign_patterns(
            self.rng, self.X, n_samples=10
        )

    def test_default_options(self):
        """Test using solver with default options."""

        model, metrics = optimize(
            self.X, self.y, formulation="grelu_mlp", max_patterns=100
        )
        model, metrics = optimize(
            self.X, self.y, formulation="grelu_mlp", max_patterns=100
        )

    def test_solving_l2_problem(self):
        """Test solving ridge-regression problem with fast iterative method."""

        model, metrics = optimize(
            self.X,
            self.y,
            formulation="grelu_mlp",
            max_patterns=100,
            verbose=False,
            reg_type="l2",
        )

    def test_backends(self):
        """Test using PyTorch and NumPy backends"""

        model, metrics = optimize(
            self.X,
            self.y,
            formulation="grelu_mlp",
            max_patterns=100,
            verbose=False,
            backend=lab.backend,
        )

        model, metrics = optimize(
            self.X,
            self.y,
            formulation="relu_mlp",
            max_patterns=100,
            verbose=False,
            backend=lab.backend,
        )

    def test_dtypes(self):
        """Test using different data types"""

        model, metrics = optimize(
            self.X,
            self.y,
            formulation="grelu_mlp",
            max_patterns=100,
            verbose=False,
            backend=lab.backend,
            dtype="float32",
        )

        model, metrics = optimize(
            self.X,
            self.y,
            formulation="grelu_mlp",
            max_patterns=100,
            verbose=False,
            backend=lab.backend,
            dtype="float64",
        )

    def test_settings(self):
        """Test modifying some settings"""

        model, metrics = optimize(
            self.X,
            self.y,
            formulation="relu_mlp",
            max_primal_iters=500,
            max_dual_iters=10,
            grad_tol=1e-6,
            constraint_tol=1e-4,
            initialization="zero",
            max_patterns=100,
            verbose=False,
            backend=lab.backend,
            dtype="float64",
        )

    def test_passing_torch_model(self):
        torch_model = torch.nn.Sequential(
            torch.nn.Linear(10, 10, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1, bias=False),
        )

        model, metrics = optimize(
            self.X,
            self.y,
            formulation="relu_mlp",
            model=torch_model,
            verbose=False,
            backend=lab.backend,
            dtype="float32",
        )

        torch_model = torch.nn.Sequential(
            GatedReLULayer(self.U),
            torch.nn.Linear(self.U.shape[1], 1, bias=False),
        )

        model, metrics = optimize(
            self.X,
            self.y,
            formulation="grelu_mlp",
            model=torch_model,
            verbose=False,
            backend=lab.backend,
            dtype="float32",
        )

    def test_data_transformation(self):
        """Test undoing unitization of the training data."""

        (X_unitized, _), _, column_norms = datasets.unitize_features(
            (self.X, self.y), return_column_norms=True
        )

        model, _ = optimize(
            X_unitized,
            self.y,
            formulation="grelu_mlp",
            max_patterns=100,
            verbose=False,
            unitize_data_cols=False,
            return_convex_form=True,
        )

        unitized_preds = model(X_unitized)
        model.weights = _transform_weights(model.weights, column_norms)
        preds = model(self.X)

        self.assertTrue(lab.allclose(unitized_preds, preds, atol=1e-4, rtol=1e-4), "")


if __name__ == "__main__":
    unittest.main()
