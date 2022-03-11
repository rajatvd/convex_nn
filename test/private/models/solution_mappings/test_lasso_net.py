"""
Tests for solutions mappings for LassoNet models.
"""

import unittest

import torch
import numpy as np
from parameterized import parameterized_class  # type: ignore

import lab

from convex_nn import activations
from convex_nn.private.models import (
    ConvexLassoNet,
    AL_LassoNet,
    GatedReLULayer,
)
import convex_nn.private.models.solution_mappings.lasso_net as sm
from convex_nn.private.utils.data import gen_regression_data


@parameterized_class(lab.TEST_GRID)
class TestLNSolutionMappings(unittest.TestCase):
    """Test mappings between solutions to the convex and non-convex LassoNet training problems."""

    d: int = 2
    n: int = 4
    c: int = 1
    rng: np.random.Generator = np.random.default_rng(778)

    def setUp(self):
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)

        train_set, _, _ = gen_regression_data(self.rng, self.n, 0, self.d, c=self.c)
        self.U = activations.sample_gate_vectors(self.rng, self.d, 100)
        self.D, self.U = lab.all_to_tensor(
            activations.compute_activation_patterns(train_set[0], self.U)
        )
        self.X, self.y = lab.all_to_tensor(train_set)

        self.P = self.D.shape[1]

        self.networks = {}
        self.gated_ln = ConvexLassoNet(
            self.d, self.D, self.U, kernel="einsum", c=self.c
        )
        self.relu_ln = AL_LassoNet(self.d, self.D, self.U, kernel="einsum", c=self.c)

    def test_mapping_gated_models_to_nonconvex(self):

        self.gated_ln.weights = lab.tensor(self.rng.random(self.gated_ln.weights.shape))

        manual_model = sm.construct_nc_ln_manual(
            self.gated_ln, grelu=True, remove_sparse=False
        )

        self.assertTrue(
            lab.allclose(manual_model(self.X), self.gated_ln(self.X)),
            "The manual version of the model did not have the same predictions!",
        )

        # test sparse models
        self.gated_ln.weights[:, [0, 2]] = 0.0

        manual_model = sm.construct_nc_ln_manual(
            self.gated_ln, grelu=True, remove_sparse=True
        )

        self.assertTrue(
            lab.allclose(manual_model(self.X), self.gated_ln(self.X)),
            "The manual version of the model did not have the same predictions!",
        )

    def test_mapping_relu_models_to_nonconvex(self):

        # set weights to be an interior point of the constraint set.
        network_weights = lab.stack(
            [lab.expand_dims(self.U.T, axis=0), 0.1 * lab.expand_dims(self.U.T, axis=0)]
        )
        skip_w = lab.tensor(lab.np_rng.random((2, self.c, 1, self.d)))

        self.relu_ln.weights = self.relu_ln._join_weights(network_weights, skip_w)

        manual_model = sm.construct_nc_ln_manual(
            self.relu_ln, grelu=False, remove_sparse=False
        )

        # check that predictions are identical for the two models
        self.assertTrue(
            lab.allclose(manual_model(self.X), self.relu_ln(self.X)),
            "The manual version of the model did not have the same predictions!",
        )

        # test sparse models
        self.relu_ln.weights[0, :, [0, 2]] = 0.0
        self.relu_ln.weights[0, :, [1]] = 0.0

        manual_model = sm.construct_nc_ln_manual(
            self.relu_ln, grelu=False, remove_sparse=True
        )

        # check that predictions are identical for the two models
        self.assertTrue(
            lab.allclose(manual_model(self.X), self.relu_ln(self.X)),
            "The manual version of the model did not have the same predictions!",
        )


if __name__ == "__main__":
    unittest.main()
