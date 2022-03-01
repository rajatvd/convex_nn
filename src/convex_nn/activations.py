"""Generate activation patterns for convex reformulations of neural networks.

Overview:
    This module provides functions for generating activation patterns for ReLU or threshold activation patterns which are used when forming (subsampled) convex reformulations of ReLU and threshold-activation neural networks.
    An activation pattern is a vector of the form,

    .. math:: d_i = 1(X w \\geq 0),

    where :math:`1(z \\geq 0)` is an element-wise indicator function whose i'th element is one when :math:`z_i \\geq 0` and zero-otherwise and :math:`w \\in \\mathbb{R}^d` is a "gate vector".
    Forming the convex reformulation of a neural network with ReLU activations requires enumerating the activation patterns a single ReLU or threshold neuron can take on,

    .. math:: \\mathcal{D} = \\left\\{  d = 1(X w \\geq 0) : w \\in \\mathbb{R}^d \\right\\}.

    In practice, :math`\\mathcal{D}` can approximated sampling vectors :math:`w \\sim P` according to some distribution :math:`P` and then computing the corresponding pattern :math:`d_i`.
"""

from typing import Tuple

import numpy as np
from numpy.random import Generator


def sample_gate_vectors(
    rng: Generator,
    d: int,
    n_samples: int,
    convolutional_patterns: bool = False,
):
    """Generate gate vectors by random sampling.

    Args:
        rng: a random number generator.
        d: the dimensionality of the gate vectors.
        n_samples: the number of samples to use when computing the approximation.
            More samples should find more sign patterns.
        convolutional_patterns: whether or not to sample the gates as convolutional
            filters.

    Returns:
        G -  a :math:`d \\times \\text{n_samples}` matrix of gate vectors.
    """
    G = rng.standard_normal((d, n_samples))

    if convolutional_patterns:
        if d == 784:
            conv_masks = _generate_conv_masks(rng, n_samples, 28, 1)
        elif d == 3072:
            conv_masks = _generate_conv_masks(rng, n_samples, 32, 3)
        else:
            assert (
                False
            ), "Convolutional patterns only implemented for MNIST or CIFAR datasets"

        G = G * conv_masks

    return G


def compute_activation_patterns(
    X: np.ndarray,
    G: np.ndarray,
    filter_duplicates: bool = True,
    filter_zero: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute activation patterns corresponding to a set of gate vectors.

    Args:
        X: an :math:`n \\times d` data matrix, where the examples are stored as rows.
        G: an :math:`d \\times m` matrix of "gate vectors" used to generate the sign patterns.
        filter_duplicates: whether or not to remove duplicate activation patterns and the corresponding
        filter_zero: whether or not to filter the zero activation pattern and corresponding gates.
            Defaults to `True`.
    Returns:
        (D, G), where

        - D is a :math:`n \\times b` matrix of (possibly unique) sign patterns where each sign pattern is a column of the matrix and :math:`b \\leq m`.

        - G is a :math:`d \\times b` matrix of gates vectors generating D.
    """
    n, d = X.shape

    XG = np.maximum(np.matmul(X, G), 0)
    XG[XG > 0] = 1

    if filter_duplicates:
        D, indices = np.unique(XG, axis=1, return_index=True)
        G = G[:, indices]

    # filter out the zero column.
    if filter_zero:
        non_zero_cols = np.logical_not(np.all(D == np.zeros((n, 1)), axis=0))
        D = D[:, non_zero_cols]
        G = G[:, non_zero_cols]

    return D, G


def _generate_conv_masks(
    rng: Generator,
    num_samples: int,
    image_size: int = 32,
    channels: int = 3,
    kernel_size: int = 3,
):
    upper_left_coords = rng.integers(
        low=0, high=image_size - kernel_size - 1, size=(num_samples, 2)
    )
    upper_left_indices = image_size * upper_left_coords[:, 0] + upper_left_coords[:, 1]
    upper_rows = [
        np.arange(upper_left_indices[i], upper_left_indices[i] + kernel_size)
        for i in range(num_samples)
    ]
    first_patch = [
        np.concatenate(
            [
                np.arange(
                    upper_rows[i][j],
                    upper_rows[i][j] + kernel_size * image_size,
                    image_size,
                )
                for j in range(kernel_size)
            ]
        )
        for i in range(num_samples)
    ]
    all_patches = [
        np.concatenate(
            [
                np.arange(
                    first_patch[i][j],
                    first_patch[i][j] + channels * image_size ** 2,
                    image_size ** 2,
                )
                for j in range(kernel_size ** 2)
            ]
        ).tolist()
        for i in range(num_samples)
    ]
    mask = np.zeros((num_samples, channels * image_size ** 2))

    def _transpose(x):
        dims = np.arange(len(x.shape))
        dims[[0, 1]] = [1, 0]

        return np.transpose(x, dims)

    mask[np.arange(num_samples), _transpose(all_patches)] = 1.0

    return mask.t()
