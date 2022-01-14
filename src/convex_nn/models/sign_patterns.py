"""
Generate sing patterns (aka hyperplane arrangements) for convex formulations of neural networks.
"""
from typing import Dict, Any, Tuple, Optional

import numpy as np
from numpy.random import Generator

import lab

# constants

DEFAULT_CONFIG = {"method": "sampler", "n_samples": 1000, "conv_patterns": False}

# functions


def get_sign_patterns(
    X: lab.Tensor,
    pattern_config: Optional[Dict[str, Any]] = None,
    U: Optional[lab.Tensor] = None,
) -> Tuple[lab.Tensor, lab.Tensor]:
    """Generate sign patterns for a convex neural network.

    Args:
        X: a data matrix with shape (n,d).
        pattern_config: (optional) a dictionary object specifying the sign-pattern
            generator and its arguments. If 'None' is passed, the config will
            default to 'DEFAULT_CONFIG'.
        U: (optional) a d x m matrix of pre-generated gate vectors.

    Returns:
        (D, U) - Tuple of sign patterns and gate vectors.
    """

    if U is None:
        if pattern_config is None:
            pattern_config = DEFAULT_CONFIG

        name = pattern_config["name"]
        # create sign patterns.
        if name == "sampler":
            rng = np.random.default_rng(seed=pattern_config.get("seed", 650))
            U = sample_gate_vectors(
                rng,
                X.shape[1],
                pattern_config.get("n_samples", 100),
                pattern_config.get("conv_patterns", False),
            )
        else:
            raise ValueError(f"Sign pattern generator {name} not recognized!")

    D, U = compute_sign_patterns(X, U)

    return D, U


def sample_gate_vectors(
    rng: Generator, d: int, n_samples: int, convolutional_patterns=False
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
        U -  a d x m matrix of gate vectors, where m <= n is guaranteed.
    """
    U = rng.standard_normal((d, n_samples))

    if convolutional_patterns:
        if d == 784:
            conv_masks = generate_conv_masks(rng, n_samples, 28, 1)
        elif d == 3072:
            conv_masks = generate_conv_masks(rng, n_samples, 32, 3)
        else:
            assert (
                False
            ), "Convolutional patterns only implemented for MNIST or CIFAR datasets"

        conv_masks = lab.to_np(conv_masks)
        U = U * conv_masks

    return lab.tensor(U, dtype=lab.get_dtype())


def compute_sign_patterns(
    X: lab.Tensor, U: lab.Tensor
) -> Tuple[lab.Tensor, lab.Tensor]:
    """Compute the set of unique sign patterns of 'XU'.
    :param X: data/feature matrix. Examples are expected to be rows.
    :param U: the "gate vectors" used to generate sign patterns. Gates are columns.
    :returns: a matrix of unique sign patterns, where each sign pattern is a column of the matrix.
    """
    n, d = X.shape

    XU = lab.smax(lab.matmul(X, U), 0)
    XU[XU > 0] = 1
    np_XU = lab.to_np(XU)

    np_D, indices = np.unique(np_XU, axis=1, return_index=True)
    D = lab.tensor(np_D, dtype=lab.get_dtype())
    U = U[:, indices]

    # filter out the zero column.
    non_zero_cols = lab.logical_not(lab.all(D == lab.zeros((n, 1)), axis=0))
    D = D[:, non_zero_cols]
    U = U[:, non_zero_cols]

    return D, U


def generate_conv_masks(
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
        lab.arange(upper_left_indices[i], upper_left_indices[i] + kernel_size)
        for i in range(num_samples)
    ]
    first_patch = [
        lab.concatenate(
            [
                lab.arange(
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
        lab.concatenate(
            [
                lab.arange(
                    first_patch[i][j],
                    first_patch[i][j] + channels * image_size ** 2,
                    image_size ** 2,
                )
                for j in range(kernel_size ** 2)
            ]
        ).tolist()
        for i in range(num_samples)
    ]
    mask = lab.zeros((num_samples, channels * image_size ** 2))
    mask[
        lab.arange(num_samples), lab.transpose(lab.tensor(all_patches).long(), 0, 1)
    ] = 1.0
    return mask.t()
