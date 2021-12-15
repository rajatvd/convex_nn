"""
Generate sing patterns (aka hyperplane arrangements) for convex formulations of neural networks.
"""
from typing import Dict, Any, Tuple

import numpy as np
from numpy.random import Generator

import lab

# constants

DEFAULT_CONFIG = {"method": "sampler", "n_samples": 1000, "conv_patterns": False}

# functions


def get_sign_patterns(
    X: lab.Tensor, pattern_config: Dict[str, Any]
) -> Tuple[lab.Tensor, lab.Tensor]:
    """
    :param X: a data matrix with shape (n,d).
    :param model_config: a dictionary object specifying the sign-pattern
        generator and its arguments. If 'None' is passed, the config will
        default to 'DEFAULT_CONFIG'.
    """

    if pattern_config is None:
        pattern_config = DEFAULT_CONFIG

    name = pattern_config["name"]
    # create sign patterns.
    if name == "sampler":
        rng = np.random.default_rng(seed=pattern_config.get("seed", 650))
        D, U = approximate_sign_patterns(
            rng,
            X,
            pattern_config["n_samples"],
            pattern_config.get("conv_patterns", False),
        )
    else:
        raise ValueError(f"Sign pattern generator {name} not recognized!")

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


def approximate_sign_patterns(
    rng: Generator, X: lab.Tensor, n_samples: int = None, convolutional_patterns=False
) -> Tuple[lab.Tensor, lab.Tensor]:
    """Compute an approximation of the set of possible sign patterns of 'Xu, u in R'.
    :param rng: random number generator.
    :param X: data/feature matrix. Examples are expected to be rows.
    :param n_samples: (optional) the number of samples to use when computing the approximation.
        More samples should find more sign patterns. Defaults to number of features.
    :returns: (sign_patterns, normal_vectors) --- matrix of unique sign patterns, where each sign pattern is a column of the matrix, and matrix of vectors creating these sign patterns.
    """
    np_X = lab.to_np(X)
    n, d = np_X.shape

    if n_samples is None:
        n_samples = d

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

    XU = np.maximum(np.matmul(np_X, U), 0)
    XU[XU > 0] = 1

    D, indices = np.unique(XU, axis=1, return_index=True)
    U = U[:, indices]

    # filter out the zero column.
    non_zero_cols = np.logical_not(np.all(D == np.zeros((n, 1)), axis=0))
    D = D[:, non_zero_cols]
    U = U[:, non_zero_cols]

    return lab.tensor(D, dtype=lab.get_dtype()), lab.tensor(U, dtype=lab.get_dtype())


def compute_sign_patterns(X: np.ndarray):
    """Compute the exact set of sign patterns of 'Xu, u in R'.
    :param X: data/feature matrix. Examples are expected to be rows.
    :returns: matrix of unique sign patterns. Each sign pattern is a column of the matrix.
    """

    raise NotImplementedError(
        "Computing exact hyperplane arrangements using the Edelsbrunner et al. (1986) algorithm has not been implemented yet."
    )
