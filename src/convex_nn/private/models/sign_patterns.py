from typing import Dict, Any, Tuple, Optional

from convex_nn.activations import (
    sample_gate_vectors,
    compute_activation_patterns,
)

import numpy as np

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

    D, U = compute_activation_patterns(X, U)

    return D, U
