"""
`convex_nn`: a Python packing for learning with convex neural networks.
"""

# Expose easy-to-use wrappers for optimizing convex models.
from convex_nn.wrappers import optimize, optimize_path
from convex_nn.wrappers import (
    GReLU_MLP,
    GReLU_LN,
    ReLU_MLP,
    ReLU_LN,
    REGULARIZERS,
    INITIALIZATIONS,
    PRECISIONS,
    FORMULATIONS,
)

__all__ = [
    "optimize",
    "optimize_path",
]