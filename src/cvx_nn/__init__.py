"""
`cvx_nn`: a Python packing for learning with convex neural networks.
"""

# Expose easy-to-use wrappers for optimizing convex models.
from cvx_nn.wrappers import optimize
from cvx_nn.wrappers import (
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
]
