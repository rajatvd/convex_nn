"""Solution mappings for convex and non-convex formulations of neural network
training problems."""

from .mlps import (
    is_relu_compatible,
    is_grelu_compatible,
    construct_nc_manual,
    construct_nc_torch,
)

from .lasso_net import construct_nc_ln_manual

__all__ = [
    "is_relu_compatible",
    "is_grelu_compatible",
    "is_compatible",
    "get_nc_mlp",
    "get_nc_formulation",
    "is_compatible",
]

from typing import Union

import torch

from convex_nn.private.models.model import Model
from convex_nn.private.models.convex import AL_MLP, AL_LassoNet, ConvexMLP, ConvexLassoNet


def is_compatible(torch_model: torch.nn.Module) -> bool:
    """Check to see if there is a solution mapping mapping which is compatible
    with the architecture of the given model.

    :param torch_model: an instance of torch.nn.Module for which we want a convex program.
    :returns: true or false.
    """

    return is_relu_compatible(torch_model) or is_grelu_compatible(torch_model)


def get_nc_formulation(
    convex_model: Model,
    implementation: str = "torch",
    remove_sparse: bool = False,
) -> Union[torch.nn.Module, Model]:

    grelu = True
    if isinstance(convex_model, (AL_LassoNet, AL_MLP)):
        grelu = False

    if isinstance(convex_model, (ConvexLassoNet, AL_LassoNet)):
        if implementation == "torch":
            raise NotImplementedError(
                "PyTorch models have not been implemented for LassoNet."
            )
        elif implementation == "manual":
            return construct_nc_ln_manual(convex_model, grelu, remove_sparse)
    elif isinstance(convex_model, ConvexMLP):
        if implementation == "torch":
            return construct_nc_torch(convex_model, grelu, remove_sparse)
        elif implementation == "manual":
            return construct_nc_manual(convex_model, grelu, remove_sparse)
    else:
        raise ValueError(
            f"Implementation {implementation} not recognized. Please add it to 'solution_mappings.mlps.py'"
        )
