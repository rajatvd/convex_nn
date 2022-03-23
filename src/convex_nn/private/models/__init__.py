"""Models."""

from .model import Model
from .linear import LinearRegression, LogisticRegression
from .non_convex import (
    SequentialWrapper,
    LayerWrapper,
    ReLUMLP,
    GatedReLUMLP,
    GatedLassoNet,
    ReLULassoNet,
    GatedReLULayer,
)

from .convex import (
    ConvexMLP,
    ConvexLassoNet,
    AL_MLP,
    AL_LassoNet,
)

from .one_vs_all import OneVsAllModel

from .regularizers import (
    Regularizer,
    Constraint,
    GroupL1Orthant,
    GroupL1Regularizer,
    FeatureGroupL1Regularizer,
    L2Regularizer,
    L1Regularizer,
    OrthantConstraint,
    LassoNetConstraint,
    L1SquaredRegularizer,
)

from .solution_mappings import (
    is_compatible,
    get_nc_formulation,
)


__all__ = [
    "Model",
    "LinearRegression",
    "LogisticRegression",
    "SequentialWrapper",
    "LayerWrapper",
    "ReLUMLP",
    "GatedReLUMLP",
    "GatedLassoNet",
    "ReLULassoNet",
    "GatedReLULayer",
    "ConvexMLP",
    "ConvexLassoNet",
    "AL_MLP",
    "AL_LassoNet",
    "OneVsAllModel",
    "Regularizer",
    "Constraint",
    "GroupL1Regularizer",
    "FeatureGroupL1Regularizer",
    "GroupL1Orthant",
    "L2Regularizer",
    "L1Regularizer",
    "L1SquaredRegularizer",
    "OrthantConstraint",
    "LassoNetConstraint",
    "is_compatible",
    "get_nc_formulation",
]
