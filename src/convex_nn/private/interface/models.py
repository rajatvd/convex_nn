"""Convert models from :module:`convex_nn.models` into internal representations
and vice versa."""

from typing import Optional

import lab

from convex_nn.regularizers import (
    Regularizer,
    NeuronGL1,
    FeatureGL1,
    L2,
    L1,
)

from convex_nn.models import (
    Model,
    ConvexGatedReLU,
    NonConvexGatedReLU,
    ConvexReLU,
    NonConvexReLU,
)

from convex_nn.private.models import (
    ConvexMLP,
    AL_MLP,
    ReLUMLP,
    GatedReLUMLP,
    GroupL1Regularizer,
    FeatureGroupL1Regularizer,
    L2Regularizer,
    L1Regularizer,
)
from convex_nn.activations import compute_activation_patterns

from convex_nn.private.models import Model as InternalModel
from convex_nn.private.models import Regularizer as InternalRegularizer


def build_internal_regularizer(
    regularizer: Optional[Regularizer] = None,
) -> InternalRegularizer:
    """Convert public-facing regularizer objects into private implementations.

    Args:
        regularizer: a regularizer object from the public API.

    Returns:
        An internal regularizer object with the same state as the public regularizer.
    """
    reg: Optional[InternalRegularizer] = None

    lam = 0.0
    if regularizer is not None:
        lam = regularizer.lam

    if isinstance(regularizer, NeuronGL1):
        reg = GroupL1Regularizer(lam, group_by_feature=False)
    elif isinstance(regularizer, FeatureGL1):
        reg = FeatureGroupL1Regularizer(lam)
    elif isinstance(regularizer, L2):
        reg = L2Regularizer(lam)
    elif isinstance(regularizer, L1):
        reg = L1Regularizer(lam)

    return reg


def build_internal_model(
    model: Model, regularizer: Regularizer, X_train: lab.Tensor
) -> InternalModel:
    """Convert public-facing model objects into private implementations.

    Args:
        model: a model object from the public API.
        regularizer: a regularizer object from the public API.
        X_train: the :math:`n \\times d` training set.

    Returns:
        An internal model object with the same state as the public model.
    """
    assert isinstance(model, (ConvexReLU, ConvexGatedReLU))

    internal_model: InternalModel
    d, c = model.d, model.c
    internal_reg = build_internal_regularizer(regularizer)

    G = lab.tensor(model.G, dtype=lab.get_dtype())
    D, G = lab.all_to_tensor(
        compute_activation_patterns(lab.to_np(X_train), lab.to_np(G))
    )

    if isinstance(model, ConvexReLU):
        internal_model = AL_MLP(
            d,
            D,
            G,
            "einsum",
            1000,
            regularizer=internal_reg,
            c=c,
        )
        internal_model.weights = lab.stack(
            [
                lab.tensor(model.parameters[0], dtype=lab.get_dtype()),
                lab.tensor(model.parameters[1], dtype=lab.get_dtype()),
            ]
        )
    elif isinstance(model, ConvexGatedReLU):
        internal_model = ConvexMLP(
            d, D, G, "einsum", regularizer=internal_reg, c=c
        )
        internal_model.weights = lab.tensor(
            model.parameters[0], dtype=lab.get_dtype()
        )
    else:
        raise ValueError(f"Model object {model} not supported.")

    return internal_model


def update_public_model(model: Model, internal_model: InternalModel) -> Model:
    """Update public-facing model object to match state of internal model.

    Args:
        model: the public-facing model.
        internal_model: the internal model object.

    Returns:
        The updated public-facing model.
    """

    if isinstance(model, ConvexGatedReLU):
        assert isinstance(internal_model, ConvexMLP)
        model.parameters = [lab.to_np(internal_model.weights)]
    elif isinstance(model, ConvexReLU):
        assert isinstance(internal_model, AL_MLP)
        model.parameters = [
            lab.to_np(internal_model.weights[0]),
            lab.to_np(internal_model.weights[1]),
        ]

    return model


def build_public_model(internal_model: InternalModel) -> Model:
    """Construct a public-facing model from an internal model representation.

    Args:
        internal_model: the internal model.

    Returns:
        A public-facing model with identical state.
    """
    model: Model

    if isinstance(internal_model, GatedReLUMLP):
        U = lab.to_np(internal_model.U)
        model = NonConvexGatedReLU(U, internal_model.c)
        w1, w2 = internal_model._split_weights(internal_model.weights)

        model.set_parameters([lab.to_np(w1), lab.to_np(w2)])

    elif isinstance(internal_model, ReLUMLP):
        model = NonConvexReLU(
            internal_model.d, internal_model.p, internal_model.c
        )
        w1, w2 = internal_model._split_weights(internal_model.weights)

        model.set_parameters([lab.to_np(w1), lab.to_np(w2)])
    else:
        raise ValueError(f"Model {internal_model} not supported.")

    return model
