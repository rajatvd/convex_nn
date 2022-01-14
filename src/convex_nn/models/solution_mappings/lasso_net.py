"""
Solution mappings for LassoNet models.
"""

import lab

from .mlps import grelu_solution_mapping, relu_solution_mapping

from convex_nn.models.convex import ConvexLassoNet, AL_LassoNet, ConvexMLP
from convex_nn.models.non_convex import GatedLassoNet, ReLULassoNet
from convex_nn.models.one_vs_all import OneVsAllModel
from convex_nn.models.regularizers.l2 import L2Regularizer


def grelu_ln_solution_mapping(
    convex_model: ConvexLassoNet, remove_sparse: bool = False
):
    network_weights, skip_weights = convex_model._split_weights(convex_model.weights)
    skip_w = skip_weights[:, 0] - skip_weights[:, 1]

    dummy_model = ConvexMLP(
        convex_model.d, convex_model.D, convex_model.U, c=convex_model.c
    )
    dummy_model.weights = network_weights

    w1, w2, U = grelu_solution_mapping(dummy_model, remove_sparse)

    return skip_w, w1, w2, U


def relu_ln_solution_mapping(convex_model: AL_LassoNet, remove_sparse: bool = False):
    network_weights, skip_weights = convex_model._split_weights(convex_model.weights)
    skip_w = skip_weights[0] - skip_weights[1]

    dummy_model = AL_LassoNet(
        convex_model.d, convex_model.D, convex_model.U, c=convex_model.c
    )
    dummy_model.weights = network_weights

    w1, w2 = relu_solution_mapping(dummy_model, remove_sparse=remove_sparse)

    return skip_w, w1, w2


def convex_ln_to_manual_ln(
    convex_model: ConvexLassoNet,
    manual_model: ReLULassoNet,
    grelu: bool = False,
    remove_sparse: bool = False,
):
    if grelu:
        skip_w, first_layer, second_layer, U = grelu_ln_solution_mapping(
            convex_model, remove_sparse
        )
        assert isinstance(manual_model, GatedLassoNet)

        network_weights = manual_model.nl_model._join_weights(first_layer, second_layer)
        manual_model.weights = manual_model._join_weights(network_weights, skip_w)
        manual_model.U = U
        manual_model.p = U.shape[1]
        manual_model.nl_model.p = manual_model.p
        manual_model.nl_model.U = U
    else:
        skip_w, first_layer, second_layer = relu_ln_solution_mapping(
            convex_model, remove_sparse
        )
        network_weights = manual_model.nl_model._join_weights(first_layer, second_layer)
        manual_model.weights = manual_model._join_weights(network_weights, skip_w)
        manual_model.p = first_layer.shape[0]
        manual_model.nl_model.p = manual_model.p

    return manual_model


def construct_nc_ln_manual(
    convex_model: ConvexMLP,
    grelu: bool = False,
    remove_sparse: bool = False,
):
    per_class_models = []
    full_weights = convex_model.weights

    for c in range(convex_model.c):
        if grelu:
            convex_model.weights = full_weights[c : c + 1]
            manual_model = GatedLassoNet(
                convex_model.d,
                convex_model.U,
                c=1,
            )
        else:
            convex_model.weights = full_weights[:, c : c + 1]
            manual_model = ReLULassoNet(convex_model.d, convex_model.p, c=1)

        nc_model = convex_ln_to_manual_ln(
            convex_model, manual_model, grelu, remove_sparse
        )
        per_class_models.append(nc_model)

    convex_model.weights = full_weights

    l2_regularizer = None
    if convex_model.regularizer is not None:
        l2_regularizer = L2Regularizer(convex_model.regularizer.lam)

    if convex_model.c == 1:
        nc_model = per_class_models[0]
        nc_model.regularizer = l2_regularizer
    else:
        nc_model = OneVsAllModel(
            convex_model.d, per_class_models, convex_model.c, l2_regularizer
        )

    return nc_model
