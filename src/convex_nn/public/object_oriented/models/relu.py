"""
Non-convex and convex formulations of two-layer ReLU networks.

Overview:
    This module provides implementations of non-convex and convex formulations for two-layer ReLU networks.
    The non-convex model has the prediction function,
    .. math::

        h(X) = \\sum_{i=1}^p (X W_{1i}^{\\top})_+ \\cdot W_{2i}^{\\top},

    where :math:`W_{1} \\in \\mathbb{R}^{p \\times d}` are the parameters of the first layer, and :math:`W_{2} \\in \\mathbb{R}^{c \\times p}` are the parameters of the second layer.

    The convex reformulation of the non-convex Gated ReLU network is obtain by enumerating the activation patterns :math:`D_i = \\text{diag}(1(X g_i > 0)), where :math:`g_i` is a fixed "gate" vector. Using a (possibly subsampled) set of activations :math:`\\tilde \\mathcal{D}`, the prediction function for the convex reformulation can be written as
    .. math::

        g(X) = \\sum_{D_i \\in \\mathcal{D}}^m D_i X (v_{i} - w_{i}),

    where :math:`v_i, w_i \\in \\mathbb{R}^{m \\times d}` are the model parameters. A one-vs-all strategy is used for the convex reformulation when the output dimension satisfies :math:`c > 1`.
"""

from typing import List

import lab

from convex_nn.private.models.convex.kernels.einsum_kernel import data_mvp


class ConvexReLU:
    """Convex reformulation of a ReLU Network with two-layers.

    This model has the prediction function

    .. math::

        g(X) = \\sum_{D_i \\in \\mathcal{D}}^m D_i X (v_{i} - w_{i}),

    A one-vs-all strategy is used to extend the model to multi-dimensional targets.

    Attributes:
        c: the output dimension.
        d: the input dimension.
        p: the number of neurons.
        G: the gate vectors used to generate the activation patterns :math:`D_i`, stored as a (d x p) matrix.
        parameters: the parameters of the model stored as a list of two (c x p x d) matrices.
    """

    def __init__(
        self,
        G: lab.Tensor,
        c: int = 1,
    ) -> None:
        """Construct a new convex Gated ReLU model.

        Args:
            G: (d x p) matrix of get vectors, where p is the number neurons.
            c: the output dimension.
        """

        self.G = G
        self.d, self.p = G.shape
        self.c = c

        # one linear model per gate vector
        self.parameters = [
            lab.zeros((c, self.p, self.d)),
            lab.zeros((c, self.p, self.d)),
        ]

    def get_parameters(self):
        """Get the model parameters."""
        return self.parameters

    def set_parameters(self, parameters: lab.Tensor):
        """Set the model parameters.

        This method safety checks the dimensionality of the new parameters.

        Args:
            parameters: the new model parameters.
        """
        assert len(parameters) == 2
        assert parameters[0].shape == (self.c, self.p, self.d)
        assert parameters[1].shape == (self.c, self.p, self.d)

        self.parameters = [parameters]

    def __call__(self, X: lab.Tensor) -> lab.Tensor:
        """Compute the model predictions for a given dataset.

        Args:
            X: an (n,d) array containing the data examples on which to predict.

        Returns:
            g(X) --- the model predictions for X.
        """
        local_D = lab.smax(X @ self.G, 0)
        local_D[local_D > 0] = 1

        return data_mvp(self.parameters[0] - self.parameters[1], X, local_D)


class NonConvexGatedReLU:
    """Convex reformulation of a ReLU Network with two-layers.

    This model has the prediction function

    .. math::

        h(X) = \\sum_{i=1}^p (X W_{1i}^{\\top})_+ \\cdot W_{2i}^{\\top},

    Attributes:
        c: the output dimension.
        d: the input dimension.
        p: the number of neurons.
        parameters: the parameters of the model stored as a list of matrices with shapes: [(p x d), (c x p)]
    """

    def __init__(
        self,
        d: int,
        p: int,
        c: int = 1,
    ) -> None:
        """Construct a new convex Gated ReLU model.

        Args:
            d: the input dimension.
            p: the number of neurons or "hidden units" in the network.
            c: the output dimension.
        """

        self.d = d
        self.p = p
        self.c = c

        # one linear model per gate vector
        self.parameters = [lab.zeros((self.p, self.d)), lab.zeros((self.c, self.p))]

    def get_parameters(self) -> List[lab.Tensor]:
        """Get the model parameters.

        Returns:
            [W_1, W_2] --- a list of model parameters.
        """

        return self.parameters

    def set_parameters(self, parameters: List[lab.Tensor]):
        """Set the model parameters.

        This method safety checks the dimensionality of the new parameters.

        Args:
            parameters: the new model parameters.
        """
        assert len(parameters) == 2
        assert parameters[0].shape == (self.p, self.d)
        assert parameters[1].shape == (self.c, self.p)

        self.parameters = parameters

    def __call__(self, X: lab.Tensor) -> lab.Tensor:
        """Compute the model predictions for a given dataset.

        Args:
            X: an (n,d) array containing the data examples on which to predict.

        Returns:
            h(X) --- the model predictions for X.
        """

        return lab.smax(X @ self.parameters[0].T, 0) @ self.parameters[1].T
