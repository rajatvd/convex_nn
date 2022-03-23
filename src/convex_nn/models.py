"""Non-convex and convex formulations of two-layer neural networks.

Overview:
    This module provides implementations of non-convex and convex formulations
    for two-layer ReLU and Gated ReLU networks. The difference between ReLU
    and Gated ReLU networks is the activation function; Gated ReLU networks
    use fixed "gate" vectors when computing the activation pattern while
    standard ReLU networks use the model parameters. Concretely, the prediction
    function for a two ReLU network is

    .. math:: h(X) = \\sum_{i=1}^p (X W_{1i}^{\\top})_+ \\cdot W_{2i}^{\\top},

    where :math:`W_{1} \\in \\mathbb{R}^{p \\times d}` are the parameters of
    the first layer, and :math:`W_{2} \\in \\mathbb{R}^{c \\times p}` are the
    parameters of the second layer. In contrast, Gated ReLU networks predict as

    .. math:: h(X) = \\sum_{i=1}^p \\text{diag}(X g_i > 0) X W_{1i}^{\\top}
        \\cdot W_{2i}^{\\top},

    where the :math:`g_i` vectors are fixed (ie. not learned) gates.

    The convex reformulations of the ReLU and Gated ReLU models are obtained
    by enumerating the possible activation patterns
    :math:`D_i = \\text{diag}(1(X g_i > 0))`.  For a Gated ReLU model, the
    activations are exactly specified by the set of gate vectors, while for
    ReLU models the space of activation is much larger.
    Using a (possibly subsampled) set of activations :math:`\\mathcal{D}`,
    the prediction function for the convex reformulation of a two-layer ReLU
    network can be written as

    .. math:: g(X) = \\sum_{D_i \\in \\mathcal{D}}^m D_i X (v_{i} - w_{i}),

    where :math:`v_i, w_i \\in \\mathbb{R}^{m \\times d}` are the model
    parameters. For Gated ReLU models, the convex reformulation is

    .. math:: g(X) = \\sum_{i=1}^m \\text{diag}(X g_i > 0) X U_{i},

    where :math:`U \\in \\mathbb{R}^{m \\times d}` are the model parameters
    and :math:`g_i` are the gate vectors from the non-convex model. For both
    convex reformulations, a one-vs-all strategy is used for the convex
    reformulation when the output dimension satisfies :math:`c > 1`.
"""

from typing import List

import numpy as np


class Model:
    """Base class for convex and non-convex models.

    Attributes:
        c: the output dimension.
        d: the input dimension.
        p: the number of neurons.
        parameters: a list of NumPy arrays comprising the model parameters.
    """

    d: int
    p: int
    c: int
    parameters: List[np.ndarray]


class LinearModel(Model):
    """Basic linear model.

    This model has the prediction function :math:`g(X) = X W^\\top`, where
    :math:`W \\in \\mathbb{R}^{c \\times d}` is a matrix of weights.


    Attributes:
        c: the output dimension.
        d: the input dimension.
        p: the number of neurons. This is is always `1` for a linear model.
        parameters: a list of NumPy arrays comprising the model parameters.
    """

    def __init__(self, d: int, c: int):
        """
        Args:
            d: the input dimension.
            c: the output dimension.
        """
        self.d = d
        self.c = c
        self.p = 1

        self.parameters = [np.zeros((c, d))]

    def get_parameters(self) -> List[np.ndarray]:
        """Get the model parameters."""
        return self.parameters

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set the model parameters.

        This method safety checks the dimensionality of the new parameters.

        Args:
            parameters: the new model parameters.
        """
        assert parameters[0].shape == (self.c, self.d)

        self.parameters = parameters

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Compute the model predictions for a given dataset.

        Args:
            X: an (n  d) array containing the data examples on
                which to predict.

        Returns:
            g(X) --- the model predictions for X.
        """
        return X @ self.parameters[0].T


class ConvexGatedReLU(Model):
    """Convex reformulation of a Gated ReLU Network with two-layers.

    This model has the prediction function

    .. math::

        g(X) = \\sum_{i=1}^m \\text{diag}(X g_i > 0) X U_{1i}.

    A one-vs-all strategy is used to extend the model to multi-dimensional
    targets.

    Attributes:
        c: the output dimension.
        d: the input dimension.
        p: the number of neurons.
        G: the gate vectors for the Gated ReLU activation stored as a
            (d x p) matrix.
        parameters: the parameters of the model stored as a list of one
            (c x p x d) tensor.
    """

    def __init__(
        self,
        G: np.ndarray,
        c: int = 1,
    ) -> None:
        """Construct a new convex Gated ReLU model.

        Args:
            G: a (d x p) matrix of get vectors, where p is the
                number neurons.
            c: the output dimension.
        """

        self.G = G
        self.d, self.p = G.shape
        self.c = c

        # one linear model per gate vector
        self.parameters = [np.zeros((c, self.p, self.d))]

    def get_parameters(self) -> List[np.ndarray]:
        """Get the model parameters."""
        return self.parameters

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set the model parameters.

        This method safety checks the dimensionality of the new parameters.

        Args:
            parameters: the new model parameters.
        """
        assert parameters[0].shape == (self.c, self.p, self.d)

        self.parameters = parameters

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Compute the model predictions for a given dataset.

        Args:
            X: an (n  d) array containing the data examples on
                which to predict.

        Returns:
            g(X) --- the model predictions for X.
        """
        D = np.maximum(X @ self.G, 0)
        D[D > 0] = 1

        return np.einsum("ij, lkj, ik->il", X, self.parameters[0], D)


class NonConvexGatedReLU(Model):
    """Convex reformulation of a Gated ReLU Network with two-layers.

    This model has the prediction function

    .. math:: h(X) = \\sum_{i=1}^m \\text{diag}(X g_i > 0) X W_{1i} \\cdot
        W_{2i},

    Attributes:
        c: the output dimension.
        d: the input dimension.
        p: the number of neurons.
        G: the gate vectors for the Gated ReLU activation stored as a
            (d x p) matrix.
        parameters: the parameters of the model stored as a list of matrices
            with shapes: [(p x d), (c x p)]
    """

    def __init__(
        self,
        G: np.ndarray,
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
            np.zeros((self.p, self.d)),
            np.zeros((self.c, self.p)),
        ]

    def get_parameters(self) -> List[np.ndarray]:
        """Get the model parameters.

        Returns:
            [W_1, W_2] --- a list of model parameters.
        """

        return self.parameters

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set the model parameters.

        This method safety checks the dimensionality of the new parameters.

        Args:
            parameters: the new model parameters.
        """
        assert len(parameters) == 2
        assert parameters[0].shape == (self.p, self.d)
        assert parameters[1].shape == (self.c, self.p)

        self.parameters = parameters

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Compute the model predictions for a given dataset.

        Args:
            X: an (n,d) array containing the data examples on which to predict.

        Returns:
            h(X) --- the model predictions for X.
        """
        D = np.maximum(X @ self.G, 0)
        D[D > 0] = 1

        return np.multiply(D, X @ self.parameters[0].T) @ self.parameters[1].T


class ConvexReLU(Model):
    """Convex reformulation of a ReLU Network with two-layers.

    This model has the prediction function

    .. math:: g(X) = \\sum_{D_i \\in \\mathcal{D}}^m D_i X (v_{i} - w_{i}),

    A one-vs-all strategy is used to extend the model to multi-dimensional
        targets.

    Attributes:
        c: the output dimension.
        d: the input dimension.
        p: the number of neurons.
        G: the gate vectors used to generate the activation patterns
            :math:`D_i`, stored as a (d x p) matrix.
        parameters: the parameters of the model stored as a list of two
            (c x p x d) matrices.
    """

    def __init__(
        self,
        G: np.ndarray,
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
            np.zeros((c, self.p, self.d)),
            np.zeros((c, self.p, self.d)),
        ]

    def get_parameters(self) -> List[np.ndarray]:
        """Get the model parameters."""
        return self.parameters

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set the model parameters.

        This method safety checks the dimensionality of the new parameters.

        Args:
            parameters: the new model parameters.
        """
        assert len(parameters) == 2
        assert parameters[0].shape == (self.c, self.p, self.d)
        assert parameters[1].shape == (self.c, self.p, self.d)

        self.parameters = parameters

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Compute the model predictions for a given dataset.

        Args:
            X: an (n,d) array containing the data examples on which to predict.

        Returns:
            g(X) --- the model predictions for X.
        """
        D = np.maximum(X @ self.G, 0)
        D[D > 0] = 1
        p_diff = self.parameters[0] - self.parameters[1]
        return np.einsum("ij, lkj, ik->il", X, p_diff, D)


class NonConvexReLU(Model):
    """Convex reformulation of a ReLU Network with two-layers.

    This model has the prediction function

    .. math:: h(X) = \\sum_{i=1}^p (X W_{1i}^{\\top})_+ \\cdot W_{2i}^{\\top},

    Attributes:
        c: the output dimension.
        d: the input dimension.
        p: the number of neurons.
        parameters: the parameters of the model stored as a list of matrices
            with shapes: [(p x d), (c x p)]
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
        self.parameters = [
            np.zeros((self.p, self.d)),
            np.zeros((self.c, self.p)),
        ]

    def get_parameters(self) -> List[np.ndarray]:
        """Get the model parameters.

        Returns:
            [W_1, W_2] --- a list of model parameters.
        """

        return self.parameters

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set the model parameters.

        This method safety checks the dimensionality of the new parameters.

        Args:
            parameters: the new model parameters.
        """
        assert len(parameters) == 2
        assert parameters[0].shape == (self.p, self.d)
        assert parameters[1].shape == (self.c, self.p)

        self.parameters = parameters

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Compute the model predictions for a given dataset.

        Args:
            X: an (n,d) array containing the data examples on which to predict.

        Returns:
            h(X) --- the model predictions for X.
        """

        return np.max(X @ self.parameters[0].T, 0) @ self.parameters[1].T
