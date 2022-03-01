"""
Regularizers for training neural networks by convex reformulation.
"""


class Regularizer:
    """Base class for all regularizers."""

    def __init__(self, lam: float):
        """Initialize the squared-error loss function."""
        self.lam = lam


class NeuronGL1(Regularizer):
    """A neuron-wise group-L1 regularizer.

    This regularizer produces neuron sparsity in the final model, meaning that some neurons will be completely inactive after training.
    The regularizer has for the form,

    .. math:: R(U) = \\lambda \\sum_{i = 1}^p \\|U_i\\|_2,

    where :math:`\\lambda` is the regularization strength.

    Attributes:
        lam: the regularization strength.
    """


class FeatureGL1(Regularizer):
    """A feature-wise group-L1 regularizer.

    This regularizer produces feature sparsity in the final model, meaning that some features will not be used after training.
    The regularizer has for the form,

    .. math:: R(U) = \\lambda \\sum_{i = 1}^d \\|U_{\\cdot, i}\\|_2,

    where :math:`\\lambda` is the regularization strength.

    Attributes:
        lam: the regularization strength.
    """


class L2(Regularizer):
    """The standard squared-L2 norm regularizer, sometimes called weight-decay.

    The regularizer has for the form,

    .. math:: R(U) = \\lambda \\sum_{i = 1}^p \\|U_{i}\\|^2_2,

    where :math:`\\lambda` is the regularization strength.

    Attributes:
        lam: the regularization strength.
    """
