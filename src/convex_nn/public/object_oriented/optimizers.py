"""
Optimization methods for training neural networks by convex reformulation.

Notes:
    We only support the squared loss for the time being.
"""

from .models import (
    Model,
    ConvexReLU,
    NonConvexReLU,
    ConvexGatedReLU,
    NonConvexGatedReLU,
)


class Optimizer:
    """Base class for optimizers.

    Attributes:
        model: the model that should be optimized.
    """

    def __init__(self, model: Model):
        """Initialize the optimizer.

        Args:
            model: the model to be optimized. Note that it will be checked for compatibility with the optimizer.
        """
        self.model = Model


class RFISTA(Optimizer):
    """

    Attributes:
        model: the model that should be optimized.
        max_iters: the maximum number of iterations to run the optimization method.
        tol: the tolerance for terminating the optimization procedure early.
    """

    def __init__(self, model: Model, max_iters: int = 10000, tol: float = 1e-6):
        """Initialize the restarted FISTA (R-FISTA) optimizer.

        Args:
            model: the model to be optimized. Note that it will be checked for compatibility with the optimizer.
            max_iters: the maximum number of iterations to run the optimizer before exiting.
            tol: the tolerance for terminating the optimization procedure early.
                Specifically, the procedure will exit if the squared (L2) norm of the minimum-norm subgradient drops below tol.
        """

        if not isinstance(model, (ConvexGatedReLU, NonConvexGatedReLU)):
            raise ValueError(
                "The RFISTA optimization method can only be used to train Gated ReLU models."
            )

        super().__init__(model)
        self.max_iters = max_iters
        self.tol = tol


class AL(Optimizer):
    """Initialize an augmented Lagrangian method using R-FISTA as a sub-solver.

    Attributes:
        model: the model to be optimized. Note that it will be checked for compatibility with the optimizer.
        max_primal_iters: the maximum number of iterations to run the primal optimization method (i.e. R-FISTA) before exiting.
        max_dual_iters: the maximum number of dual updates that can be performed before terminating.
        tol: the tolerance for terminating the primal optimization procedure early.
        constraint_tol: the maximum violation of the constraints permitted.
    """

    def __init__(
        self,
        model: Model,
        max_primal_iters: int = 10000,
        max_dual_iters: int = 10000,
        tol: float = 1e-6,
        constraint_tol: float = 1e-6,
    ):
        """Initialize an augmented Lagrangian method using R-FISTA as a sub-solver.

        Args:
            model: the model to be optimized. Note that it will be checked for compatibility with the optimizer.
            max_primal_iters: the maximum number of iterations to run the primal optimization method (i.e. R-FISTA) before exiting.
            max_dual_iters: the maximum number of dual updates that can be performed before terminating.
            tol: the tolerance for terminating the primal optimization procedure early.
            constraint_tol: the maximum violation of the constraints permitted.
                The AL method will exit when the squared (L2) norm of the constraint violations is less than `constraint_tol` and
                the primal optimization method has converged according to `tol`.
        """
        self.model = Model

        if not isinstance(model, (ConvexReLU, NonConvexReLU)):
            raise ValueError(
                "The AL optimization method can only be used to train ReLU models."
            )


class ConeDecomposition(Optimizer):
    """ """

    def __init__(self, model: Model):
        """Initialize the optimizer."""
        raise NotImplementedError("Cone decompositions are not supported yet!")
