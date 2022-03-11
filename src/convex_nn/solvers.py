"""
Optimization methods for training neural networks by convex reformulation.

Notes:
    We only support the squared loss at the moment.

Todo:
    - Implement the cone decomposition optimizer for training ReLU model by (1) training a Gated ReLU model and (2) decomposing that model onto the conic difference.
    - Implement CVXPY solvers.
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

    def cpu_only(self) -> bool:
        return False


class RFISTA(Optimizer):
    """Accelerated proximal-gradient solver with line-search and restarts for Gated ReLU models.

    This optimizer solves the Gated ReLU training problem by directly solving the convex reformulation,

    .. math:: F(u) = L\\left(\\sum_{D_i \\in \\mathcal{D}} D_i X u_{i}), y\\right) + \\lambda R(u),

    where :math:`L` is a convex loss function, :math:`R` is a regularizer, and :math:`\\lambda` is the regularization strength.

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
            raise ValueError("R-FISTA can only be used to train Gated ReLU models.")

        super().__init__(model)
        self.max_iters = max_iters
        self.tol = tol


class AL(Optimizer):
    """Augmented Lagrangian (AL) method using R-FISTA as a sub-solver.

    This optimizer solves the convex re-formulation for ReLU networks by forming the "augmented Lagrangian",

    .. math:: \\mathcal{L}(v, w, \\gamma, \\xi) = F(v,w) + \\delta \\sum_{D_i} \\left[\\|(\\frac{\\gamma_i}{\\delta} - (2D_i - I)X v_i)_+\\|_2^2 + \\|(\\frac{\\xi_i}{\\delta} - (2D_i - I)X v_i)_+\\|_2^2 \\right],

    where :math:`\\delta > 0` is the penalty strength, :math:`(\\gamma, \\xi)` are the dual parameters, and

    .. math:: F(v,w) = L\\left(\\sum_{D_i \\in \\mathcal{D}} D_i X (v_{i} - w_{i}), y\\right) + \\lambda R(v, w),

    is the regularized training loss.
    The AL method alternates between the "primal" problem of minimizing :math:`\\mathcal{L}(v, w, \\gamma, \\xi)` in terms of :math:`v, w` and the dual problem of updating :math:`\\gamma, \\xi`.
    Only the dual parameters are guaranteed to converge, but :math:`v, w` often converge in practice.

    Attributes:
        model: the model to be optimized.
        max_primal_iters: the maximum number of iterations to run the primal optimization method (i.e. R-FISTA) before exiting.
        max_dual_iters: the maximum number of dual updates that can be performed before terminating.
        tol: the tolerance for terminating the primal optimization procedure early.
        constraint_tol: the maximum violation of the constraints permitted.
        delta: the penalty strength.
    """

    def __init__(
        self,
        model: Model,
        max_primal_iters: int = 10000,
        max_dual_iters: int = 10000,
        tol: float = 1e-6,
        constraint_tol: float = 1e-6,
        delta: float = 1000,
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
        if not isinstance(model, (ConvexReLU, NonConvexReLU)):
            raise ValueError("The AL optimizer can only be used to train ReLU models.")

        super().__init__(model)
        self.max_primal_iters = max_primal_iters
        self.max_dual_iters = max_dual_iters
        self.tol = tol
        self.constraint_tol = constraint_tol
        self.delta = delta


class ConeDecomposition(Optimizer):
    """Two-step method for approximately optimizing ReLU models.

    ConeDecomposition first solves the Gated ReLU problem using R-FISTA,

    .. math:: \\min_{u} L\\left(\\sum_{D_i \\in \\mathcal{D}} D_i X u_{i}), y\\right) + \\lambda R(u),

    and then decomposes the solution onto the Minkowski differences :math:`K_i - K_i` to approximate the ReLU training problem.
    The resulting solution is guaranteed to preserve the value of the loss :math:`L`, but can substantially blow-up the model norm.
    As such, it is only an approximation to the ReLU training problem when :math:`\\lambda > 0`.
    """

    def __init__(self, model: Model):
        """Initialize the optimizer."""
        raise NotImplementedError("ConeDecomposition is not supported yet.")


class LeastSquaresSolver(Optimizer):
    """Direct solver for the unregularized or L2-regularized Gated ReLU problem based on LSMR [1] or LSQR [2].

    This optimizer solves the (L2-regularized) Gated ReLU training problem by solving the convex reformulation,

    .. math:: F(u) = \\frac{1}{2}\\|\\sum_{D_i \\in \\mathcal{D}} D_i X u_{i} - y\\|_2^2 + \\lambda \sum_{D_i \\in \\mathcal{D}} \\|u_i\\|_2^2,

    using conjugate-gradient (CG) type methods. Only the squared-error loss-function is supported.
    Implementations of LSMR and LSQR are provided by
    `scipy.sparse.linalg <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_.
    See `SOL <https://web.stanford.edu/group/SOL/download.html>`_ for details on these methods.

    Attributes:
        model: the model that should be optimized.
        solver: the underlying CG-type solver to use.
        max_iters: the maximum number of iterations to run the optimization method.
        tol: the tolerance for terminating the optimization procedure early.

    Notes:
        This solver only supports computation on CPU. The user's choice of backend will be overridden if necessary.

    References:
        [1] D. C.-L. Fong and M. A. Saunders, LSMR: An iterative algorithm for sparse least-squares problems, SIAM J. Sci. Comput. 33:5, 2950-2971, published electronically Oct 27, 2011.

        [2] C. C. Paige and M. A. Saunders, LSQR: An algorithm for sparse linear equations and sparse least squares, TOMS 8(1), 43-71 (1982)
    """

    def __init__(
        self, model: Model, solver: str = "lsmr", max_iters: int = 1000, tol=1e-6
    ):
        """Initialize the least-squares solver.

        Args:
            model: the model to be optimized. Note that it will be checked for compatibility with the optimizer.
            solver: the underlying CG-type solver to use. Must be one of "LSMR" or "LSQR".
            max_iters: the maximum number of iterations to run the optimization method.
            tol: the tolerance for terminating the optimization procedure early.
        """
        if not isinstance(model, (ConvexGatedReLU, NonConvexGatedReLU)):
            raise ValueError(
                "LeastSquaresSolver can only be used to train Gated ReLU models."
            )

        super().__init__(model)

        self.solver = solver
        self.max_iters = max_iters
        self.tol = tol

    def cpu_only(self) -> bool:
        return True


class CVXPYSolver(Optimizer):
    """Solver based on CVXPY.

    Notes:
        This solver only supports computation on CPU. The user's choice of backend will be overridden if necessary.
    """

    def __init__(self, model: Model):
        """Initialize the optimizer."""
        raise NotImplementedError("CVXPY is not supported yet.")

    def cpu_only(self) -> bool:
        return True
