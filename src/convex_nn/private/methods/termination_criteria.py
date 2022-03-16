"""Termination criteria for optimization methods.

TODO:
    - Update `StepLength` to be the gradient mapping. The major
        difference is multiplication by a step-size.
"""

from typing import Optional

import lab

from convex_nn.private.models.model import Model


class TerminationCriterion:
    """Base class for termination criteria.

    A boolean function with one or more internal tuning parameters that returns `True`
    when an optimization procedure should terminate and `False` otherwise.

    Attributes:
        tol: a parameter controlling sensitivity of the termination criterion.
    """

    tol: float

    def __init__(self, tol: float):
        """
        Args:
            tol: a tolerance parameter controlling sensitivity of the termination criterion.
        """

        self.tol = tol

    def __call__(
        self,
        model: Model,
        X: lab.Tensor,
        y: lab.Tensor,
        objective: Optional[lab.Tensor] = None,
        grad: Optional[lab.Tensor] = None,
    ) -> bool:
        """Evaluate the termination criterion given a model and a dataset.

        The current objective value and gradient are optional parameters;
        these should be supplied if they have been pre-calculated for another
        purpose. Otherwise, the criterion will compute them as necessary.

        Args:
            model: the prediction model that is being optimized.
            X: a :math:`n \\times d` matrix of training examples.
            y: a :math:`n \\times c` matrix of training targets.
            objective: the current objective value.
                Provide only if already computed.
            grad: the current gradient of the objective.
                Provide only if already computed.

        Returns:
            Boolean indicating whether or not optimization should terminate.
        """
        raise NotImplementedError(
            "Termination criteria must implement __call__!"
        )


class GradientNorm(TerminationCriterion):
    """First-order optimality criterion.

    Terminate optimization if and only if the norm of minimum-norm subgradient
    is below a certain tolerance.
    """

    def __call__(
        self,
        model: Model,
        X: lab.Tensor,
        y: lab.Tensor,
        objective: Optional[lab.Tensor] = None,
        grad: Optional[lab.Tensor] = None,
    ) -> bool:
        """Terminate if gradient norm is sufficiently small.

        Determine if the norm of the minimum-norm sub-gradient (or gradient if
        the function is smooth) is small enough (according to self.tol) to constitute
        a first-order stationary point.

        The current objective value and gradient are optional parameters;
        these should be supplied if they have been pre-calculated for another
        purpose. Otherwise, the criterion will compute them as necessary.

        Args:
            model: the prediction model that is being optimized.
            X: a :math:`n \\times d` matrix of training examples.
            y: a :math:`n \\times c` matrix of training targets.
            objective: the current objective value.
                Provide only if already computed.
            grad: the current gradient of the objective.
                Provide only if already computed.

        Returns:
            Boolean indicating whether or not optimization should terminate.
        """

        if grad is None:
            grad = model.grad(X, y)

        return lab.sum(grad ** 2) <= self.tol


class StepLength(TerminationCriterion):
    """Criterion based on length of the most recent step.

    Terminate optimization if and only if the norm of the last step was
    below a certain tolerance.

    Attributes:
        previous_weights: the parameters of the model from the previous
            iteration. These are necessary to compute the length of the
            step.

    Notes:
        This criterion can significant increase the memory requirements
        of an optimization procedure due to the `previous_weights` attributes.
    """

    previous_weights: Optional[lab.Tensor] = None

    def __call__(
        self,
        model: Model,
        X: lab.Tensor,
        y: lab.Tensor,
        objective: Optional[lab.Tensor] = None,
        grad: Optional[lab.Tensor] = None,
    ) -> bool:
        """Terminate if step-length is sufficiently small.

        Determine if the norm of previous step is small enough (according to self.tol)
        to indicate the method has converged.

        The current objective value and gradient are not used.

        Args:
            model: the prediction model that is being optimized.
            X: a :math:`n \\times d` matrix of training examples.
            y: a :math:`n \\times c` matrix of training targets.
            objective: the current objective value. NOT USED.
            grad: the current gradient of the objective. NOT USED.

        Returns:
            Boolean indicating whether or not optimization should terminate.
        """
        if self.previous_weights is None:
            self.previous_weights = model.weights
            return False

        step = model.weights - self.previous_weights
        self.previous_weights = model.weights

        return lab.sqrt(lab.sum(step ** 2)) <= self.tol


class ConstrainedHeuristic(TerminationCriterion):
    """Terminate if the gradient and constraint violations are both small.

    A heuristic condition which terminates optimization if and only if the
    norm of minimum-norm
    subgradient is below a certain tolerance and the norm of the constraints violations is below
    a separate tolerance.

    This criterion is not the same as checking stationarity of the Lagrangian, but appears to work
    well in practice.
    """

    grad_tol: float
    constraint_tol: float

    def __init__(self, grad_tol: float = TOL, constraint_tol: float = TOL):
        """
        :param grad_tol: the tolerance for determining first-order optimality.
        :param constraint_tol: the tolerance for determining feasibility.
        """
        self.grad_tol = grad_tol
        self.constraint_tol = constraint_tol

    def __call__(
        self,
        model: Model,
        X: lab.Tensor,
        y: lab.Tensor,
        objective: Optional[lab.Tensor] = None,
        grad: Optional[lab.Tensor] = None,
    ) -> bool:
        """Determine if the current point is both feasible and a first-order
        optimal point.

        First-order optimality is checked by evaluating the norm of the
        gradient mapping.
        """
        if grad is None:
            grad = model.grad(X, y)

        e_gap, i_gap = model.constraint_gaps(X)
        if (
            lab.sum(e_gap ** 2 + lab.smax(i_gap, 0) ** 2)
            <= self.constraint_tol
        ):
            return lab.sum(grad ** 2) <= self.grad_tol

        return False


class LagrangianGradNorm(TerminationCriterion):
    grad_tol: float
