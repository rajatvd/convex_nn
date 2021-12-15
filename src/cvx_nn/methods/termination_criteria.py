"""
Termination criteria for iterative optimization methods.
"""
from typing import Dict, Any, Optional

import lab

from cvx_nn.models.model import Model

# constants

TOL = 1e-6

# classes


class TerminationCriterion:

    """Base class for termination criteria."""

    tol: float

    def __init__(self, tol: float = TOL):
        """
        :param tol: the tolerance for determining a first-order stationary point.
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
        """Evaluate the termination criterion given a model and a dataset."""
        raise NotImplementedError("Termination criteria must implement __call__!")


class GradientNorm(TerminationCriterion):

    """Termination criterion based on the norm of the gradient."""

    def __call__(
        self,
        model: Model,
        X: lab.Tensor,
        y: lab.Tensor,
        objective: Optional[lab.Tensor] = None,
        grad: Optional[lab.Tensor] = None,
    ) -> bool:
        """Determine if the gradient is small enough (according to self.tol) to constitute
        a first-order stationary point.
        """
        if grad is None:
            grad = model.grad(X, y)

        return lab.sum(grad ** 2) <= self.tol


# TODO: replace with projected gradient mapping
class StepLength(TerminationCriterion):

    """Termination criterion based on the length of the most recent step."""

    previous_weights: Optional[lab.Tensor] = None

    def __call__(
        self,
        model: Model,
        X: lab.Tensor,
        y: lab.Tensor,
        objective: Optional[lab.Tensor] = None,
        grad: Optional[lab.Tensor] = None,
    ) -> bool:
        """Determine if the last step length is small enough (according to self.tol) to deduce
        that the method has converged.
        """
        if self.previous_weights is None:
            self.previous_weights = model.weights
            return False

        step = model.weights - self.previous_weights
        self.previous_weights = model.weights

        return lab.sqrt(lab.sum(step ** 2)) <= self.tol


class ConstrainedOptimality(TerminationCriterion):

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
        """Determine if the current point is both feasible and a first-order optimal point.
        First-order optimality is checked by evaluating the norm of the  gradient mapping.
        """
        if grad is None:
            grad = model.grad(X, y)

        e_gap, i_gap = model.constraint_gaps(X)
        if lab.sum(e_gap ** 2 + lab.smax(i_gap, 0) ** 2) <= self.constraint_tol:
            return lab.sum(grad ** 2) <= self.grad_tol

        return False


# index


def get_criterion(config: Dict[str, Any]) -> TerminationCriterion:
    """Load a termination criterion by name using the passed configuration parameters.
    :param config: configuration object specifying the termination criterion.
    :returns: an instance of TerminationCriterion which can be used to determine if an optimizer as converged.
    """
    name = config.get("name", None)

    if name is None:
        raise ValueError("Termination criterion must have name!")
    elif name == "grad_norm":
        return GradientNorm(config.get("tol", TOL))
    elif name == "step_length":
        return StepLength(config.get("tol", TOL))
    elif name == "constrained_opt":
        return ConstrainedOptimality(
            config.get("grad_tol", TOL), config.get("constraint_tol", TOL)
        )
    else:
        raise ValueError(f"Termination criterion {name} not recognized!")
