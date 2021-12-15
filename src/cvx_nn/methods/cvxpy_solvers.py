"""
Interfaces for CVXPY (https://www.cvxpy.org/index.html) solvers.
"""

from typing import Dict, Any, Tuple

from logging import root, INFO
import numpy as np
import cvxpy as cp

import lab

from cvx_nn.models import (
    Model,
    ConvexMLP,
    AL_MLP,
    ConvexLassoNet,
    AL_LassoNet,
)
from cvx_nn.methods.external_solver import ExternalSolver


class CVXPYSolver(ExternalSolver):
    """
    Interface for solvers based on the CVXPY DSL.
    """

    def __init__(self, solver: str = "ecos"):
        """
        :param solver: the solver to use with CVXPY.
        """

        # save the desired solver
        if solver == "ecos":
            self.solver = cp.ECOS_BB
        elif solver == "cvxopt":
            self.solver = cp.CVXOPT
        elif solver == "scs":
            self.solver = cp.SCS
        # note: these are commercial solvers requiring a licence.
        elif solver == "gurobi":
            self.solver = cp.GUROBI
        elif solver == "mosek":
            self.solver = cp.MOSEK
        else:
            raise ValueError(f"CVXPY solver {solver} not recognized!")

    def __call__(
        self, model: Model, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Model, Dict[str, Any]]:
        """Call the CVXPY solver to fit the model."""

        raise NotImplementedError("A CVXPY-based solver must implement '__call__'.")


class RelaxedMLPSolver(CVXPYSolver):
    """
    Solver for the relaxed convex MLP training problem,
        min_W (1/2)||sum_i D_i X W_i - y||^2 + lam * sum_i ||W_i||_2,
    which corresponds to a MLP with "gated" ReLU activations.
    """

    def __call__(
        self, model: ConvexMLP, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Model, Dict[str, Any]]:
        """Use CVXPY to solve the relaxed optimization problem.
        :param model: the model to fit.
        :param X: the training inputs/features.
        :param y: the training targets/classes.
        :returns: (model, status_dict) --- the updated model and a dictionary describing the optimizer's exit state.
        """

        # lookup problem dimensions
        n, d = X.shape
        c = y.shape[1]
        p = model.D.shape[1]

        # Cast variables to NumPy arrays.
        X_np = lab.to_np(X)
        y_np = lab.to_np(y)
        D_np = lab.to_np(model.D)

        # create optimization variables
        W = cp.Variable((p * c, d))

        # define optimization objective
        # note that this is declarative and loops have no impact on performance.
        loss = 0
        for i in range(c):
            loss += cp.sum_squares(
                cp.sum(cp.multiply(D_np, X_np @ W[i * p : (i + 1) * p].T), axis=1)
                - y_np[:, i]
            ) / (2 * n * c)

        if model.regularizer is not None and model.regularizer.lam > 0.0:
            lam = model.regularizer.lam
            loss = loss + lam * cp.mixed_norm(W, p=2, q=1)

        objective = cp.Minimize(loss)

        problem = cp.Problem(objective)

        # infer verbosity of sub-solver.
        verbose = root.level <= INFO
        # solve the optimization problem
        problem.solve(solver=self.solver, verbose=verbose)

        # extract solution
        model.weights = lab.tensor(W.value, dtype=lab.get_dtype()).reshape(c, p, d)

        # extract solver information
        exit_status = {
            "success": problem.status == cp.OPTIMAL,
            "status": problem.status,
            "final_val": problem.value,
        }

        return model, exit_status


class OrthantConstrainedMLPSolver(CVXPYSolver):
    """
    Solver for the convex MLP training problem,
        min_W (1/2)||sum_i D_i X W_i - y||^2 + lam * sum_i ||W_i||_2
            s.t. (2 D_i - I) X W_i >= 0, i in [p],
    which corresponds to a MLP with ReLU activations.
    """

    def __call__(
        self, model: AL_MLP, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Model, Dict[str, Any]]:
        """Use CVXPY to solve the orthant-constrained optimization problem.
        :param model: the model to fit.
        :param X: the training inputs/features.
        :param y: the training targets/classes.
        :returns: (model, status_dict) --- the updated model and a dictionary describing the optimizer's exit state.
        """

        # lookup problem dimensions
        n, d = X.shape
        c = y.shape[1]
        p = model.D.shape[1]

        # Cast variables to NumPy arrays.
        X_np = lab.to_np(X)
        y_np = lab.to_np(y)
        D_np = lab.to_np(model.D)

        # create optimization variables
        U = cp.Variable((p * c, d))
        V = cp.Variable((p * c, d))
        W = U - V

        # define optimization objective
        loss = 0
        for i in range(c):
            loss += cp.sum_squares(
                cp.sum(cp.multiply(D_np, X_np @ W[i * p : (i + 1) * p].T), axis=1)
                - y_np[:, i]
            ) / (2 * n * c)

        if model.regularizer is not None:
            lam = model.regularizer.lam
            loss = loss + lam * (
                cp.mixed_norm(U, p=2, q=1) + cp.mixed_norm(V, p=2, q=1)
            )

        objective = cp.Minimize(loss)

        # define orthant constraints
        A = 2 * D_np - np.ones_like(D_np)

        constraints = [
            cp.multiply(A, X_np @ U[i * p : (i + 1) * p].T) >= 0 for i in range(c)
        ]
        constraints += [
            cp.multiply(A, X_np @ V[i * p : (i + 1) * p].T) >= 0 for i in range(c)
        ]

        problem = cp.Problem(objective, constraints)

        verbose = root.level <= INFO
        # solve the optimization problem
        problem.solve(solver=self.solver, verbose=verbose)

        # extract solution
        model.weights = lab.stack(
            [
                lab.tensor(U.value, dtype=lab.get_dtype()).reshape(c, p, d),
                lab.tensor(V.value, dtype=lab.get_dtype()).reshape(c, p, d),
            ]
        )

        # extract solver information
        exit_status = {
            "success": problem.status == cp.OPTIMAL,
            "status": problem.status,
            "final_val": problem.value,
        }

        return model, exit_status


class RelaxedLassoNetSolver(CVXPYSolver):
    """"""

    def __call__(
        self, model: ConvexLassoNet, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Model, Dict[str, Any]]:
        """Use CVXPY to solve the relaxed LassoNet optimization problem without group-L1 regularization.
        :param model: the model to fit.
        :param X: the training inputs/features.
        :param y: the training targets/classes.
        :returns: (model, status_dict) --- the updated model and a dictionary describing the optimizer's exit state.
        """

        # lookup problem dimensions
        n, d = X.shape
        c = y.shape[1]
        p = model.D.shape[1]

        if c > 1:
            raise ValueError(
                "Currently we can only support LassoNet models with one target."
            )
        y = lab.squeeze(y)

        # Cast variables to NumPy arrays.
        X_np = lab.to_np(X)
        y_np = lab.to_np(lab.squeeze(y))
        D_np = lab.to_np(model.D)

        # create optimization variables
        W = cp.Variable((p, d))

        # skip connections
        beta = cp.Variable((2, d))

        # define optimization objective
        residual = (
            X_np @ (beta[0] - beta[1])
            + cp.sum(cp.multiply(D_np, X_np @ W.T), axis=1)
            - y_np
        )

        loss = cp.sum_squares(residual) / (2 * n)

        if model.gamma is not None and model.gamma > 0.0:
            loss = loss + model.gamma * cp.sum(beta)

        objective = cp.Minimize(loss)

        # create non-negativity and magnitude constraints.
        constraints = [beta >= 0]

        if model.regularizer is not None:
            constraints = constraints + [
                cp.abs(W[:, i]) <= model.regularizer.M * (beta[0, i] + beta[1, i])
                for i in range(d)
            ]

        problem = cp.Problem(objective, constraints)

        # solve the optimization problem
        verbose = root.level <= INFO
        problem.solve(solver=self.solver, verbose=verbose)

        # extract solution
        model.weights = lab.expand_dims(
            lab.concatenate(
                [
                    lab.tensor(W.value, dtype=lab.get_dtype()),
                    lab.tensor(beta.value, dtype=lab.get_dtype()),
                ],
                axis=0,
            ),
            axis=0,
        )

        # extract solver information
        exit_status = {
            "success": problem.status == cp.OPTIMAL,
            "status": problem.status,
            "final_val": problem.value,
        }

        return model, exit_status


class OrthantConstrainedLassoNetSolver(CVXPYSolver):
    """"""

    def __call__(
        self, model: AL_LassoNet, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Model, Dict[str, Any]]:
        """Use CVXPY to solve the LassoNet optimization problem with orthant constraints.
        :param model: the model to fit.
        :param X: the training inputs/features.
        :param y: the training targets/classes.
        :returns: (model, status_dict) --- the updated model and a dictionary describing the optimizer's exit state.
        """

        # lookup problem dimensions
        n, d = X.shape
        p = model.D.shape[1]
        c = y.shape[1]

        if c > 1:
            raise ValueError(
                "Currently we can only support LassoNet models with one target."
            )
        y = lab.squeeze(y)

        # Cast variables to NumPy arrays.
        X_np = lab.to_np(X)
        y_np = lab.to_np(lab.squeeze(y))
        D_np = lab.to_np(model.D)

        # create optimization variables
        U = cp.Variable((p, d))
        V = cp.Variable((p, d))
        W = U - V

        # skip connections
        beta = cp.Variable((2, d))

        # define optimization objective
        residual = (
            X_np @ (beta[0] - beta[1])
            + cp.sum(cp.multiply(D_np, X_np @ W.T), axis=1)
            - y_np
        )

        loss = cp.sum_squares(residual) / (2 * n)

        if model.gamma is not None and model.gamma > 0.0:
            loss = loss + model.gamma * cp.sum(beta)

        objective = cp.Minimize(loss)

        # create orthant, non-negativity and magnitude constraints.
        A = 2 * D_np - np.ones_like(D_np)

        constraints = [
            cp.multiply(A, X_np @ U.T) >= 0,
            cp.multiply(A, X_np @ V.T) >= 0,
            beta >= 0,
        ]

        if model.regularizer is not None:
            constraints = constraints + [
                cp.abs(W[:, i]) <= model.regularizer.M * (beta[0, i] + beta[1, i])
                for i in range(d)
            ]

        problem = cp.Problem(objective, constraints)

        # solve the optimization problem
        verbose = root.level <= INFO
        problem.solve(solver=self.solver, verbose=verbose)

        # extract solution
        skip_weights = lab.tensor(beta.value, dtype=lab.get_dtype())
        stacked_weights = lab.stack(
            [
                lab.concatenate(
                    [lab.tensor(U.value, dtype=lab.get_dtype()), skip_weights[:1]],
                    axis=0,
                ),
                lab.concatenate(
                    [lab.tensor(V.value, dtype=lab.get_dtype()), skip_weights[1:]],
                    axis=0,
                ),
            ],
        )
        model.weights = lab.expand_dims(stacked_weights, axis=1)

        # extract solver information
        exit_status = {
            "success": problem.status == cp.OPTIMAL,
            "status": problem.status,
            "final_val": problem.value,
        }

        return model, exit_status
