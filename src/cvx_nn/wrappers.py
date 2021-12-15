"""
Convenience functions creating and optimizing convex neural networks.
"""
import logging
from typing import Optional, Literal, cast, List

import torch
import numpy as np

import lab

from cvx_nn.models import (
    sign_patterns,
    ConvexMLP,
    AL_MLP,
    ConvexLassoNet,
    AL_LassoNet,
    GroupL1Regularizer,
    L2Regularizer,
    LassoNetConstraint,
    is_compatible,
    get_nc_formulation,
)

from cvx_nn.datasets import unitize_features
from cvx_nn.initializers import get_initializer
from cvx_nn.methods import (
    OptimizationProcedure,
    IterativeOptimizationProcedure,
    DoubleLoopProcedure,
    LinearSolver,
    FISTA,
    AugmentedLagrangian,
    GradientNorm,
    ConstrainedOptimality,
    QuadraticBound,
    MultiplicativeBacktracker,
    Lassplore,
)
from cvx_nn.prox import GroupL1, HierProx

# Constants #

from lab import CPU, CUDA, FLOAT32, FLOAT64, NUMPY, TORCH
from cvx_nn.initializers import ZERO, RANDOM, GATES, LEAST_SQRS

# regularizers
L2 = "l2"
GL1 = "group_l1"
FeatureGL1 = "feature_gl1"
LNConstraint = "lasso_net_constraint"

# exposed types
REGULARIZERS = [L2, GL1, FeatureGL1]
INITIALIZATIONS = [ZERO, RANDOM, GATES, LEAST_SQRS]
DEVICES = [CPU, CUDA]
PRECISIONS = [FLOAT32, FLOAT64]

# exposed problem formulations
GReLU_MLP = "grelu_mlp"
ReLU_MLP = "relu_mlp"

GReLU_LN = "grelu_lasso_net"
ReLU_LN = "relu_lasso_net"

FORMULATIONS = [GReLU_MLP, GReLU_LN, ReLU_MLP, ReLU_LN]


# ========================
# ==== Logging Helper ====
# ========================


def _get_logger(
    name: str, verbose: bool = False, debug: bool = False, log_file: str = None
) -> logging.Logger:
    """Construct a logging.Logger instance with an appropriate configuration.
    :param name: name for the Logger instance.
    :param verbose: (optional) whether or not the logger should print verbosely (ie. at the INFO level).
        Defaults to False.
    :param debug: (optional) whether or not the logger should print in debug mode (ie. at the DEBUG level).
        Defaults to False.
    :param log_file: (optional) path to a file where the log should be stored. The log is printed to stdout when 'None'.
    :returns: instance of logging.Logger.
    """

    level = logging.WARNING
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO

    logging.basicConfig(level=level, filename=log_file)
    logger = logging.getLogger(name)
    logging.root.setLevel(level)
    logger.setLevel(level)
    return logger


# =========================================
# ==== Default Optimization Procedures ====
# =========================================


def _default_ridge_solver(max_iters, grad_tol) -> OptimizationProcedure:
    preconditioner = None
    linear_solver = LinearSolver("lsmr", max_iters, grad_tol, preconditioner)
    return OptimizationProcedure(linear_solver)


def _default_relaxation_solver(
    max_iters, grad_tol, reg_type, reg_strength, M
) -> OptimizationProcedure:
    term_criterion = GradientNorm(grad_tol)
    if reg_type == GL1:
        prox = GroupL1(reg_strength)
    elif reg_type == FeatureGL1:
        prox = GroupL1(reg_strength, group_by_feature=True)
    elif reg_type == LNConstraint:
        prox = HierProx(M)

    optimizer = FISTA(
        10.0,
        QuadraticBound(),
        MultiplicativeBacktracker(beta=0.8),
        Lassplore(alpha=1.25, threshold=5.0),
        ls_type="fista",
        prox=prox,
        restart_rule="gradient_mapping",
    )

    return IterativeOptimizationProcedure(
        optimizer,
        max_iters,
        term_criterion,
        name="fista",
        divergence_check=True,
        log_freq=25,
    )


def _default_constrained_solver(
    max_dual_iters,
    max_subprob_iters,
    max_primal_iters,
    grad_tol,
    constraint_tol,
    reg_type,
    reg_strength,
    M,
) -> OptimizationProcedure:
    inner_term_criterion = GradientNorm(grad_tol)
    outer_term_criterion = ConstrainedOptimality(grad_tol, constraint_tol)

    if reg_type == GL1:
        prox = GroupL1(reg_strength)
    elif reg_type == FeatureGL1:
        prox = GroupL1(reg_strength, group_by_feature=True)
    elif reg_type == LNConstraint:
        prox = HierProx(M)

    inner_optimizer = FISTA(
        10.0,
        QuadraticBound(),
        MultiplicativeBacktracker(beta=0.8),
        Lassplore(alpha=1.25, threshold=5.0),
        ls_type="fista",
        prox=prox,
        restart_rule="gradient_mapping",
    )

    outer_optimizer = AugmentedLagrangian(
        use_delta_init=True,
        subprob_tol=grad_tol,
    )

    return DoubleLoopProcedure(
        inner_optimizer,
        outer_optimizer,
        max_subprob_iters,
        max_dual_iters,
        inner_term_criterion,
        outer_term_criterion,
        max_total_iters=max_primal_iters,
        name="al",
        divergence_check=False,
        log_freq=25,
    )


# ================================
# ==== Backend Configuration  ====
# ================================


def _configure_backend(model, device, dtype, backend, seed):
    # infer device to use from model parameters
    if device is None:
        if model is not None:
            device = next(model.parameters()).device
        elif backend == TORCH and torch.cuda.is_available():
            device = CUDA
        else:
            device = CPU

    device = cast(str, device)
    if backend is None:
        backend = TORCH if CUDA in device else NUMPY

    # set backend settings.
    lab.set_backend(backend)
    lab.set_device(device)
    lab.set_dtype(dtype)
    lab.set_seeds(seed)


# ======================
# ==== Process Data ====
# ======================


def _transform_weights(model_weights, column_norms):
    return model_weights / column_norms


def _untransform_weights(model_weights, column_norms):
    return model_weights * column_norms


def _process_data(X_train, y_train, X_test, y_test, unitize_data_cols):

    train_set = (np.array(X_train.tolist()), np.array(y_train.tolist()))
    test_set = (
        (np.array(X_test.tolist()), np.array(y_test.tolist()))
        if X_test is not None
        else train_set
    )

    column_norms = None
    if unitize_data_cols:
        train_set, test_set, column_norms = unitize_features(train_set, test_set, True)
        column_norms = lab.tensor(column_norms, dtype=lab.get_dtype())

    X_train, y_train = lab.all_to_tensor(train_set, dtype=lab.get_dtype())
    X_test, y_test = lab.all_to_tensor(test_set, dtype=lab.get_dtype())

    # add extra target dimension if necessary
    if len(y_train.shape) == 1:
        y_train = lab.expand_dims(y_train, axis=1)
        y_test = lab.expand_dims(y_test, axis=1)

    return X_train, y_train, (X_test, y_test), column_norms


# ============================
# ==== Build Convex Model ====
# ============================


def _construct_convex_model(
    model,
    X_train,
    y_train,
    max_patterns,
    reg_type,
    reg_strength,
    rng,
    seed,
    logger,
    formulation,
    initialization,
    M,
):
    n, d = X_train.shape
    c = y_train.shape[1]

    if model is not None:
        first_layer = cast(torch.nn.Linear, next(model.children()))
        max_patterns = first_layer.out_features

    D, U = sign_patterns.get_sign_patterns(
        X_train, {"name": "sampler", "seed": seed, "n_samples": max_patterns}
    )

    regularizer = None
    if reg_type == L2:
        regularizer = L2Regularizer(reg_strength)
    elif reg_type == GL1:
        regularizer = GroupL1Regularizer(reg_strength)
    elif reg_type == FeatureGL1:
        regularizer = GroupL1Regularizer(reg_strength, group_by_feature=True)
    elif reg_type == LNConstraint:
        regularizer = LassoNetConstraint(M)

    if formulation == GReLU_MLP:
        convex_model = ConvexMLP(d, D, U, "einsum", regularizer=regularizer, c=c)
    elif formulation == ReLU_MLP:
        convex_model = AL_MLP(
            d,
            D,
            U,
            "einsum",
            100,
            regularizer=regularizer,
            c=c,
        )
    elif formulation == GReLU_LN:
        convex_model = ConvexLassoNet(
            d, D, U, "einsum", regularizer=regularizer, c=c, gamma=reg_strength
        )
    elif formulation == ReLU_LN:
        convex_model = AL_LassoNet(
            d,
            D,
            U,
            "einsum",
            100,
            regularizer=regularizer,
            c=c,
            gamma=reg_strength,
        )
    else:
        raise ValueError(f"Problem formulation {formulation} not recognized!")

    # get model initializer
    initializer = get_initializer(
        logger, rng, (X_train, y_train), {"name": initialization}
    )

    return convex_model, initializer


# ==================================
# ==== Optimize Neural Network =====
# ==================================


def optimize(
    # data
    X_train: lab.Tensor,
    y_train: lab.Tensor,
    X_test: Optional[lab.Tensor] = None,
    y_test: Optional[lab.Tensor] = None,
    # metrics
    train_metrics: List = [],
    test_metrics: List = [],
    additional_metrics: List = [],
    # modeling
    max_patterns: Optional[int] = None,
    formulation: str = GReLU_MLP,
    model: Optional[torch.nn.Module] = None,
    warm_start: Optional[ConvexMLP] = None,
    reg_type: str = GL1,
    reg_strength: float = 0.001,
    M: float = 1.0,
    # optimization
    max_primal_iters: int = 10000,
    max_dual_iters: int = 10000,
    max_subprob_iters: int = 2000,
    grad_tol: float = 1e-6,
    constraint_tol: float = 1e-6,
    initialization: str = ZERO,
    unitize_data_cols: bool = True,
    # backend parameters
    backend: Optional[str] = None,
    device: Optional[str] = None,
    dtype: str = FLOAT32,
    # etc
    return_convex_form: bool = False,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
    log_file: Optional[str] = None,
    seed: int = 650,
):
    """Use convex optimization to fit a two-layer neural network to a given data set.
    :param X_train: the (n,d) matrix of training examples.
    :param y_train: the (n,1) vector of training targets.
    :param X_test: (optional) the (t,d) matrix of test examples.
    :param y_test: (optional) the (t,1) vector of test targets.
    :param train_metrics: (optional) a list of training-set metrics to record. Objective and gradient norm are always recorded.
        Valid options are: "objective" -- the optimization objective with constraint penalty terms. For ReLU networks, this is
                                            the augmented Lagrangian.
                           "base_objective": the training objective **without** constraint penalty terms. This is usually
                                            squared error + regularization.
                           "grad_norm": 2-norm of the minimum-norm subgradient of the optimization objective **with**
                                            penalty terms *or* 2- norm of the gradient mapping if the problem is constrained.
                           "accuracy": accuracy.
                           "squared_error": average squared error.
                           "constraint_gaps": 2-norm of constraint violations.
                           "lagrangian_grad": 2-norm of the (primal) subgradient of the Lagrangian function.
    :param test_metrics: (optional) a list of test-set metrics to record. Defaults to no test-set metrics. See 'train_metrics'
        for valid options.
    :param additional_metrics: (optional) a list of "additional" metrics to record. Defaults to no additional metrics.
        Valid options are: "time_stamp": time between iterations.
                           "total_neurons": total number of neurons in the model.
                           "feature_sparsity": proportion of features which are *not* used by the model
                                (ie. all weights are exactly zero for those features).
                           "active_features": number of features used by the model.
                           "group_sparsity": proportion of neurons which are *not* used by the model
                                (ie. all weights are exactly zero for those neurons).
                            "sparsity": proportion of weights are which zero.
                            "num_backtracks": number of backtracking steps required in the last iteration.
                            "sp_success": whether or not the line-search succeeded.
                            "step_size": the step-size after the last iteration.
    :param max_patterns: (optional) the maximum number of max_patterns to use in the convex formulation.
        The arguments 'max_patterns', 'model', and 'warm_start' are mutually exclusive; exactly one must be specified.
    :param formulation: (optional) the problem formulation to solve. Defaults to two-layer MLP with Gated ReLU activations.
        Valid options are: 'grelu_mlp': two-layer MLP with Gated ReLU activations.
                           'relu_mlp': two-layer MLP with ReLU activations.
                           'grelu_lasso_net': two-layer LassoNet with Gated ReLU activations.
                           'relu_lasso_net': two-layer LassoNet with ReLU activations.
    :param model: (optional) a torch.nn.Module instance corresponding to the neural network to be optimized.
        Only two architectures are permitted: (Linear, ReLU, Linear) or (GatedReluLayer, Linear).
    :param warm_start: (optional) a convex neural network from which to warm-start the optimization procedure.
        Mutual exclusive with 'model' and 'max_patterns'.
    :param reg_type: (optional) the type of regularization to use. Default is 'group_l1'.
        Valid options are: 'l2': classic squared 2-norm or "weight-decay" penalty.
                           'group_l1': a group L1 or "Group LASSO" penalty where weights are group by neuron.
                           'feature_gl1': a group L1 or "Group LASSO" penalty where weights are grouped by feature.
                           'lasso_net_constraint': **only** for LassoNet models.
    :param reg_strength: (optional) the strength of the regularization term (ie. lambda).
    :param M: (optional) scaling for LassoNet constraints. Defaults to 1.0. Only used when
        'formulation' is one of 'grelu_lasso_net', 'relu_lasso_net'.
    :param max_primal_iters: (optional) the maximum number of iterations to run the primal optimization method.
    :param max_dual_iters: (optional) the maximum number of iterations to run a dual optimization method.
        Only used when 'formulation' is one of 'relu_mlp' or 'relu_lasso_net'.
    :param max_subprob_iters: (optional) the maximum number of iterations to run when solving any sub-problem.
        Only used when 'formulation' is one of 'relu_mlp' or 'relu_lasso_net'.
    :param grad_tol: (optional) the tolerance for the first-order convergence criterion.
    :param constraint_tol: (optional) the tolerance for violation of the constraints.
        Only used when 'formulation' is one of 'relu_mlp' or 'relu_lasso_net'.
    :param initialization: (optional) the initialization strategy to use. Valid options are
        'zero' -- initialize at zero, 'least_squares' -- initialize at least-squares or
        ridge-regression solution, 'random' --- initialize at a point drawn from the standard
        normal distribution.
    :param unitize_data_cols: (optional) whether or not to unitize the columns of the data matrix
        before optimization. This improves the conditioning of the problem and is useful for
        optimization, but changes the scale of the regularization parameter.
    :param backend: (optional) the linear-algebra back-end to use. Valid options are 'torch' or 'numpy'.
    :param device: (optional) the device to run linear-algebra computations on. Valid options are
        'cpu', 'cuda', or a specific cuda device. This parameter is ignored when using the 'numpy' back-end.
    :param dtype: (optional) the data type to use during optimization. Valid options are 'float32' or 'float64'.
        'float32' can be much faster, but increases numerical error.
    :param return_convex_form: (optional) whether or not to return the convex formulation of the model.
        By default, a non-convex neural network is returned.
    :param verbose: (optional) whether or not to print verbosely during optimization.
    :param logger: (optional) a logging instance to use.
    :param log_file: (optional) a file path where log informations should be stored.
    :param seed: (optional) the random seed to use.
    """

    # Check for Conflicts #
    if X_test is not None and y_test is None:
        raise ValueError("`y_test` must not be None when `X_test` is provided.")

    if y_test is not None and X_test is None:
        raise ValueError("`X_test` must not be None when `y_test` is provided.")

    if model is not None and max_patterns is not None:
        raise ValueError(
            "Cannot infer the number of patterns to use when both `model` and `max_patterns` are provided."
        )
    elif model is None and max_patterns is None and warm_start is None:
        raise ValueError(
            "Need one of `model`, `max_patterns` or `warm_start` to infer the number of patterns to use in the convex formulation."
        )

    # check model architecture for compatibility.
    if model is not None:
        if not is_compatible(model):
            raise ValueError(
                "Invalid model architecture provided. Only architectures of the form [torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear] are supported."
            )

    if formulation not in FORMULATIONS:
        raise ValueError(
            f"Problem formulation {formulation} not recognized! Valid formulations are {FORMULATIONS}."
        )

    # Initialize Logger #
    if logger is None:
        logger = _get_logger("cvx_nn", verbose, False, log_file)

    if formulation in [GReLU_LN, ReLU_LN] and reg_type != LNConstraint:
        reg_type = LNConstraint
        logger.warning("Overriding regularization type to LassoNet penalty.")

    # Configure Backend #
    _configure_backend(model, device, dtype, backend, seed)

    rng = lab.np_rng

    # Process Data #

    logger.info("Processing data.")
    X_train, y_train, test_set, column_norms = _process_data(
        X_train, y_train, X_test, y_test, unitize_data_cols
    )
    train_set = (X_train, y_train)

    # Initialize Model #
    logger.info("Constructing convex model.")

    convex_model, initializer = _construct_convex_model(
        model,
        X_train,
        y_train,
        max_patterns,
        reg_type,
        reg_strength,
        rng,
        seed,
        logger,
        formulation,
        initialization,
        M,
    )
    if warm_start is not None:
        logger.info("Warm-starting from existing model...")
        starting_weights = _untransform_weights(warm_start.weights, column_norms)
        convex_model.U = warm_start.U
        convex_model.D = warm_start.D
        convex_model.p = warm_start.p
        convex_model.set_weights(starting_weights)

        if formulation in [ReLU_LN, ReLU_MLP]:
            assert isinstance(warm_start, AL_MLP)

            convex_model.e_multipliers = warm_start.e_multipliers
            convex_model.i_multipliers = warm_start.i_multipliers
            convex_model.orthant = warm_start.orthant
            convex_model.delta = warm_start.delta

        # overwrite initializer to be identity
        initializer = get_initializer(logger, rng, (X_train, y_train), {})

    # Configure Optimization Procedure #

    logger.info("Preparing optimizer.")

    if formulation in [GReLU_MLP, GReLU_LN]:
        # use fast iterative solver for least-squares problems.
        if reg_type == L2:
            opt_procedure = _default_ridge_solver(max_primal_iters, grad_tol)
        # use FISTA
        elif reg_type in [GL1, FeatureGL1, LNConstraint]:
            opt_procedure = _default_relaxation_solver(
                max_primal_iters, grad_tol, reg_type, reg_strength, M
            )
        else:
            raise ValueError(f"Regularization type {reg_type} not supported.")
    elif formulation in [ReLU_MLP, ReLU_LN]:
        opt_procedure = _default_constrained_solver(
            max_dual_iters,
            max_subprob_iters,
            max_primal_iters,
            grad_tol,
            constraint_tol,
            reg_type,
            reg_strength,
            M,
        )

    # Solve Optimization Problem #

    logger.info("Optimizing convex model.")
    exit_status, convex_model, metrics = opt_procedure(
        logger,
        convex_model,
        initializer,
        train_set,
        test_set,
        (["objective", "grad_norm"] + train_metrics, test_metrics, additional_metrics),
    )

    # transform model back to original data space.
    if unitize_data_cols:
        convex_model.weights = _transform_weights(convex_model.weights, column_norms)

    if return_convex_form:
        return convex_model, metrics

    implementation = "manual"
    if model is not None:
        implementation = "torch"

    return_model = get_nc_formulation(convex_model, implementation, remove_sparse=True)

    return return_model, metrics
