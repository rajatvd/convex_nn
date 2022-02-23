"""
Metrics that can be recorded while training models.
"""
import numpy as np


class Metrics(object):

    """Metrics collected while optimizing models.

    By default, only the objective, training time, and norm of the minimum-norm subgradient are collected.

    Attributes:
        objective: the optimization objective, including any constraint penalty terms.
        time: the cumulative wall-clock time.
        grad_norm: squared 2-norm of the minimum-norm subgradient of the optimization objective, including penalty terms.
        model_loss: the regularized loss of the model on the training set.
            This is the same as `objective` but does not include constraint penalty terms.
        constraint_gaps: 2-norm of constraint violations.
            Ignored for unconstrained problems.
        lagrangian_grad: 2-norm of the (primal) subgradient of the Lagrangian function.
            Ignored for unconstrained problems.
        train_accuracy: accuracy on the training set.
        train_mse: mean squared error on the training set.
        test_accuracy": accuracy on the test set.
        test_mse: means squared error on the test set.
        total_neurons: total number of neurons in the model.
        neuron_sparsity: proportion of neurons which are not used by the model
          (ie. all weights are exactly zero for those neurons).
        active_features: number of features used by the model.
        feature_sparsity: proportion of features which are not used by the model
          (ie. all weights are exactly zero for those features).
        weight_sparsity: proportion of weights are which zero.
        metric_freq: the frequency at which metrics should be collected in iterations.
        metrics_to_collect: internal dictionary specifying which metrics should be collected.
    """

    objective: np.ndarray
    grad_norm: np.ndarray
    time: np.ndarray
    model_loss: np.ndarray
    constraint_gaps: np.ndarray
    lagrangian_grad: np.ndarray
    train_accuracy: np.ndarray
    train_mse: np.ndarray
    test_accuracy: np.ndarray
    test_mse: np.ndarray
    total_neurons: np.ndarray
    neuron_sparsity: np.ndarray
    total_features: np.ndarray
    feature_sparsity: np.ndarray
    total_weights: np.ndarray
    weight_sparsity: np.ndarray

    def __init__(
        self,
        metric_freq: int = 25,
        objective: bool = True,
        grad_norm: bool = True,
        time: bool = True,
        model_loss: bool = False,
        constraint_gaps: bool = False,
        lagrangian_grad: bool = False,
        train_accuracy: bool = False,
        train_mse: bool = False,
        test_accuracy: bool = False,
        test_mse: bool = False,
        total_neurons: bool = False,
        neuron_sparsity: bool = False,
        total_features: bool = False,
        feature_sparsity: bool = False,
        total_weights: bool = False,
        weight_sparsity: bool = False,
    ):
        """Instantiate a metrics object by specifying which metrics should be collected during optimization.

        Args:
            metric_freq: the frequency at which to log metrics.
            objective: whether or not to log the optimization objective, including any constraint penalty terms.
            time: whether or not to log the cumulative wall-clock time.
            grad_norm: whether or not to log squared 2-norm of the minimum-norm subgradient of the optimization objective, including penalty terms.
            model_loss: whether or not to log the regularized loss of the model on the training set.
                This is the same as `objective` but does not include constraint penalty terms.
            constraint_gaps: whether or not to log the squared 2-norm of constraint violations.
                Ignored for unconstrained problems.
            lagrangian_grad: whether or not to log the squared 2-norm of the (primal) subgradient of the Lagrangian function.
                Ignored for unconstrained problems.
            train_accuracy: whether or not to log accuracy on the training set.
            train_mse: whether or not to log mean the squared error on the training set.
            test_accuracy": whether or not to log accuracy on the test set.
            test_mse: whether or not to log the mean squared error on the test set.
            total_neurons: whether or not to log the total number of neurons in the model.
            neuron_sparsity: whether or not to log the proportion of neurons which are not used by the model
              (ie. all weights are exactly zero for those neurons).
            active_features: whether or not to log the number of features used by the model.
            feature_sparsity: whether or not to log the proportion of features which are not used by the model
              (ie. all weights are exactly zero for those features).
            weight_sparsity: whether or not to log the proportion of weights are which zero.
        """

        self.metric_freq = metric_freq
        self.metrics_to_collect = {
            "objective": objective,
            "grad_norm": grad_norm,
            "time": time,
            "model_loss": model_loss,
            "constraint_gaps": constraint_gaps,
            "lagrangian_grad": lagrangian_grad,
            "train_accuracy": train_accuracy,
            "train_mse": train_mse,
            "test_accuracy": test_accuracy,
            "test_mse": test_mse,
            "total_neurons": total_neurons,
            "neuron_sparsity": neuron_sparsity,
            "total_features": total_features,
            "feature_sparsity": feature_sparsity,
            "total_weights": total_weights,
            "weight_sparsity": weight_sparsity,
        }
