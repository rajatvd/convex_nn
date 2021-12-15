"""
Datasets and related utilities.
"""
import os
import pickle as pkl
from typing import Tuple, Dict, Any, Optional, List
from math import ceil

import numpy as np
from scipy.stats import ortho_group  # type: ignore
from sklearn.model_selection import KFold  # type: ignore
from sklearn.preprocessing import OneHotEncoder  # type: ignore

import torch
from torch.utils import data
from torchvision import datasets, transforms  # type: ignore

import lab

from cvx_nn.models import ReLUMLP

from cvx_nn.uci_names import (
    UCI_DATASETS,
    SMALL_UCI_DATASETS,
    BINARY_SMALL_UCI_DATASETS,
)

# types

Dataset = Tuple[lab.Tensor, lab.Tensor]

# constants

PYTORCH_DATASETS = ["mnist", "cifar_10", "cifar_100"]

# datasets from the stochastic line-search paper.

SLS_DATASETS = [
    "quantum",
    "rcv1",
    "protein",
    "news",
    "mushrooms",
    "splice",
    "ijcnn",
    "w8a",
    "covtype",
]

SLS_TO_NUM = {
    "quantum": 0,
    "rcv1": 1,
    "protein": 2,
    "news": 3,
    "mushrooms": 4,
    "splice": 5,
    "ijcnn": 6,
    "w8a": 7,
    "covtype": 8,
}

# See "uci_names.py" for UCI datasets.

# loaders


def load_dataset(dataset_config: Dict[str, Any], data_dir: str = "data"):
    """Load a dataset by name using the passed configuration parameters.
    :param rng: a random number generator.
    :param dataset_config: configuration object specifying the dataset to load.
    :param data_dir: the base directory to look for datasets.
    :returns: (X, y, w_opt), where X is the feature matrix, y are the targets, and w_opt
        is either the true generating model or None if no such model is available.
    """

    name = dataset_config.get("name", None)
    valid_prop = dataset_config.get("valid_prop", 0.2)
    test_prop = dataset_config.get("test_prop", 0.2)
    use_valid = dataset_config.get("use_valid", True)
    split_seed = dataset_config.get("split_seed", 1995)
    n_folds = dataset_config.get("n_folds", None)
    fold_index = dataset_config.get("fold_index", None)
    unitize_data_cols = dataset_config.get("unitize_data_cols", True)

    w_opt = None

    if name is None:
        raise ValueError("Dataset configuration must have a name parameter!")
    elif name == "synthetic_regression":
        data_seed = dataset_config.get("data_seed", 951)
        train_data, test_data, w_opt = generate_synthetic_regression(
            data_seed,
            dataset_config["n"],
            dataset_config["n_test"],
            dataset_config["d"],
            dataset_config.get("sigma", 0.0),
            dataset_config.get("kappa", 1.0),
        )
    elif name == "synthetic_classification":
        data_seed = dataset_config.get("data_seed", 951)
        train_data, test_data = generate_synthetic_classification(
            data_seed,
            dataset_config["n"],
            dataset_config["n_test"],
            dataset_config["d"],
            dataset_config.get("hidden_units", 50),
            dataset_config.get("kappa", 1.0),
        )
    elif name in PYTORCH_DATASETS:
        pytorch_src = os.path.join(data_dir, "pytorch")
        transform = load_transforms(dataset_config.get("transforms", None), name)
        train_data = load_pytorch_dataset(
            name,
            pytorch_src,
            train=True,
            transform=transform,
            valid_prop=valid_prop,
            use_valid=use_valid,
            split_seed=split_seed,
            n_folds=n_folds,
            fold_index=fold_index,
        )

        test_data = load_pytorch_dataset(
            name,
            pytorch_src,
            train=False,
            transform=transform,
            valid_prop=valid_prop,
            use_valid=use_valid,
            split_seed=split_seed,
            n_folds=n_folds,
            fold_index=fold_index,
        )
        train_data, test_data = unitize_features(train_data, test_data)

    elif name in UCI_DATASETS:
        uci_src = os.path.join(data_dir, "uci", "datasets")
        train_data, test_data = load_uci_dataset(
            name,
            uci_src,
            test_prop,
            valid_prop,
            use_valid,
            split_seed,
            n_folds,
            fold_index,
            unitize_data_cols,
        )

    else:
        raise ValueError(
            f"Dataset with name {name} not recognized! Please configure it first."
        )

    return train_data, test_data, w_opt


def generate_synthetic_classification(
    data_seed: int,
    n: int,
    n_test: int,
    d: int,
    hidden_units: int = 50,
    kappa: float = 1.0,
    unitize_data_cols: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Sample a binary classification dataset with Gaussian features, condition number
        (approximately) kappa, and targets given by a two-layer neural network with random Gaussian weights.
    :param data_seed: the seed to use when generating the synthetic dataset.
    :param n: number of examples in dataset.
    :param n_test: number of test examples.
    :param d: number of features for each example.
    :returns: '(X, y), (X_test, y_test), w_opt', where 'X' is an n x d matrix containing the
        training examples, 'y' is a n-length vector containing the training targets and,
        (X_test, y_test) similarly form the test with `n_test` examples and 'w_opt' is the "true"
        model.
    """
    rng = np.random.default_rng(seed=data_seed)
    # sample "true" model
    backend = lab.backend
    lab.set_backend("numpy")
    model = ReLUMLP(d, p=hidden_units)
    model.weights = rng.random(model.weights.shape)

    # sample random orthonormal matrix
    Q = ortho_group.rvs(d, random_state=rng)
    # sample eigenvalues so that lambda_1 / lambda_d = kappa.
    eigs = rng.uniform(low=1, high=kappa, size=d - 2)
    eigs = np.concatenate([np.array([kappa, 1]), eigs])
    # compute covariance
    Sigma = np.dot(Q.T, np.multiply(np.expand_dims(eigs, axis=1), Q))

    X = []
    y = []
    n_pos, n_neg = 0, 0
    n_total = n + n_test

    # simple rejection sampling
    while n_pos + n_neg < n_total:
        xi = rng.multivariate_normal(np.zeros(d), cov=Sigma)
        yi = model(xi)
        if yi <= 0 and n_neg < ceil(n_total):
            y.append(-1)
            X.append(xi)
            n_neg += 1
        elif yi > 0 and n_pos < ceil(n_total):
            y.append(1)
            X.append(xi)
            n_pos += 1
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    # shuffle dataset.
    indices = np.arange(n_total)
    rng.shuffle(indices)
    X, y = X[indices], y[indices]

    train_set = (X[:n], y[:n])
    test_set = (X[n:], y[n:])

    if unitize_data_cols:
        train_set, test_set = unitize_features(train_set, test_set)

    lab.set_backend(backend)
    return (
        lab.all_to_tensor(train_set, lab.get_dtype()),
        lab.all_to_tensor(test_set, lab.get_dtype()),
    )


def generate_synthetic_regression(
    data_seed: int,
    n: int,
    n_test: int,
    d: int,
    sigma: float = 0,
    kappa: float = 1,
    sparse_opt=True,
    c: int = 1,
    vector_output=False,
    unitize_data_cols: bool = True,
) -> Tuple[Dataset, Dataset, lab.Tensor]:
    """Sample a synthetic regression dataset with Gaussian features and condition number
        (approximately) kappa.
    :param data_seed: the seed to use when generating the synthetic dataset.
    :param n: number of examples in dataset.
    :param n_test: number of test examples.
    :param d: number of features for each example.
    :param sigma: variance of (Gaussian) noise added to targets. Pass '0' for a noiseless model.
    :param kappa: condition number of E[X.T X]. Defaults to 1 (perfectly conditioned covariance).
    :returns: '(X, y), (X_test, y_test), w_opt', where 'X' is an n x d matrix containing the
        training examples, 'y' is a n-length vector containing the training targets and,
        (X_test, y_test) similarly form the test with `n_test` examples and 'w_opt' is the "true"
        model.
    """
    rng = np.random.default_rng(seed=data_seed)
    # sample "true" model
    if vector_output:
        w_opt = rng.standard_normal((d, c))
    else:
        w_opt = rng.standard_normal(d)
    if sparse_opt:
        w_opt[rng.choice(np.arange(d), int(np.floor(d / 2)))] = 0
    # sample random orthonormal matrix
    Q = ortho_group.rvs(d, random_state=rng)
    # sample eigenvalues so that lambda_1 / lambda_d = kappa.
    eigs = rng.uniform(low=1, high=kappa, size=d - 2)
    eigs = np.concatenate([np.array([kappa, 1]), eigs])
    # compute covariance
    Sigma = np.dot(Q.T, np.multiply(np.expand_dims(eigs, axis=1), Q))

    X = rng.multivariate_normal(np.zeros(d), cov=Sigma, size=n + n_test)
    y = np.dot(X, w_opt)

    if sigma != 0:
        y = y + rng.normal(0, scale=sigma)

    train_set = (X[:n], y[:n])
    test_set = (X[n:], y[n:])

    if unitize_data_cols:
        train_set, test_set = unitize_features(train_set, test_set)

    return (
        lab.all_to_tensor(train_set, lab.get_dtype()),
        lab.all_to_tensor(test_set, lab.get_dtype()),
        lab.tensor(w_opt, dtype=lab.get_dtype()),
    )


def load_uci_dataset(
    name: str,
    src: str = "data/uci/datasets",
    test_prop: float = 0.2,
    valid_prop: float = 0.2,
    use_valid: bool = True,
    split_seed: int = 1995,
    n_folds: Optional[int] = None,
    fold_index: Optional[int] = None,
    unitize_data_cols: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Load one of the UCI datasets by name.
    :param name: the name of the dataset to load.
    :param src: base path for pytorch datasets..
    :param train: whether to load the training set (True) or test set (False)
    :param transform: torchvision transform for processing the image features.
    :param use_valid: whether or not to use a train/validation split of the training set.
    :param split_seed: the seed to use when constructing the train/validation split.
    :param n_folds: the number of cross-validation folds to use. Defaults to 'None',
        ie. no cross validation is performed.
    :param fold_index: the particular cross validation split to load. This must be provided
        when 'n_folds' is not None.
    :returns: training and test sets.
    """

    data_dict = {}
    for k, v in map(
        lambda x: x.split(),
        open(os.path.join(src, name, name + ".txt"), "r").readlines(),
    ):
        data_dict[k] = v

    # load data
    f = open(os.path.join(src, name, data_dict["fich1="]), "r").readlines()[1:]
    full_X = np.asarray(list(map(lambda x: list(map(float, x.split()[1:-1])), f)))
    full_y = np.asarray(list(map(lambda x: int(x.split()[-1]), f))).squeeze()

    classes = np.unique(full_y)
    if len(classes) == 2:
        full_y[full_y == classes[0]] = -1
        full_y[full_y == classes[1]] = 1
        full_y = np.expand_dims(full_y, 1)
    else:
        # use one-hot encoding for multi-class problems.
        full_y = np.expand_dims(full_y, 1)
        encoder = OneHotEncoder()
        encoder.fit(full_y)
        full_y = encoder.transform(full_y).toarray()

    # for vector-outputs

    # split dataset
    train_set, test_set = train_test_split(
        full_X, full_y, test_prop, split_seed=split_seed
    )
    if n_folds is not None:
        assert fold_index is not None

        train_set, test_set = cv_split(
            train_set[0], train_set[1], fold_index, n_folds, split_seed=split_seed
        )

    elif use_valid:
        train_set, test_set = train_test_split(
            train_set[0], train_set[1], valid_prop, split_seed=split_seed
        )

    if unitize_data_cols:
        train_set, test_set = unitize_features(train_set, test_set)

    return lab.all_to_tensor(train_set, dtype=lab.get_dtype()), lab.all_to_tensor(
        test_set, dtype=lab.get_dtype()
    )


def load_pytorch_dataset(
    name: str,
    src: str = "data/pytorch",
    train: bool = True,
    transform: Optional[Any] = None,
    valid_prop: float = 0.2,
    use_valid: bool = True,
    split_seed: int = 1995,
    n_folds: Optional[int] = None,
    fold_index: Optional[int] = None,
) -> Any:
    """Load TorchVision dataset.
    :param name: the name of the dataset to load.
    :param src: base path for pytorch datasets..
    :param train: whether to load the training set (True) or test set (False)
    :param transform: torchvision transform for processing the image features.
    :param use_valid: whether or not to use a train/validation split of the training set.
    :param split_seed: the seed to use when constructing the train/validation split.
    :param n_folds: the number of cross-validation folds to use. Defaults to 'None',
        ie. no cross validation is performed.
    :param fold_index: the particular cross validation split to load. This must be provided
        when 'n_folds' is not None.
    :returns: torch.utils.data.Dataset.
    """

    if name == "cifar_100":
        cls = datasets.CIFAR100
        num_classes = 100
    elif name == "cifar_10":
        cls = datasets.CIFAR10
        num_classes = 10
    elif name == "mnist":
        cls = datasets.MNIST
        num_classes = 10
    else:
        raise ValueError(
            f"PyTorch dataset with name '{name}' not recognized! Please register it in 'datasets.py'."
        )

    fetch_train = train or use_valid
    # avoid annoying download message by first trying to load the dataset without downloading.
    try:
        dataset = cls(
            root=src,
            transform=transform,
            download=False,
            train=fetch_train,
        )
    except Exception:
        dataset = cls(
            root=src,
            transform=transform,
            download=True,
            train=fetch_train,
        )

    X = []
    y = []
    # iterate through dataset to obtain transformed NumPy arrays
    # one-hot encoding for multi-class labels
    for X_batch, y_batch in data.DataLoader(dataset, batch_size=256):
        X.append(X_batch.numpy())
        y.append(torch.nn.functional.one_hot(y_batch, num_classes).numpy())

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    if n_folds is not None:
        assert fold_index is not None

        train_set, valid_set = cv_split(
            X, y, fold_index, n_folds, split_seed=split_seed
        )

        if train:
            return lab.all_to_tensor(train_set, dtype=lab.get_dtype())
        else:
            return lab.all_to_tensor(valid_set, dtype=lab.get_dtype())

    elif use_valid:

        train_set, valid_set = train_test_split(X, y, valid_prop, split_seed)

        if train:
            return lab.all_to_tensor(train_set, dtype=lab.get_dtype())
        else:
            return lab.all_to_tensor(valid_set, dtype=lab.get_dtype())

    else:
        return lab.all_to_tensor((X, y), dtype=lab.get_dtype())


def train_test_split(X, y, valid_prop=0.2, split_seed=1995):
    n = y.shape[0]
    split_rng = np.random.default_rng(seed=split_seed)
    num_test = int(np.floor(n * valid_prop))
    indices = np.arange(n)
    split_rng.shuffle(indices)
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    # subset the dataset
    X_train = X[train_indices, :]
    y_train = y[train_indices]

    X_test = X[test_indices, :]
    y_test = y[test_indices]

    return (X_train, y_train), (X_test, y_test)


def cv_split(X, y, fold_index, n_folds=5, split_seed=1995):
    kf = KFold(n_folds, shuffle=True, random_state=split_seed)

    train_indices, valid_indices = list(kf.split(X))[fold_index]

    return (X[train_indices], y[train_indices]), (X[valid_indices], y[valid_indices])


def unitize_features(train_set, test_set=None, return_column_norms: bool = False):

    X_train = train_set[0]
    if isinstance(X_train, np.ndarray):
        column_norms = np.sqrt(np.sum(X_train ** 2, axis=0, keepdims=True))
    else:
        column_norms = lab.sqrt(lab.sum(X_train ** 2, axis=0, keepdims=True))

    X_train = X_train / column_norms

    if test_set is not None:
        test_set = (test_set[0] / column_norms, test_set[1])

    if return_column_norms:
        return (X_train, train_set[1]), test_set, column_norms

    return (X_train, train_set[1]), test_set


def load_transforms(transform_names: Optional[List[str]], dataset_name: str) -> Any:
    """Load transformations for a PyTorch dataset.
    :param transform_list: a list of transformations to apply to the dataset.
    :param dataset_name: name of the pytorch dataset (used for normalization)
        Order *matters* as it determines the order in which the transforms are applied.
    :returns: transform
    """

    # no transformations to load.
    if transform_names is None or len(transform_names) == 0:
        return None

    transform_list = []
    for name in transform_names:
        if name == "to_tensor":
            transform_list.append(transforms.ToTensor())
        elif name == "flatten":
            transform_list.append(transforms.Lambda(lambd=lambda x: x.reshape(-1)))
        elif name == "normalize":
            if dataset_name == "mnist":
                transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
            elif dataset_name == "cifar_10":
                transform_list.append(
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    )
                )
            elif dataset_name == "cifar_100":
                transform_list.append(
                    transforms.Normalize(
                        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                    )
                )
            else:
                raise ValueError(
                    f"Dataset {dataset_name} not recognized for normalization! Please register it in 'datasets.py'"
                )

        else:
            raise ValueError(
                f"Transform {name} not recognized! Please register it 'datasets.py'"
            )

    return transforms.Compose(transform_list)
