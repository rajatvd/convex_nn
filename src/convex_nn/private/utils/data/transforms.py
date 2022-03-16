"""Data transformations and related utilities.

Public Functions:

    unitize_columns: unitize the columns of a training set, optionally, a test set.
"""

from typing import Tuple, Optional

import lab


Dataset = Tuple[lab.Tensor, lab.Tensor]


def unitize_columns(
    train_set: Dataset,
    test_set: Optional[Dataset] = None,
) -> Tuple[Dataset, Dataset, lab.Tensor]:
    """Transform a dataset so that the columns of the design matrix have unit
    norm,

    .. math::

        \\text{diag} (\\tilde X^\\top \\tilde X) = I

    If a test set is also provided, the column-norms of the training set are used to apply the same transformation to the test data.

    Args:
        train_set: an (X, y) tuple.
        test_set: (optional) an (X_test, y_test) tuple.

    Returns:
       (train_set, test_set, column_norms) --- a tuple containing the transformed training set, test set, and the column norms of the training design matrix.
       If a test_set is not provided, then `test_set` is None.
    """

    X_train = train_set[0]
    column_norms = lab.sqrt(lab.sum(X_train ** 2, axis=0, keepdims=True))

    X_train = X_train / column_norms
    train_set = (X_train, train_set[1])

    if test_set is not None:
        test_set = (test_set[0] / column_norms, test_set[1])

    return train_set, test_set, column_norms
