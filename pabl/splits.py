from __future__ import annotations
import numpy as np
from sklearn.model_selection import (
    PredefinedSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
)


def resolve_crossfit_splits(
    X,
    treatment,
    *,
    n_splits=5,
    random_state=42,
    shuffle=True,
    groups=None,
    splitter=None,
):
    treatment = np.asarray(treatment).reshape(-1)
    n = len(treatment)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for cross-fitting")
    if len(X) != n:
        raise ValueError("X and treatment must have matching lengths")
    group_array = None
    if groups is not None:
        group_array = np.asarray(groups).reshape(-1)
        if len(group_array) != n:
            raise ValueError("groups and treatment must have matching lengths")
    if splitter is None:
        if group_array is None:
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            )
            raw_splits = cv.split(X, treatment)
        else:
            cv = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            )
            raw_splits = cv.split(X, treatment, group_array)
    elif isinstance(splitter, PredefinedSplit):
        raw_splits = splitter.split(X, treatment)
    else:
        raw_splits = splitter.split(X, treatment, group_array)
    expected_classes = np.unique(treatment)
    holdout_count = np.zeros(n, dtype=np.int64)
    splits = []
    for fold_id, (train_index, validation_index) in enumerate(raw_splits):
        train_index = np.asarray(train_index, dtype=np.int64)
        validation_index = np.asarray(validation_index, dtype=np.int64)
        if train_index.size == 0 or validation_index.size == 0:
            raise ValueError(f"Cross-fit fold {fold_id} is empty")
        if np.intersect1d(train_index, validation_index).size:
            raise ValueError(f"Cross-fit fold {fold_id} has row leakage")
        if not np.array_equal(np.unique(treatment[train_index]), expected_classes):
            raise ValueError(
                f"Cross-fit fold {fold_id} training set is missing a treatment arm"
            )
        if not np.array_equal(np.unique(treatment[validation_index]), expected_classes):
            raise ValueError(
                f"Cross-fit fold {fold_id} held-out set is missing a treatment arm"
            )
        if (
            group_array is not None
            and np.intersect1d(
                group_array[train_index], group_array[validation_index]
            ).size
        ):
            raise ValueError(f"Cross-fit fold {fold_id} has subject leakage")
        holdout_count[validation_index] += 1
        splits.append((train_index, validation_index))
    if len(splits) < 2:
        raise ValueError("Cross-fitting requires at least two folds")
    if not np.all(holdout_count == 1):
        raise ValueError("Each observation must be held out exactly once")
    return splits
