from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder

ARM_ORDER = ["cefepime", "ceftriaxone", "piperacillin_tazobactam"]


def one_hot_encoder(dense: bool) -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="ignore", min_frequency=5, sparse_output=not dense
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse=not dense)


def splits(
    frame: pd.DataFrame, target: np.ndarray, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    placeholder = np.zeros((len(frame), 1))
    return list(splitter.split(placeholder, target, frame["subject_id"].to_numpy()))
