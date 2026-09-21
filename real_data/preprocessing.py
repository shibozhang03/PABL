from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import real_data.encoding as core

RAW_Z = [
    "z_wbc",
    "z_anc",
    "z_alc",
    "z_neutrophil_percent",
    "z_bands",
    "z_immature_granulocytes",
]
VALUE_Z = [f"{column}_compact_signed_log1p" for column in RAW_Z]
MISSING_X = [f"x_{column}_missing" for column in RAW_Z if column != "z_wbc"]


def add_compact_proxy_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in RAW_Z if column not in frame]
    if missing:
        raise ValueError(f"Missing raw CBC proxy columns: {missing}")
    result = frame.copy()
    for raw, transformed in zip(RAW_Z, VALUE_Z):
        values = result[raw].astype(float)
        result[transformed] = np.sign(values) * np.log1p(np.abs(values))
        if raw != "z_wbc":
            result[f"x_{raw}_missing"] = values.isna().astype(float)
    if result["z_wbc"].isna().any():
        raise ValueError("The frozen cohort contract requires complete WBC")
    if result[MISSING_X].isna().any().any():
        raise RuntimeError("Compact missingness flags must be complete")
    return result


def _numeric_pipeline(*, add_indicator: bool) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=add_indicator)),
            ("scale", StandardScaler()),
        ]
    )


def make_compact_preprocessor(
    *,
    base_numeric: Iterable[str],
    proxy_values: Iterable[str],
    missingness_x: Iterable[str],
    categorical: Iterable[str],
    dense: bool,
) -> ColumnTransformer:
    base_numeric = list(base_numeric)
    proxy_values = list(proxy_values)
    missingness_x = list(missingness_x)
    categorical = list(categorical)
    transformers = []
    if base_numeric:
        transformers.append(
            ("base_numeric", _numeric_pipeline(add_indicator=True), base_numeric)
        )
    if proxy_values:
        transformers.append(
            ("proxy_values", _numeric_pipeline(add_indicator=False), proxy_values)
        )
    if missingness_x:
        transformers.append(("proxy_missingness_x", StandardScaler(), missingness_x))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="constant", fill_value="<MISSING>"),
                        ),
                        ("onehot", core.one_hot_encoder(dense=dense)),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("Compact preprocessor requires at least one feature block")
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0 if dense else 0.3,
    )


def crossfit_support_probabilities(
    frame: pd.DataFrame,
    *,
    base_numeric: list[str],
    categorical: list[str],
    learner: str,
    seed: int,
    c_value: float = 0.03,
) -> np.ndarray:
    target = pd.Categorical(frame["arm"], categories=core.ARM_ORDER).codes
    if np.any(target < 0):
        raise ValueError("Unknown treatment arm in compact support audit")
    columns = base_numeric + VALUE_Z + MISSING_X + categorical
    probabilities = np.full((len(frame), len(core.ARM_ORDER)), np.nan)
    for train, validation in core.splits(frame, target, seed):
        if learner == "logistic":
            model = LogisticRegression(solver="lbfgs", C=c_value, max_iter=4000)
            dense = False
        elif learner == "boosted":
            model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=seed,
            )
            dense = True
        else:
            raise ValueError(f"Unknown learner: {learner}")
        pipeline = Pipeline(
            [
                (
                    "preprocess",
                    make_compact_preprocessor(
                        base_numeric=base_numeric,
                        proxy_values=VALUE_Z,
                        missingness_x=MISSING_X,
                        categorical=categorical,
                        dense=dense,
                    ),
                ),
                ("model", model),
            ]
        )
        pipeline.fit(frame.loc[train, columns], target[train])
        fold_probability = pipeline.predict_proba(frame.loc[validation, columns])
        classes = pipeline.named_steps["model"].classes_.astype(int)
        for source, target_column in enumerate(classes):
            probabilities[validation, target_column] = fold_probability[:, source]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Compact support cross-fit produced nonfinite values")
    return np.clip(probabilities, 1e-08, 1.0)
