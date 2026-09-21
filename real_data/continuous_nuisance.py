from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import real_data.preprocessing as compact
import real_data.continuous_policy as policies
import real_data.columns as bc
import real_data.encoding as core
from pabl.bridge_h import MultiTreatmentProxH
from pabl.bridge_q import MultiTreatmentProxQ


def _fit_policy_matrices(
    frame: pd.DataFrame,
    train: np.ndarray,
    validation: np.ndarray,
    base_numeric_x: list[str],
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int], dict[str, list[str]]
]:
    matrices_train: dict[str, np.ndarray] = {}
    matrices_validation: dict[str, np.ndarray] = {}
    dimensions: dict[str, int] = {}
    feature_names: dict[str, list[str]] = {}
    specifications = {
        "x": (base_numeric_x, []),
        "zx": (base_numeric_x, compact.VALUE_Z),
        "wx": (base_numeric_x + ["w_culture_positive"], []),
    }
    for name, (base_numeric, proxy_values) in specifications.items():
        columns = (
            list(base_numeric)
            + list(proxy_values)
            + compact.MISSING_X
            + bc.CATEGORICAL_X
        )
        processor = compact.make_compact_preprocessor(
            base_numeric=base_numeric,
            proxy_values=proxy_values,
            missingness_x=compact.MISSING_X,
            categorical=bc.CATEGORICAL_X,
            dense=True,
        )
        matrices_train[name] = np.asarray(
            processor.fit_transform(frame.loc[train, columns]), dtype=float
        )
        matrices_validation[name] = np.asarray(
            processor.transform(frame.loc[validation, columns]), dtype=float
        )
        if not (
            np.isfinite(matrices_train[name]).all()
            and np.isfinite(matrices_validation[name]).all()
        ):
            raise RuntimeError(f"nonfinite policy matrix for {name}")
        dimensions[name] = int(matrices_train[name].shape[1])
        names = [str(value) for value in processor.get_feature_names_out()]
        if len(names) != dimensions[name]:
            raise RuntimeError(f"policy feature-name dimension mismatch for {name}")
        feature_names[name] = names
    return (matrices_train, matrices_validation, dimensions, feature_names)


def _observed_probability(
    model: LogisticRegression, features: np.ndarray, treatment: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    probability = model.predict_proba(features)
    classes = model.classes_.astype(int)
    expected = np.arange(len(core.ARM_ORDER))
    if not np.array_equal(classes, expected):
        raise RuntimeError("propensity model is missing a treatment arm")
    observed = probability[np.arange(len(treatment)), treatment]
    return (observed, probability)


def fit_compact_nuisances_predict(
    frame: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    train: np.ndarray,
    predict: np.ndarray,
    base_numeric_x: list[str],
    *,
    seed: int,
    h_components: int,
    q_components: int,
    h_alpha_scales: tuple[float, ...],
    q_alpha_scales: tuple[float, ...],
    kernel_geometry: str,
    propensity_c: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    x_columns = base_numeric_x + compact.MISSING_X + bc.CATEGORICAL_X
    x_processor = compact.make_compact_preprocessor(
        base_numeric=base_numeric_x,
        proxy_values=[],
        missingness_x=compact.MISSING_X,
        categorical=bc.CATEGORICAL_X,
        dense=True,
    )
    x_train = np.asarray(
        x_processor.fit_transform(frame.loc[train, x_columns]), dtype=float
    )
    x_predict = np.asarray(
        x_processor.transform(frame.loc[predict, x_columns]), dtype=float
    )
    z_processor = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=False)),
            ("scale", StandardScaler()),
        ]
    )
    z_train = np.asarray(
        z_processor.fit_transform(frame.loc[train, compact.VALUE_Z]), dtype=float
    )
    z_predict = np.asarray(
        z_processor.transform(frame.loc[predict, compact.VALUE_Z]), dtype=float
    )
    w_processor = StandardScaler()
    w = frame[["w_culture_positive"]].to_numpy(dtype=float)
    w_train = w_processor.fit_transform(w[train])
    w_predict = w_processor.transform(w[predict])
    if not all(
        (
            np.isfinite(values).all()
            for values in (x_train, x_predict, z_train, z_predict, w_train, w_predict)
        )
    ):
        raise RuntimeError("nonfinite compact nuisance matrix")
    use_blocks = kernel_geometry == "block_dimension_normalized"
    block_weights = (1.0, 1.0) if use_blocks else None
    h_model = MultiTreatmentProxH(
        treatment[train], x_train, z_train, w_train, outcome[train]
    )
    h_model.fit_cv(
        gamma_f="auto",
        gamma_hs="auto",
        n_gamma_hs=3,
        n_alphas=len(h_alpha_scales),
        alpha_scales=list(h_alpha_scales),
        n_components=h_components,
        cv=3,
        nystrom_random_state=seed,
        input_block_sizes=(1, x_train.shape[1]) if use_blocks else None,
        condition_block_sizes=(
            (z_train.shape[1], x_train.shape[1]) if use_blocks else None
        ),
        input_block_distance_weights=block_weights,
        condition_block_distance_weights=block_weights,
    )
    h_predict = h_model.predict_counterfactual(x_predict, w_predict)
    q_model = MultiTreatmentProxQ(treatment[train], x_train, z_train, w_train)
    q_model.fit_cv(
        gamma_f="auto",
        gamma_hs="auto",
        n_gamma_hs=3,
        n_alphas=len(q_alpha_scales),
        alpha_scales=list(q_alpha_scales),
        n_components=q_components,
        cv=3,
        nystrom_random_state=seed + 271828,
        input_block_sizes=(z_train.shape[1], x_train.shape[1]) if use_blocks else None,
        condition_block_sizes=(x_train.shape[1], 1) if use_blocks else None,
        input_block_distance_weights=block_weights,
        condition_block_distance_weights=block_weights,
    )
    q_train_observed = q_model.predict_observed()
    denominators = np.asarray(
        [
            np.mean((treatment[train] == arm) * q_train_observed)
            for arm in range(len(core.ARM_ORDER))
        ],
        dtype=float,
    )
    if not np.isfinite(denominators).all() or np.any(denominators <= 0):
        raise RuntimeError("q empirical-moment normalization is invalid")
    q_scales = 1.0 / denominators
    q_predict_raw = q_model.predict_observed_for(
        x_predict, z_predict, treatment[predict]
    )
    q_predict = q_predict_raw * q_scales[treatment[predict]]
    propensity_x_model = LogisticRegression(
        solver="lbfgs", C=float(propensity_c), max_iter=4000, random_state=seed
    ).fit(x_train, treatment[train])
    (propensity_x, _) = _observed_probability(
        propensity_x_model, x_predict, treatment[predict]
    )
    xwz_train = np.column_stack([x_train, w_train, z_train])
    xwz_predict = np.column_stack([x_predict, w_predict, z_predict])
    propensity_xwz_model = LogisticRegression(
        solver="lbfgs", C=float(propensity_c), max_iter=4000, random_state=seed + 1
    ).fit(xwz_train, treatment[train])
    (propensity_xwz, _) = _observed_probability(
        propensity_xwz_model, xwz_predict, treatment[predict]
    )
    if (
        int(np.max(propensity_x_model.n_iter_)) >= 4000
        or int(np.max(propensity_xwz_model.n_iter_)) >= 4000
    ):
        raise RuntimeError("propensity nuisance reached its iteration limit")
    bundle = {
        "h": h_predict,
        "q": q_predict,
        "propensity_x": propensity_x,
        "propensity_xwz": propensity_xwz,
    }
    if not all((np.isfinite(values).all() for values in bundle.values())):
        raise RuntimeError("nuisance prediction bundle contains nonfinite values")
    return bundle


def _empty_bundle(n: int) -> dict[str, np.ndarray]:
    arm_count = len(core.ARM_ORDER)
    return {
        "h": np.full((n, arm_count), np.nan, dtype=float),
        "q": np.full(n, np.nan, dtype=float),
        "propensity_x": np.full(n, np.nan, dtype=float),
        "propensity_xwz": np.full(n, np.nan, dtype=float),
    }


def _slice_bundle(
    bundle: dict[str, np.ndarray],
    excluded: tuple[int, ...],
    fold_rows: list[np.ndarray],
) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    start = 0
    for fold in excluded:
        stop = start + len(fold_rows[fold])
        result[fold] = {
            key: np.asarray(value[start:stop]) for (key, value) in bundle.items()
        }
        start = stop
    if start != len(bundle["q"]):
        raise RuntimeError("nuisance prediction slicing invariant failed")
    return result


def _assign_bundle(
    destination: dict[str, np.ndarray], rows: np.ndarray, values: dict[str, np.ndarray]
) -> None:
    for key in destination:
        destination[key][rows] = values[key]


def _make_policy_fold(
    frame: pd.DataFrame,
    train: np.ndarray,
    validation: np.ndarray,
    base_numeric_x: list[str],
    outcome: np.ndarray,
    treatment: np.ndarray,
    train_nuisance: dict[str, np.ndarray],
    validation_nuisance: dict[str, np.ndarray],
    policy_fold: int,
) -> tuple[policies.PolicyFold, dict[str, int], dict[str, list[str]]]:
    (features_train, features_validation, dimensions, feature_names) = (
        _fit_policy_matrices(frame, train, validation, base_numeric_x)
    )
    payload = policies.PolicyFold(
        features_train=features_train,
        features_validation=features_validation,
        outcome_train=outcome[train],
        outcome_validation=outcome[validation],
        treatment_train=treatment[train],
        treatment_validation=treatment[validation],
        h_train=train_nuisance["h"][train],
        h_validation=validation_nuisance["h"],
        q_train=train_nuisance["q"][train],
        q_validation=validation_nuisance["q"],
        propensity_x_train=train_nuisance["propensity_x"][train],
        propensity_x_validation=validation_nuisance["propensity_x"],
        propensity_xwz_train=train_nuisance["propensity_xwz"][train],
        propensity_xwz_validation=validation_nuisance["propensity_xwz"],
        policy_fold=int(policy_fold),
    )
    payload.validate(len(core.ARM_ORDER))
    return (payload, dimensions, feature_names)


def _score_policy(
    recommendation: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    nuisance: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    rows = np.arange(len(recommendation))
    h_value = nuisance["h"][rows, recommendation]
    q_value = outcome * (treatment == recommendation) * nuisance["q"]
    dr = policies.dr_potential(outcome, treatment, nuisance["h"], nuisance["q"])
    return {"h": h_value, "q": q_value, "dr": dr[rows, recommendation]}
