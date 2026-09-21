from __future__ import annotations
import io
import os
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import real_data.preprocessing as compact
import real_data.censoring as censoring_ipcw
import real_data.survival_policy as policies
import real_data.columns as bc
import real_data.encoding as core
from pabl.bridge_h import MultiTreatmentProxH
from pabl.bridge_q import MultiTreatmentProxQ


def _load_survival_features(database: str, sql_path: Path) -> pd.DataFrame:
    environment = os.environ.copy()
    environment["PGOPTIONS"] = "-c work_mem=512MB -c statement_timeout=1800000"
    result = subprocess.run(
        [
            "psql",
            "-X",
            "-v",
            "ON_ERROR_STOP=1",
            "-v",
            "emit_survival90_analysis_rows=1",
            "-d",
            database,
            "--csv",
            "-f",
            str(sql_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"psql survival feature view failed:\n{result.stderr[-8000:]}"
        )
    lines = result.stdout.splitlines()
    try:
        header_index = next(
            (
                index
                for (index, line) in enumerate(lines)
                if line.startswith("definition,subject_id,arm,w_culture_positive,")
            )
        )
    except StopIteration as error:
        raise RuntimeError("protected survival CSV header not found") from error
    frame = pd.read_csv(io.StringIO("\n".join(lines[header_index:])))
    valid = (frame["definition"] == "B_24h_treatment_24h_specimen") & frame["arm"].isin(
        core.ARM_ORDER
    )
    if int((~valid).sum()) > 1:
        raise RuntimeError("unexpected protected survival output trailer")
    frame = frame.loc[valid].copy().reset_index(drop=True)
    required = {
        "survival90_observed_time_days",
        "survival90_death_event",
        "survival90_coverage_censor_event",
        "survival90_nominal_coverage_reaches_horizon",
        "survival90_censoring_survival_g",
        "survival90_reconstructed_y",
    }
    if len(frame) != 14906 or frame["subject_id"].duplicated().any():
        raise RuntimeError("unexpected protected survival cohort")
    if required - set(frame.columns):
        raise RuntimeError(
            f"survival columns missing: {sorted(required - set(frame.columns))}"
        )
    if not np.array_equal(
        frame["survival90_reconstructed_y"].to_numpy(dtype=int),
        frame["y_survival90"].to_numpy(dtype=int),
    ):
        raise RuntimeError(
            "survival90 protected reconstruction disagrees with frozen outcome"
        )
    for column in (
        "survival90_death_event",
        "survival90_coverage_censor_event",
        "survival90_nominal_coverage_reaches_horizon",
    ):
        if not frame[column].isin((0, 1)).all():
            raise RuntimeError(f"nonbinary survival column: {column}")
    return frame


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
    coverage_reaches_horizon: np.ndarray | None = None,
    master_fold_id: np.ndarray | None = None,
    censoring_c: float = 0.1,
    censoring_lower_clip: float = 0.05,
    censoring_learner: str = "full_logistic_A_Z_W_X",
    censoring_true_g: np.ndarray | None = None,
    block_label: str = "nuisance",
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
    if coverage_reaches_horizon is None:
        train_g_raw = np.ones(len(train), dtype=float)
        train_g = train_g_raw.copy()
        predict_g_raw = np.ones(len(predict), dtype=float)
        predict_g = predict_g_raw.copy()
        train_coverage = np.ones(len(train), dtype=int)
        predict_coverage = np.ones(len(predict), dtype=int)
    else:
        if master_fold_id is None:
            raise RuntimeError("strict IPCW requires frozen master folds")
        g_bundle = censoring_ipcw.fit_strict_g_block(
            frame,
            treatment,
            coverage_reaches_horizon,
            master_fold_id,
            train,
            predict,
            base_numeric_x,
            c_value=censoring_c,
            lower_clip=censoring_lower_clip,
            seed=seed + 9000000,
            block_label=block_label,
            learner=censoring_learner,
            true_g=censoring_true_g,
        )
        train_g_raw = np.asarray(g_bundle["train_oof_raw"], dtype=float)
        train_g = np.asarray(g_bundle["train_oof_clipped"], dtype=float)
        predict_g_raw = np.asarray(g_bundle["predict_raw"], dtype=float)
        predict_g = np.asarray(g_bundle["predict_clipped"], dtype=float)
        train_coverage = np.asarray(coverage_reaches_horizon, dtype=int)[train]
        predict_coverage = np.asarray(coverage_reaches_horizon, dtype=int)[predict]
    bridge_outcome_train = (
        np.asarray(outcome, dtype=float)[train] * train_coverage / train_g
    )
    ipcw_outcome_predict = (
        np.asarray(outcome, dtype=float)[predict] * predict_coverage / predict_g
    )
    if not (
        np.isfinite(bridge_outcome_train).all()
        and np.isfinite(ipcw_outcome_predict).all()
    ):
        raise RuntimeError("nonfinite strict cross-fitted IPCW outcome")
    use_blocks = kernel_geometry == "block_dimension_normalized"
    block_weights = (1.0, 1.0) if use_blocks else None
    h_model = MultiTreatmentProxH(
        treatment[train], x_train, z_train, w_train, bridge_outcome_train
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
    (propensity_x_train, _) = _observed_probability(
        propensity_x_model, x_train, treatment[train]
    )
    xwz_train = np.column_stack([x_train, w_train, z_train])
    xwz_predict = np.column_stack([x_predict, w_predict, z_predict])
    propensity_xwz_model = LogisticRegression(
        solver="lbfgs", C=float(propensity_c), max_iter=4000, random_state=seed + 1
    ).fit(xwz_train, treatment[train])
    (propensity_xwz, _) = _observed_probability(
        propensity_xwz_model, xwz_predict, treatment[predict]
    )
    (propensity_xwz_train, _) = _observed_probability(
        propensity_xwz_model, xwz_train, treatment[train]
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
        "ipcw_outcome": ipcw_outcome_predict,
        "censoring_g_raw": predict_g_raw,
        "censoring_g": predict_g,
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
        "ipcw_outcome": np.full(n, np.nan, dtype=float),
        "censoring_g_raw": np.full(n, np.nan, dtype=float),
        "censoring_g": np.full(n, np.nan, dtype=float),
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
    observed_time: np.ndarray,
    death_event: np.ndarray,
    coverage_censor_event: np.ndarray,
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
        outcome_train=train_nuisance["ipcw_outcome"][train],
        outcome_validation=validation_nuisance["ipcw_outcome"],
        treatment_train=treatment[train],
        treatment_validation=treatment[validation],
        observed_time_train=observed_time[train],
        observed_time_validation=observed_time[validation],
        death_event_train=death_event[train],
        death_event_validation=death_event[validation],
        coverage_censor_event_train=coverage_censor_event[train],
        coverage_censor_event_validation=coverage_censor_event[validation],
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
