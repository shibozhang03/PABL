from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from pabl.common import (
    fit_proximal_itr_linear_fista,
    fit_weighted_proximal_itr_linear_fista,
)
from pabl.bridge_h import MultiTreatmentProxH
from pabl.bridge_q import MultiTreatmentProxQ

SIM_POLICY_FOLDS = 3
SIM_NUISANCE_FOLDS = 3
SIM_BRIDGE_COMPONENTS = 50
GROUP_LASSO_RATIO_GRID = tuple((float(2.0 ** (-j)) for j in range(11)))
LASSO_RATIO_GRID = tuple((float(10.0 ** (-j / 4.0)) for j in range(13)))
XUE_LAMBDA_GRID = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)
XUE_B_GRID = tuple((float(j) / 2.0 for j in range(1, 21)))
PI_PROB_CLIP_LOWER = 0.05
PI_WEIGHT_CLIP_UPPER = 20.0
CENSOR_SURVIVAL_LOWER = 0.001


@dataclass(frozen=True)
class FrozenSimulationSplitPlan:
    policy_splits: Tuple[Tuple[np.ndarray, np.ndarray], ...]
    n_rows: int
    seed: int

    def audit(self) -> Dict[str, object]:
        validation_counts = np.zeros(self.n_rows, dtype=int)
        rows = []
        for fold_id, (train, validation) in enumerate(self.policy_splits):
            overlap = np.intersect1d(train, validation)
            if overlap.size:
                raise AssertionError(f"policy fold {fold_id} has row overlap")
            validation_counts[validation] += 1
            rows.append(
                {
                    "fold": int(fold_id),
                    "train_n": int(len(train)),
                    "validation_n": int(len(validation)),
                    "overlap_n": int(overlap.size),
                }
            )
        if not np.all(validation_counts == 1):
            raise AssertionError(
                "each simulation row must occur in one policy validation fold"
            )
        return {
            "policy_folds": int(len(self.policy_splits)),
            "nuisance_folds_within_policy_train": int(SIM_NUISANCE_FOLDS),
            "frozen_across_methods": True,
            "validation_membership_exactly_once": True,
            "folds": rows,
        }


def make_simulation_split_plan(
    treatment: Sequence[int], n_splits: int = SIM_POLICY_FOLDS, seed: int = 42
) -> FrozenSimulationSplitPlan:
    treatment = np.asarray(treatment).reshape(-1).astype(int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = tuple(
        (
            (train.astype(int), validation.astype(int))
            for (train, validation) in splitter.split(
                np.zeros(len(treatment)), treatment
            )
        )
    )
    plan = FrozenSimulationSplitPlan(splits, len(treatment), int(seed))
    plan.audit()
    return plan


def _as_2d(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def _as_1d(x, dtype=float) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1).astype(dtype)


def _resolve_feature_extras(raw: Dict[str, np.ndarray], spec: str) -> List[np.ndarray]:
    if spec == "X":
        return []
    mapping = {"XU": ["U"], "XW": ["W"], "XZ": ["Z"], "XWZ": ["W", "Z"]}
    if spec not in mapping:
        raise ValueError(f"Unknown policy/propensity feature specification: {spec}")
    return [_as_2d(raw[key]) for key in mapping[spec]]


def fit_transform_policy_features(
    raw_train: Dict[str, np.ndarray],
    raw_validation: Dict[str, np.ndarray],
    spec: str,
    linearity: str,
) -> Tuple[np.ndarray, np.ndarray]:
    x_train = _as_2d(raw_train["X"])
    x_validation = _as_2d(raw_validation["X"])
    if linearity == "linear":
        (x_train_basis, x_validation_basis) = (x_train, x_validation)
    else:
        spline = SplineTransformer(n_knots=4, degree=3, include_bias=False)
        x_train_basis = spline.fit_transform(x_train)
        x_validation_basis = spline.transform(x_validation)
    train_blocks = [x_train_basis, *_resolve_feature_extras(raw_train, spec)]
    validation_blocks = [
        x_validation_basis,
        *_resolve_feature_extras(raw_validation, spec),
    ]
    feature_train = np.column_stack(train_blocks)
    feature_validation = np.column_stack(validation_blocks)
    scaler = StandardScaler()
    return (scaler.fit_transform(feature_train), scaler.transform(feature_validation))


def _subset_raw(raw: Dict[str, np.ndarray], rows: np.ndarray) -> Dict[str, np.ndarray]:
    return {key: _as_2d(value)[rows] for (key, value) in raw.items()}


def _observed_probability(
    model, features: np.ndarray, treatment: np.ndarray
) -> np.ndarray:
    probabilities = model.predict_proba(features)
    columns = {int(label): j for (j, label) in enumerate(model.classes_)}
    selected = np.asarray([columns[int(a)] for a in treatment], dtype=int)
    return probabilities[np.arange(len(treatment)), selected]


def fit_propensity_oof_and_validation(
    raw: Dict[str, np.ndarray],
    treatment: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: Optional[np.ndarray],
    spec: str,
    linearity: str,
    n_splits: int = SIM_NUISANCE_FOLDS,
    seed: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    treatment = np.asarray(treatment).reshape(-1).astype(int)
    local_a = treatment[train_rows]
    oof = np.empty(len(train_rows), dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for nuisance_train_local, nuisance_validation_local in splitter.split(
        np.zeros(len(local_a)), local_a
    ):
        nuisance_train_rows = train_rows[nuisance_train_local]
        nuisance_validation_rows = train_rows[nuisance_validation_local]
        (x_train, x_validation) = fit_transform_policy_features(
            _subset_raw(raw, nuisance_train_rows),
            _subset_raw(raw, nuisance_validation_rows),
            spec,
            linearity,
        )
        model = LogisticRegression(solver="lbfgs", max_iter=2000)
        model.fit(x_train, treatment[nuisance_train_rows])
        oof[nuisance_validation_local] = _observed_probability(
            model, x_validation, treatment[nuisance_validation_rows]
        )
    oof = np.clip(oof, PI_PROB_CLIP_LOWER, 1.0)
    if validation_rows is None:
        return (oof, None)
    (x_train, x_validation) = fit_transform_policy_features(
        _subset_raw(raw, train_rows), _subset_raw(raw, validation_rows), spec, linearity
    )
    model = LogisticRegression(solver="lbfgs", max_iter=2000)
    model.fit(x_train, treatment[train_rows])
    validation_probability = _observed_probability(
        model, x_validation, treatment[validation_rows]
    )
    return (oof, np.clip(validation_probability, PI_PROB_CLIP_LOWER, 1.0))


def _fit_continuous_q(
    raw: Dict[str, np.ndarray],
    treatment: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: Optional[np.ndarray],
    bridge_cv: int,
    n_components: int,
    seed: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    local = _subset_raw(raw, train_rows)
    model = MultiTreatmentProxQ(
        treatment[train_rows], local["X"], local["Z"], local["W"]
    )
    q_oof = model.fit_cross_fitted_observed(
        n_splits=SIM_NUISANCE_FOLDS,
        random_state=seed,
        cv=bridge_cv,
        n_components=n_components,
    )
    q_validation = None
    if validation_rows is not None:
        full_model = MultiTreatmentProxQ(
            treatment[train_rows], local["X"], local["Z"], local["W"]
        )
        full_model.fit_cv(cv=bridge_cv, n_components=n_components)
        q_validation = full_model.predict_observed_for(
            X_new=_as_2d(raw["X"])[validation_rows],
            Z_new=_as_2d(raw["Z"])[validation_rows],
            A_new=treatment[validation_rows],
        )
    if not np.isfinite(q_oof).all() or (
        q_validation is not None and (not np.isfinite(q_validation).all())
    ):
        raise FloatingPointError("q contains non-finite values")
    return (
        np.asarray(q_oof, dtype=float),
        None if q_validation is None else np.asarray(q_validation, dtype=float),
    )


def _fit_continuous_h(
    raw: Dict[str, np.ndarray],
    treatment: np.ndarray,
    outcome: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: Optional[np.ndarray],
    bridge_cv: int,
    n_components: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    local = _subset_raw(raw, train_rows)
    model = MultiTreatmentProxH(
        treatment[train_rows], local["X"], local["Z"], local["W"], outcome[train_rows]
    )
    (h_observed_oof, h_all_oof) = model.fit_cross_fitted(
        n_splits=SIM_NUISANCE_FOLDS,
        random_state=seed,
        cv=bridge_cv,
        n_components=n_components,
    )
    h_observed_validation = h_all_validation = None
    if validation_rows is not None:
        full_model = MultiTreatmentProxH(
            treatment[train_rows],
            local["X"],
            local["Z"],
            local["W"],
            outcome[train_rows],
        )
        full_model.fit_cv(cv=bridge_cv, n_components=n_components)
        h_all_validation = full_model.predict_counterfactual(
            X_new=_as_2d(raw["X"])[validation_rows],
            W_new=_as_2d(raw["W"])[validation_rows],
        )
        h_observed_validation = full_model.predict_observed_for(
            X_new=_as_2d(raw["X"])[validation_rows],
            W_new=_as_2d(raw["W"])[validation_rows],
            A_new=treatment[validation_rows],
        )
    for values in (h_observed_oof, h_all_oof, h_observed_validation, h_all_validation):
        if values is not None and (not np.isfinite(values).all()):
            raise FloatingPointError("unclipped h contains non-finite values")
    return (h_observed_oof, h_all_oof, h_observed_validation, h_all_validation)


def _censor_splits(treatment, delta, n_splits=SIM_NUISANCE_FOLDS, seed=42):
    treatment = np.asarray(treatment).reshape(-1).astype(int)
    delta = np.asarray(delta).reshape(-1).astype(int)
    joint = treatment * 2 + (1 - delta)
    for strata in (joint, treatment):
        counts = np.bincount(strata)
        positive = counts[counts > 0]
        if positive.size and np.min(positive) >= n_splits:
            return list(
                StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=seed
                ).split(np.zeros(len(strata)), strata)
            )
    return list(
        KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
            np.zeros(len(treatment))
        )
    )


def _censor_design(raw_train, raw_validation, a_train, a_validation, linearity):
    (x_train, x_validation) = fit_transform_policy_features(
        raw_train, raw_validation, "XWZ", linearity
    )
    K = int(max(np.max(a_train), np.max(a_validation)) + 1)
    train_dummy = np.eye(K)[a_train.astype(int)][:, 1:]
    validation_dummy = np.eye(K)[a_validation.astype(int)][:, 1:]
    return (
        np.column_stack([x_train, train_dummy]),
        np.column_stack([x_validation, validation_dummy]),
    )


def _fit_censor_cox_predict_ipcw(
    raw, treatment, y_time, delta, train_rows, validation_rows, linearity, t_eval
):
    (x_train, x_validation) = _censor_design(
        _subset_raw(raw, train_rows),
        _subset_raw(raw, validation_rows),
        treatment[train_rows],
        treatment[validation_rows],
        linearity,
    )
    columns = [f"cov_{j}" for j in range(x_train.shape[1])]
    frame = pd.DataFrame(x_train, columns=columns)
    frame["time"] = y_time[train_rows]
    frame["censoring_event"] = 1 - delta[train_rows]
    model = CoxPHFitter(penalizer=0.01)
    model.fit(frame, duration_col="time", event_col="censoring_event")
    validation_frame = pd.DataFrame(x_validation, columns=columns)
    survival = (
        model.predict_survival_function(validation_frame, times=[float(t_eval)])
        .iloc[0]
        .to_numpy(dtype=float)
    )
    return 1.0 / np.clip(survival, CENSOR_SURVIVAL_LOWER, 1.0)


def _censor_cox_oof(
    raw, treatment, y_time, delta, allowed_rows, linearity, t_eval, seed
):
    local_splits = _censor_splits(
        treatment[allowed_rows], delta[allowed_rows], seed=seed
    )
    oof = np.empty(len(allowed_rows), dtype=float)
    for train_local, validation_local in local_splits:
        oof[validation_local] = _fit_censor_cox_predict_ipcw(
            raw,
            treatment,
            y_time,
            delta,
            allowed_rows[train_local],
            allowed_rows[validation_local],
            linearity,
            t_eval,
        )
    return oof


def _fixed_time_outcome(y_time, ipcw, t_eval):
    return (np.asarray(y_time) > float(t_eval)).astype(float) * np.asarray(ipcw)


def _fit_survival_h_and_censor(
    raw,
    treatment,
    y_time,
    delta,
    train_rows,
    validation_rows,
    linearity,
    t_eval,
    bridge_cv,
    n_components,
    seed,
):
    local_a = treatment[train_rows]
    nuisance_splitter = StratifiedKFold(
        n_splits=SIM_NUISANCE_FOLDS, shuffle=True, random_state=seed
    )
    h_observed_oof = np.empty(len(train_rows), dtype=float)
    h_all_oof = np.empty((len(train_rows), len(np.unique(treatment))), dtype=float)
    fixed_outcome_oof = np.empty(len(train_rows), dtype=float)
    for nuisance_id, (h_train_local, h_validation_local) in enumerate(
        nuisance_splitter.split(np.zeros(len(local_a)), local_a)
    ):
        h_train_rows = train_rows[h_train_local]
        h_validation_rows = train_rows[h_validation_local]
        ipcw_h_train = _censor_cox_oof(
            raw,
            treatment,
            y_time,
            delta,
            h_train_rows,
            linearity,
            t_eval,
            seed + 100 * (nuisance_id + 1),
        )
        target_h_train = _fixed_time_outcome(y_time[h_train_rows], ipcw_h_train, t_eval)
        h_local = _subset_raw(raw, h_train_rows)
        h_model = MultiTreatmentProxH(
            treatment[h_train_rows],
            h_local["X"],
            h_local["Z"],
            h_local["W"],
            target_h_train,
        )
        h_model.fit_cv(cv=bridge_cv, n_components=n_components)
        validation_raw = _subset_raw(raw, h_validation_rows)
        h_all_oof[h_validation_local] = h_model.predict_counterfactual(
            validation_raw["X"], validation_raw["W"]
        )
        h_observed_oof[h_validation_local] = h_model.predict_observed_for(
            validation_raw["X"], validation_raw["W"], treatment[h_validation_rows]
        )
        ipcw_validation = _fit_censor_cox_predict_ipcw(
            raw,
            treatment,
            y_time,
            delta,
            h_train_rows,
            h_validation_rows,
            linearity,
            t_eval,
        )
        fixed_outcome_oof[h_validation_local] = _fixed_time_outcome(
            y_time[h_validation_rows], ipcw_validation, t_eval
        )
    h_observed_validation = h_all_validation = fixed_outcome_validation = None
    if validation_rows is not None:
        ipcw_train_for_h = _censor_cox_oof(
            raw, treatment, y_time, delta, train_rows, linearity, t_eval, seed + 7000
        )
        target_train_for_h = _fixed_time_outcome(
            y_time[train_rows], ipcw_train_for_h, t_eval
        )
        local = _subset_raw(raw, train_rows)
        h_model = MultiTreatmentProxH(
            treatment[train_rows],
            local["X"],
            local["Z"],
            local["W"],
            target_train_for_h,
        )
        h_model.fit_cv(cv=bridge_cv, n_components=n_components)
        validation_raw = _subset_raw(raw, validation_rows)
        h_all_validation = h_model.predict_counterfactual(
            validation_raw["X"], validation_raw["W"]
        )
        h_observed_validation = h_model.predict_observed_for(
            validation_raw["X"], validation_raw["W"], treatment[validation_rows]
        )
        ipcw_validation = _fit_censor_cox_predict_ipcw(
            raw,
            treatment,
            y_time,
            delta,
            train_rows,
            validation_rows,
            linearity,
            t_eval,
        )
        fixed_outcome_validation = _fixed_time_outcome(
            y_time[validation_rows], ipcw_validation, t_eval
        )
    for values in (
        h_observed_oof,
        h_all_oof,
        fixed_outcome_oof,
        h_observed_validation,
        h_all_validation,
        fixed_outcome_validation,
    ):
        if values is not None and (not np.isfinite(values).all()):
            raise FloatingPointError("survival nested h/G produced non-finite values")
    return (
        h_observed_oof,
        h_all_oof,
        h_observed_validation,
        h_all_validation,
        fixed_outcome_oof,
        fixed_outcome_validation,
    )


def build_survival_honest_payloads(
    raw: Dict[str, np.ndarray],
    treatment,
    y_time,
    delta,
    linearity: str,
    t_eval: float,
    policy_specs: Iterable[str],
    propensity_specs: Iterable[str],
    need_q: bool,
    need_h: bool,
    bridge_cv: int = 3,
    n_components: int = SIM_BRIDGE_COMPONENTS,
    seed: int = 42,
):
    if int(n_components) != SIM_BRIDGE_COMPONENTS:
        raise ValueError("simulation bridge Nystrom components are frozen at m=50")
    treatment = _as_1d(treatment, int)
    y_time = _as_1d(y_time, float)
    delta = _as_1d(delta, int)
    raw = {key: _as_2d(value) for (key, value) in raw.items()}
    plan = make_simulation_split_plan(treatment, seed=seed)
    policy_specs = tuple(dict.fromkeys(policy_specs))
    propensity_specs = tuple(dict.fromkeys(propensity_specs))

    def build_one(train_rows, validation_rows, fold_seed):
        payload = {
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "A_train": treatment[train_rows],
            "A_validation": (
                None if validation_rows is None else treatment[validation_rows]
            ),
            "Y_time_train": y_time[train_rows],
            "Delta_train": delta[train_rows],
            "Y_time_validation": (
                None if validation_rows is None else y_time[validation_rows]
            ),
            "Delta_validation": (
                None if validation_rows is None else delta[validation_rows]
            ),
            "features": {},
            "pi": {},
        }
        if validation_rows is not None:
            for spec in policy_specs:
                payload["features"][spec] = fit_transform_policy_features(
                    _subset_raw(raw, train_rows),
                    _subset_raw(raw, validation_rows),
                    spec,
                    linearity,
                )
        for spec in propensity_specs:
            payload["pi"][spec] = fit_propensity_oof_and_validation(
                raw,
                treatment,
                train_rows,
                validation_rows,
                spec,
                linearity,
                seed=fold_seed,
            )
        if need_q:
            payload["q"] = _fit_continuous_q(
                raw,
                treatment,
                train_rows,
                validation_rows,
                bridge_cv,
                n_components,
                fold_seed,
            )
        if need_h:
            h_bundle = _fit_survival_h_and_censor(
                raw,
                treatment,
                y_time,
                delta,
                train_rows,
                validation_rows,
                linearity,
                t_eval,
                bridge_cv,
                n_components,
                fold_seed,
            )
            payload["h"] = h_bundle[:4]
            payload["Y_train"] = h_bundle[4]
            payload["Y_validation"] = h_bundle[5]
        else:
            ipcw_train = _censor_cox_oof(
                raw, treatment, y_time, delta, train_rows, linearity, t_eval, fold_seed
            )
            payload["Y_train"] = _fixed_time_outcome(
                y_time[train_rows], ipcw_train, t_eval
            )
            if validation_rows is None:
                payload["Y_validation"] = None
            else:
                ipcw_validation = _fit_censor_cox_predict_ipcw(
                    raw,
                    treatment,
                    y_time,
                    delta,
                    train_rows,
                    validation_rows,
                    linearity,
                    t_eval,
                )
                payload["Y_validation"] = _fixed_time_outcome(
                    y_time[validation_rows], ipcw_validation, t_eval
                )
        return payload

    payloads = [
        build_one(train, validation, seed + 1000 * (fold_id + 1))
        for (fold_id, (train, validation)) in enumerate(plan.policy_splits)
    ]
    all_rows = np.arange(len(treatment), dtype=int)
    final_payload = build_one(all_rows, None, seed + 9000)
    final_payload.update(
        {
            "split_audit": plan.audit(),
            "bridge_n_components": int(n_components),
            "censor_survival_lower_bound": float(CENSOR_SURVIVAL_LOWER),
            "censoring_nested_inside_allowed_training_set": True,
        }
    )
    return (plan, payloads, final_payload)


def build_continuous_honest_payloads(
    raw: Dict[str, np.ndarray],
    treatment,
    outcome,
    linearity: str,
    policy_specs: Iterable[str],
    propensity_specs: Iterable[str],
    need_q: bool,
    need_h: bool,
    bridge_cv: int = 3,
    n_components: int = SIM_BRIDGE_COMPONENTS,
    seed: int = 42,
) -> Tuple[FrozenSimulationSplitPlan, List[Dict[str, object]], Dict[str, object]]:
    if int(n_components) != SIM_BRIDGE_COMPONENTS:
        raise ValueError("simulation bridge Nystrom components are frozen at m=50")
    treatment = _as_1d(treatment, int)
    outcome = _as_1d(outcome, float)
    raw = {key: _as_2d(value) for (key, value) in raw.items()}
    policy_specs = tuple(dict.fromkeys(policy_specs))
    propensity_specs = tuple(dict.fromkeys(propensity_specs))
    plan = make_simulation_split_plan(treatment, seed=seed)
    payloads = []

    def build_one(train_rows, validation_rows, fold_seed):
        payload: Dict[str, object] = {
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "A_train": treatment[train_rows],
            "Y_train": outcome[train_rows],
            "A_validation": (
                None if validation_rows is None else treatment[validation_rows]
            ),
            "Y_validation": (
                None if validation_rows is None else outcome[validation_rows]
            ),
            "features": {},
            "pi": {},
        }
        if validation_rows is not None:
            for spec in policy_specs:
                payload["features"][spec] = fit_transform_policy_features(
                    _subset_raw(raw, train_rows),
                    _subset_raw(raw, validation_rows),
                    spec,
                    linearity,
                )
        for spec in propensity_specs:
            payload["pi"][spec] = fit_propensity_oof_and_validation(
                raw,
                treatment,
                train_rows,
                validation_rows,
                spec,
                linearity,
                seed=fold_seed,
            )
        if need_q:
            payload["q"] = _fit_continuous_q(
                raw,
                treatment,
                train_rows,
                validation_rows,
                bridge_cv,
                n_components,
                fold_seed,
            )
        if need_h:
            payload["h"] = _fit_continuous_h(
                raw,
                treatment,
                outcome,
                train_rows,
                validation_rows,
                bridge_cv,
                n_components,
                fold_seed,
            )
        return payload

    for fold_id, (train_rows, validation_rows) in enumerate(plan.policy_splits):
        payloads.append(
            build_one(train_rows, validation_rows, seed + 1000 * (fold_id + 1))
        )
    all_rows = np.arange(len(treatment), dtype=int)
    final_payload = build_one(all_rows, None, seed + 9000)
    final_payload["split_audit"] = plan.audit()
    final_payload["bridge_n_components"] = int(n_components)
    return (plan, payloads, final_payload)


def inverse_probability(probability) -> np.ndarray:
    probability = np.asarray(probability, dtype=float).reshape(-1)
    return np.minimum(
        1.0 / np.clip(probability, PI_PROB_CLIP_LOWER, 1.0), PI_WEIGHT_CLIP_UPPER
    )


def group_lasso_lambda_max(features, response, sample_weight=None) -> float:
    x = np.asarray(features, dtype=float)
    y = np.asarray(response, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if sample_weight is None:
        weight = np.ones(len(x), dtype=float)
    else:
        weight = np.asarray(sample_weight, dtype=float).reshape(-1)
    intercept = np.sum(weight[:, None] * y, axis=0) / np.sum(weight)
    residual = y - intercept
    gradient = x.T @ (weight[:, None] * residual) / len(x)
    value = float(np.max(np.linalg.norm(gradient, axis=1)))
    return max(value, np.finfo(float).eps)


def _continuous_target_and_score(payload, method: str, V: np.ndarray, pi_spec: str):
    a_train = payload["A_train"]
    y_train = payload["Y_train"]
    a_validation = payload["A_validation"]
    y_validation = payload["Y_validation"]
    v_train = V[a_train]
    (pi_train, pi_validation) = payload["pi"].get(pi_spec, (None, None))
    sample_weight = None
    if method == "prox_q":
        (q_train, q_validation) = payload["q"]
        response = y_train[:, None] * q_train[:, None] * v_train

        def score(prediction):
            return np.mean(y_validation * (prediction == a_validation) * q_validation)

    elif method == "prox_h":
        (h_train, _h_all_train, _h_validation, h_all_validation) = payload["h"]
        response = V.shape[0] * h_train[:, None] * v_train
        sample_weight = inverse_probability(pi_train)

        def score(prediction):
            return np.mean(h_all_validation[np.arange(len(prediction)), prediction])

    elif method == "full_h":
        (_h_train, h_all_train, _h_validation, h_all_validation) = payload["h"]
        response = h_all_train @ V

        def score(prediction):
            return np.mean(h_all_validation[np.arange(len(prediction)), prediction])

    elif method == "dr":
        (q_train, q_validation) = payload["q"]
        (h_train, _h_all_train, h_validation, _h_all_validation) = payload["h"]
        scalar = h_train * inverse_probability(pi_train) + q_train * (y_train - h_train)
        response = scalar[:, None] * v_train
        scalar_validation = h_validation * inverse_probability(
            pi_validation
        ) + q_validation * (y_validation - h_validation)

        def score(prediction):
            return np.mean((prediction == a_validation) * scalar_validation)

    elif method == "full_dr":
        (q_train, q_validation) = payload["q"]
        (h_train, h_all_train, h_validation, h_all_validation) = payload["h"]
        pseudo_all = h_all_train.copy()
        pseudo_all[np.arange(len(a_train)), a_train] += q_train * (y_train - h_train)
        response = pseudo_all @ V
        pseudo_validation = h_all_validation.copy()
        pseudo_validation[
            np.arange(len(a_validation)), a_validation
        ] += q_validation * (y_validation - h_validation)

        def score(prediction):
            return np.mean(pseudo_validation[np.arange(len(prediction)), prediction])

    elif method == "qi":
        response = V.shape[0] * y_train[:, None] * v_train
        sample_weight = inverse_probability(pi_train)
        validation_weight = inverse_probability(pi_validation)

        def score(prediction):
            matched = (prediction == a_validation) * validation_weight
            denominator = np.sum(matched)
            return (
                -np.inf
                if denominator <= 0
                else np.sum(y_validation * matched) / denominator
            )

    else:
        raise ValueError(f"Unknown continuous group-policy method: {method}")
    return (response, sample_weight, score)


def select_continuous_group_ratio(
    payloads: List[Dict[str, object]],
    feature_spec: str,
    method: str,
    V: np.ndarray,
    pi_spec: str = "X",
    ratios: Sequence[float] = GROUP_LASSO_RATIO_GRID,
) -> Tuple[float, Dict[str, object]]:
    ratios = np.asarray(ratios, dtype=float)
    fold_scores = np.empty((len(payloads), len(ratios)), dtype=float)
    fold_lambda_max = []
    v_tensor = torch.as_tensor(V, dtype=torch.float64)
    for fold_id, payload in enumerate(payloads):
        (x_train, x_validation) = payload["features"][feature_spec]
        (response, sample_weight, score) = _continuous_target_and_score(
            payload, method, V, pi_spec
        )
        lambda_max = group_lasso_lambda_max(x_train, response, sample_weight)
        fold_lambda_max.append(lambda_max)
        u_train = torch.as_tensor(
            np.column_stack([np.ones(len(x_train)), x_train]), dtype=torch.float64
        )
        u_validation = torch.as_tensor(
            np.column_stack([np.ones(len(x_validation)), x_validation]),
            dtype=torch.float64,
        )
        response_t = torch.as_tensor(response, dtype=torch.float64)
        weight_t = (
            None
            if sample_weight is None
            else torch.as_tensor(sample_weight, dtype=torch.float64)
        )
        for ratio_id, ratio in enumerate(ratios):
            penalty = float(ratio * lambda_max)
            if weight_t is None:
                coefficients = fit_proximal_itr_linear_fista(
                    u_train, response_t, penalty
                )
            else:
                coefficients = fit_weighted_proximal_itr_linear_fista(
                    u_train, response_t, weight_t, penalty
                )
            prediction = torch.argmax(
                u_validation @ coefficients @ v_tensor.T, dim=1
            ).numpy()
            fold_scores[fold_id, ratio_id] = float(score(prediction))
    mean_scores = np.mean(fold_scores, axis=0)
    maximum = np.max(mean_scores)
    tied = np.flatnonzero(np.isclose(mean_scores, maximum, rtol=0.0, atol=1e-12))
    selected_index = int(tied[np.argmax(ratios[tied])])
    return (
        float(ratios[selected_index]),
        {
            "search": "deterministic_grid",
            "grid_scale": "lambda_over_fold_specific_lambda_max",
            "ratio_grid": ratios.tolist(),
            "fold_lambda_max": [float(value) for value in fold_lambda_max],
            "fold_scores": fold_scores.tolist(),
            "mean_scores": mean_scores.tolist(),
            "selected_ratio": float(ratios[selected_index]),
            "tie_rule": "larger_ratio",
            "policy_folds": int(len(payloads)),
            "nuisance_folds_within_each_policy_train": int(SIM_NUISANCE_FOLDS),
        },
    )


def prepare_final_continuous_group_fit(
    final_payload: Dict[str, object],
    features: np.ndarray,
    method: str,
    V: np.ndarray,
    pi_spec: str,
    selected_ratio: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    (response, sample_weight, _) = _continuous_target_and_score(
        {
            **final_payload,
            "A_validation": final_payload["A_train"],
            "Y_validation": final_payload["Y_train"],
            "q": (
                (
                    final_payload.get("q", (None, None))[0],
                    final_payload.get("q", (None, None))[0],
                )
                if "q" in final_payload
                else None
            ),
            "h": (
                (
                    final_payload.get("h", (None, None, None, None))[0],
                    final_payload.get("h", (None, None, None, None))[1],
                    final_payload.get("h", (None, None, None, None))[0],
                    final_payload.get("h", (None, None, None, None))[1],
                )
                if "h" in final_payload
                else None
            ),
            "pi": {
                key: (value[0], value[0])
                for (key, value) in final_payload["pi"].items()
            },
        },
        method,
        V,
        pi_spec,
    )
    lambda_max = group_lasso_lambda_max(features, response, sample_weight)
    return (response, sample_weight, float(selected_ratio * lambda_max))


def select_continuous_xue_grid(
    payloads: List[Dict[str, object]],
    V: np.ndarray,
    fit_function,
    feature_spec: str = "X",
    propensity_spec: str = "XWZ",
) -> Tuple[float, float, Dict[str, object]]:
    candidates = [
        (float(lambda_value), float(b_value))
        for lambda_value in XUE_LAMBDA_GRID
        for b_value in XUE_B_GRID
    ]
    fold_scores = np.empty((len(payloads), len(candidates)), dtype=float)
    v_tensor = torch.as_tensor(V, dtype=torch.float64)
    for fold_id, payload in enumerate(payloads):
        (x_train, x_validation) = payload["features"][feature_spec]
        a_train = payload["A_train"]
        y_train = payload["Y_train"]
        a_validation = payload["A_validation"]
        y_validation = payload["Y_validation"]
        (pi_train, pi_validation) = payload["pi"][propensity_spec]
        train_weight = inverse_probability(pi_train)
        validation_weight = inverse_probability(pi_validation)
        u_train = torch.as_tensor(
            np.column_stack([np.ones(len(x_train)), x_train]), dtype=torch.float64
        )
        u_validation = torch.as_tensor(
            np.column_stack([np.ones(len(x_validation)), x_validation]),
            dtype=torch.float64,
        )
        v_observed = torch.as_tensor(V[a_train], dtype=torch.float64)
        weighted_outcome = torch.as_tensor(y_train * train_weight, dtype=torch.float64)
        for candidate_id, (lambda_value, b_value) in enumerate(candidates):
            coefficients = fit_function(
                u_train, weighted_outcome, v_observed, lambda_value, b=b_value, u0=0.0
            )
            prediction = torch.argmax(
                u_validation @ coefficients @ v_tensor.T, dim=1
            ).numpy()
            matched = (prediction == a_validation) * validation_weight
            denominator = np.sum(matched)
            fold_scores[fold_id, candidate_id] = (
                -np.inf
                if denominator <= 0
                else np.sum(y_validation * matched) / denominator
            )
    mean_scores = np.mean(fold_scores, axis=0)
    maximum = np.max(mean_scores)
    tied = np.flatnonzero(np.isclose(mean_scores, maximum, rtol=0.0, atol=1e-12))
    selected_index = sorted(
        tied.tolist(), key=lambda j: (-candidates[j][0], candidates[j][1])
    )[0]
    (selected_lambda, selected_b) = candidates[selected_index]
    return (
        selected_lambda,
        selected_b,
        {
            "search": "deterministic_grid",
            "lambda_grid": list(XUE_LAMBDA_GRID),
            "b_grid": list(XUE_B_GRID),
            "candidates": [list(candidate) for candidate in candidates],
            "fold_scores": fold_scores.tolist(),
            "mean_scores": mean_scores.tolist(),
            "selected_lambda": float(selected_lambda),
            "selected_b": float(selected_b),
            "tie_rule": "larger_lambda_then_smaller_b",
            "continuous_adaptation": True,
        },
    )


def select_survival_xue_grid(
    payloads: List[Dict[str, object]],
    V: np.ndarray,
    fit_function,
    t_eval: float,
    feature_spec: str = "X",
    propensity_spec: str = "XWZ",
):
    candidates = [
        (float(lambda_value), float(b_value))
        for lambda_value in XUE_LAMBDA_GRID
        for b_value in XUE_B_GRID
    ]
    scores = np.empty((len(payloads), len(candidates)), dtype=float)
    v_tensor = torch.as_tensor(V, dtype=torch.float64)
    for fold_id, payload in enumerate(payloads):
        (x_train, x_validation) = payload["features"][feature_spec]
        a_train = payload["A_train"]
        a_validation = payload["A_validation"]
        y_train = payload["Y_time_train"]
        delta_train = payload["Delta_train"]
        (pi_train, pi_validation) = payload["pi"][propensity_spec]
        y_validation_fixed = payload["Y_validation"]
        u_train = torch.as_tensor(
            np.column_stack([np.ones(len(x_train)), x_train]), dtype=torch.float64
        )
        u_validation = torch.as_tensor(
            np.column_stack([np.ones(len(x_validation)), x_validation]),
            dtype=torch.float64,
        )
        for candidate_id, (lambda_value, b_value) in enumerate(candidates):
            coefficients = fit_function(
                u_train,
                torch.as_tensor(a_train, dtype=torch.float64),
                torch.as_tensor(y_train, dtype=torch.float64),
                torch.as_tensor(delta_train, dtype=torch.float64),
                torch.as_tensor(inverse_probability(pi_train), dtype=torch.float64),
                v_tensor,
                float(t_eval),
                lambda_value,
                b_val=b_value,
                lr=0.1,
                epochs=250,
            )
            prediction = torch.argmax(
                u_validation @ coefficients @ v_tensor.T, dim=1
            ).numpy()
            scores[fold_id, candidate_id] = np.mean(
                y_validation_fixed
                * (prediction == a_validation)
                * inverse_probability(pi_validation)
            )
    mean_scores = np.mean(scores, axis=0)
    maximum = np.max(mean_scores)
    tied = np.flatnonzero(np.isclose(mean_scores, maximum, rtol=0.0, atol=1e-12))
    selected_index = sorted(
        tied.tolist(), key=lambda j: (-candidates[j][0], candidates[j][1])
    )[0]
    (selected_lambda, selected_b) = candidates[selected_index]
    return (
        selected_lambda,
        selected_b,
        {
            "search": "deterministic_grid",
            "lambda_grid": list(XUE_LAMBDA_GRID),
            "b_grid": list(XUE_B_GRID),
            "fold_scores": scores.tolist(),
            "mean_scores": mean_scores.tolist(),
            "selected_lambda": selected_lambda,
            "selected_b": selected_b,
            "tie_rule": "larger_lambda_then_smaller_b",
            "validation_target": "nested_censoring_fixed_time_ipcw",
        },
    )


def goldberg_alpha_max(design, outcome, sample_weight):
    design = np.asarray(design, dtype=float)
    outcome = np.asarray(outcome, dtype=float).reshape(-1)
    weight = np.asarray(sample_weight, dtype=float).reshape(-1)
    x_mean = np.sum(weight[:, None] * design, axis=0) / np.sum(weight)
    y_mean = np.sum(weight * outcome) / np.sum(weight)
    gradient = (design - x_mean).T @ (weight * (outcome - y_mean)) / len(outcome)
    return float(max(np.max(np.abs(gradient)), np.finfo(float).eps))


def _goldberg_target_km(treatment, y_time, delta, train_rows, validation_rows, t_eval):
    epsilon = 1e-05
    query = np.clip(np.minimum(y_time, t_eval) - epsilon, 0.0, None)
    km = KaplanMeierFitter()
    km.fit(y_time[train_rows], 1 - delta[train_rows])
    survival = km.survival_function_at_times(query[validation_rows]).to_numpy()
    survival = np.clip(survival, CENSOR_SURVIVAL_LOWER, 1.0)
    valid = (delta[validation_rows] == 1) | (y_time[validation_rows] >= t_eval)
    target = np.minimum(y_time[validation_rows], t_eval)
    weight = np.zeros(len(validation_rows), dtype=float)
    weight[valid] = 1.0 / survival[valid]
    return (target, weight, valid)


def select_goldberg_ratio_grid(
    raw,
    treatment,
    y_time,
    delta,
    linearity,
    t_eval,
    plan: FrozenSimulationSplitPlan,
    feature_spec="X",
    ratios=LASSO_RATIO_GRID,
):
    raw = {key: _as_2d(value) for (key, value) in raw.items()}
    treatment = _as_1d(treatment, int)
    y_time = _as_1d(y_time, float)
    delta = _as_1d(delta, int)
    ratios = np.asarray(ratios, dtype=float)
    encoder = OneHotEncoder(
        drop="first",
        sparse_output=False,
        categories=[range(int(np.max(treatment)) + 1)],
    ).fit(treatment.reshape(-1, 1))

    def design(x, a):
        dummy = encoder.transform(np.asarray(a).reshape(-1, 1))
        interaction = (x[:, :, None] * dummy[:, None, :]).reshape(len(x), -1)
        return np.column_stack([x, dummy, interaction])

    scores = np.empty((len(plan.policy_splits), len(ratios)), dtype=float)
    fold_alpha_max = []
    for fold_id, (train_rows, validation_rows) in enumerate(plan.policy_splits):
        (x_train, x_validation) = fit_transform_policy_features(
            _subset_raw(raw, train_rows),
            _subset_raw(raw, validation_rows),
            feature_spec,
            linearity,
        )
        target_train = np.empty(len(train_rows), dtype=float)
        weight_train = np.empty(len(train_rows), dtype=float)
        q_splitter = StratifiedKFold(
            n_splits=SIM_NUISANCE_FOLDS,
            shuffle=True,
            random_state=plan.seed + 5000 + fold_id,
        )
        for q_train_local, q_validation_local in q_splitter.split(
            np.zeros(len(train_rows)), treatment[train_rows]
        ):
            (target, weight, _valid) = _goldberg_target_km(
                treatment,
                y_time,
                delta,
                train_rows[q_train_local],
                train_rows[q_validation_local],
                t_eval,
            )
            target_train[q_validation_local] = target
            weight_train[q_validation_local] = weight
        valid_train = weight_train > 0
        phi_train = design(x_train[valid_train], treatment[train_rows][valid_train])
        alpha_max = goldberg_alpha_max(
            phi_train, target_train[valid_train], weight_train[valid_train]
        )
        fold_alpha_max.append(alpha_max)
        (target_validation, weight_validation, valid_validation) = _goldberg_target_km(
            treatment, y_time, delta, train_rows, validation_rows, t_eval
        )
        phi_validation = design(
            x_validation[valid_validation], treatment[validation_rows][valid_validation]
        )
        for ratio_id, ratio in enumerate(ratios):
            model = Lasso(
                alpha=float(ratio * alpha_max),
                fit_intercept=True,
                max_iter=50000,
                random_state=42,
            )
            model.fit(
                phi_train,
                target_train[valid_train],
                sample_weight=weight_train[valid_train],
            )
            residual = target_validation[valid_validation] - model.predict(
                phi_validation
            )
            scores[fold_id, ratio_id] = np.average(
                residual**2, weights=weight_validation[valid_validation]
            )
    mean_scores = np.mean(scores, axis=0)
    minimum = np.min(mean_scores)
    tied = np.flatnonzero(np.isclose(mean_scores, minimum, rtol=0.0, atol=1e-12))
    selected_index = int(tied[np.argmax(ratios[tied])])
    return (
        float(ratios[selected_index]),
        {
            "search": "deterministic_grid",
            "grid_scale": "alpha_over_q_fold_specific_alpha_max",
            "ratio_grid": ratios.tolist(),
            "fold_alpha_max": fold_alpha_max,
            "fold_weighted_mse": scores.tolist(),
            "mean_weighted_mse": mean_scores.tolist(),
            "selected_ratio": float(ratios[selected_index]),
            "tie_rule": "larger_ratio",
            "km_nested_inside_goldberg_q_training_fold": True,
        },
    )


def survival_cox_group_lambda_max(
    features, treatment, y_time, delta, propensity_weight, V
):
    features = np.asarray(features, dtype=float)
    treatment = _as_1d(treatment, int)
    y_time = _as_1d(y_time, float)
    delta = _as_1d(delta, float)
    weight = _as_1d(propensity_weight, float)
    u = np.column_stack([np.ones(len(features)), features])
    observed_vertex = np.asarray(V, dtype=float)[treatment]
    x_star = np.einsum("ni,nj->nij", u, observed_vertex).reshape(len(u), -1)
    order = np.argsort(y_time)[::-1]
    x_ordered = x_star[order]
    event_ordered = delta[order]
    weight_ordered = weight[order]
    cumulative_x = np.cumsum(x_ordered, axis=0)
    risk_mean = cumulative_x / np.arange(1, len(u) + 1)[:, None]
    gradient = -np.mean(
        (weight_ordered * event_ordered)[:, None] * (x_ordered - risk_mean), axis=0
    )
    grouped = gradient.reshape(u.shape[1], np.asarray(V).shape[1])
    value = np.max(np.linalg.norm(grouped[1:], axis=1))
    return float(max(value, np.finfo(float).eps))


def select_survival_qi_ratio_grid(
    payloads: List[Dict[str, object]],
    V: np.ndarray,
    fit_function,
    t_eval: float,
    feature_spec="X",
    propensity_spec="XWZ",
    ratios=GROUP_LASSO_RATIO_GRID,
):
    ratios = np.asarray(ratios, dtype=float)
    scores = np.empty((len(payloads), len(ratios)), dtype=float)
    fold_lambda_max = []
    v_tensor = torch.as_tensor(V, dtype=torch.float64)
    for fold_id, payload in enumerate(payloads):
        (x_train, x_validation) = payload["features"][feature_spec]
        a_train = payload["A_train"]
        a_validation = payload["A_validation"]
        y_train = payload["Y_time_train"]
        delta_train = payload["Delta_train"]
        (pi_train, pi_validation) = payload["pi"][propensity_spec]
        propensity_train = inverse_probability(pi_train)
        propensity_validation = inverse_probability(pi_validation)
        lambda_max = survival_cox_group_lambda_max(
            x_train, a_train, y_train, delta_train, propensity_train, V
        )
        fold_lambda_max.append(lambda_max)
        u_train = torch.as_tensor(
            np.column_stack([np.ones(len(x_train)), x_train]), dtype=torch.float64
        )
        u_validation = torch.as_tensor(
            np.column_stack([np.ones(len(x_validation)), x_validation]),
            dtype=torch.float64,
        )
        for ratio_id, ratio in enumerate(ratios):
            coefficients = fit_function(
                u_train,
                torch.as_tensor(a_train, dtype=torch.float64),
                torch.as_tensor(y_train, dtype=torch.float64),
                torch.as_tensor(delta_train, dtype=torch.float64),
                torch.as_tensor(propensity_train, dtype=torch.float64),
                v_tensor,
                float(ratio * lambda_max),
                lr=0.1,
                epochs=250,
            )
            prediction = torch.argmax(
                -(u_validation @ coefficients) @ v_tensor.T, dim=1
            ).numpy()
            scores[fold_id, ratio_id] = np.mean(
                payload["Y_validation"]
                * (prediction == a_validation)
                * propensity_validation
            )
    mean_scores = np.mean(scores, axis=0)
    maximum = np.max(mean_scores)
    tied = np.flatnonzero(np.isclose(mean_scores, maximum, rtol=0.0, atol=1e-12))
    selected_index = int(tied[np.argmax(ratios[tied])])
    return (
        float(ratios[selected_index]),
        {
            "search": "deterministic_grid",
            "grid_scale": "lambda_over_fold_specific_weighted_cox_lambda_max",
            "ratio_grid": ratios.tolist(),
            "fold_lambda_max": fold_lambda_max,
            "fold_scores": scores.tolist(),
            "mean_scores": mean_scores.tolist(),
            "selected_ratio": float(ratios[selected_index]),
            "tie_rule": "larger_ratio",
            "validation_target": "nested_censoring_ipcw_times_inverse_propensity",
        },
    )
