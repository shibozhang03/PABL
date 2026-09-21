"""Nested cross-fitting for the three reported MIMIC-IV analyses."""

from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from pabl.common import generate_simplex
from . import columns, preprocessing, continuous_nuisance, survival_nuisance
from . import continuous_policy, survival_policy
from .censoring import generate_censoring
from .encoding import ARM_ORDER

SEED = 20260813
H_ALPHA = tuple(np.geomspace(0.1, 100000.0, 5).tolist())
Q_ALPHA = (
    0.1,
    1.0,
    float(np.sqrt(10.0)),
    10.0,
    100.0,
    1000.0,
    float(1000.0 * np.sqrt(10.0)),
    10000.0,
    100000.0,
)
BLOOD_X = [
    "z_blood_specimens_24h_cap4",
    "z_blood_specimens_ge2",
    "z_blood_specimens_ge3",
]
BASE_X = columns.NUMERIC_X + BLOOD_X
LABELS = {
    "Proximal-h(X)": "PABL-h(X)",
    "Proximal-q(X)": "PABL-q(X)",
    "DR Proximal(X)": "DR-PABL(X)",
    "Qi AD(X)": "Group-AD(X)",
    "Xue-style Angle(X)": "Angle-IPW(X)",
    "l1-PLS(X)": "l1-PLS(X)",
    "Xue survival Angle(X)": "Angle-KM(X)",
    "Goldberg censored Q(X)": "Cens-Q(X)",
}


def prepare_cohort(frame, *, verify=True):
    """Apply the published outcome-free support restriction and five master folds."""
    frame = frame.copy().reset_index(drop=True)
    counts = frame["blood_specimens_24h"].astype(float)
    frame[BLOOD_X[0]] = np.minimum(counts, 4)
    frame[BLOOD_X[1]] = (counts >= 2).astype(int)
    frame[BLOOD_X[2]] = (counts >= 3).astype(int)
    frame = preprocessing.add_compact_proxy_features(frame)
    probabilities = [
        preprocessing.crossfit_support_probabilities(
            frame,
            base_numeric=BASE_X,
            categorical=columns.CATEGORICAL_X,
            learner=learner,
            seed=SEED,
            c_value=0.03,
        )
        for learner in ("logistic", "boosted")
    ]
    keep = (probabilities[0].min(axis=1) >= 0.05) & (
        probabilities[1].min(axis=1) >= 0.05
    )
    selected = frame.loc[keep].copy().reset_index(drop=True)
    treatment = pd.Categorical(selected["arm"], categories=ARM_ORDER).codes
    if verify and (
        len(selected) != 11944 or np.bincount(treatment).tolist() != [3833, 4572, 3539]
    ):
        raise ValueError(
            "The cohort does not match the published analysis (11,944 subjects)."
        )
    fold_id = np.full(len(selected), -1, dtype=int)
    for i, (_, heldout) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=SEED).split(
            np.zeros((len(selected), 1)), treatment
        )
    ):
        fold_id[heldout] = i
    selected["_strict_master_fold"] = fold_id
    return (selected, treatment, fold_id)


def _bridge_seed(excluded, layer):
    value = SEED + layer * 1000000
    for position, fold in enumerate(excluded, start=1):
        value += position * 10 ** (2 * position) * (int(fold) + 1)
    return int(value)


def analyze(selected, treatment, fold_id, outcome_name, device="cpu"):
    """Return table entries and aggregate agreement; retain no patient-level output."""
    survival = outcome_name != "daoh28"
    engine = survival_nuisance if survival else continuous_nuisance
    policies = survival_policy if survival else continuous_policy
    outcome = selected["y_survival90" if survival else "y_daoh28"].to_numpy(dtype=float)
    times = selected["survival90_observed_time_days"].to_numpy(dtype=float)
    death = selected["survival90_death_event"].to_numpy(dtype=int)
    censor = np.zeros(len(selected), dtype=int)
    coverage = None
    if outcome_name == "censored":
        (coverage, times, death, censor) = generate_censoring(selected)
    rows = [np.flatnonzero(fold_id == i) for i in range(5)]
    args = dict(
        h_components=50,
        q_components=100,
        h_alpha_scales=H_ALPHA,
        q_alpha_scales=Q_ALPHA,
        kernel_geometry="block_dimension_normalized",
        propensity_c=0.03,
    )
    if survival:
        args.update(
            coverage_reaches_horizon=coverage,
            master_fold_id=fold_id if coverage is not None else None,
            censoring_learner="arm_only_jeffreys",
            censoring_lower_clip=0.05,
        )

    def nuisance(excluded):
        train = np.flatnonzero(~np.isin(fold_id, excluded))
        predict = np.concatenate([rows[i] for i in excluded])
        return engine.fit_compact_nuisances_predict(
            selected,
            treatment,
            outcome,
            train,
            predict,
            BASE_X,
            seed=_bridge_seed(excluded, len(excluded)),
            **args,
        )

    pair = {}
    for excluded in combinations(range(5), 2):
        bundle = nuisance(excluded)
        for fold, values in engine._slice_bundle(bundle, excluded, rows).items():
            pair[excluded + (fold,)] = values
    final_train = {outer: engine._empty_bundle(len(selected)) for outer in range(5)}
    inner_train = {
        (outer, inner): engine._empty_bundle(len(selected))
        for outer in range(5)
        for inner in range(5)
        if inner != outer
    }
    for outer in range(5):
        for other in range(5):
            if other != outer:
                key = tuple(sorted((outer, other))) + (other,)
                engine._assign_bundle(final_train[outer], rows[other], pair[key])
    for excluded in combinations(range(5), 3):
        slices = engine._slice_bundle(nuisance(excluded), excluded, rows)
        for outer in excluded:
            for inner in excluded:
                if inner != outer:
                    other = next((i for i in excluded if i not in (outer, inner)))
                    engine._assign_bundle(
                        inner_train[outer, inner], rows[other], slices[other]
                    )

    def payload(train, validation, train_bundle, valid_bundle, fold):
        additional = (times, death, censor) if survival else ()
        return engine._make_policy_fold(
            selected,
            train,
            validation,
            BASE_X,
            outcome,
            treatment,
            *additional,
            train_bundle,
            valid_bundle,
            fold,
        )[0]

    (scores, recommendations) = ({}, {})
    simplex = generate_simplex(3)
    device = torch.device(device)
    for outer in range(5):
        train = np.flatnonzero(fold_id != outer)
        heldout = rows[outer]
        evaluation = nuisance((outer,))
        tuning = []
        for inner in range(5):
            if inner == outer:
                continue
            key = tuple(sorted((outer, inner))) + (inner,)
            tuning.append(
                payload(
                    np.flatnonzero((fold_id != outer) & (fold_id != inner)),
                    rows[inner],
                    inner_train[outer, inner],
                    pair[key],
                    inner,
                )
            )
        final = payload(train, heldout, final_train[outer], evaluation, outer)
        fitted = {}
        registry = [
            ("Proximal-h(X)", "h"),
            ("Proximal-q(X)", "q"),
            ("DR Proximal(X)", "dr"),
        ]
        if not survival:
            registry.append(("Qi AD(X)", "qi"))
        for name, method in registry:
            fitted[name] = policies.fit_group_policy(
                name,
                method,
                "x",
                tuning,
                final,
                simplex,
                ratios=policies.GROUP_RATIO_GRID,
                selection_rule=(
                    "one_standard_error_paired_static"
                    if method == "q"
                    else "maximum_mean"
                ),
                reference_arm=1,
            )
        if survival:
            fitted["Qi AD(X)"] = policies.fit_qi_survival_policy(
                tuning, final, simplex, device=device
            )
            fitted["Xue survival Angle(X)"] = policies.fit_xue_survival_policy(
                tuning, final, simplex, device=device
            )
            fitted["Goldberg censored Q(X)"] = policies.fit_goldberg_survival_policy(
                tuning, final, 3
            )
        else:
            fitted["Xue-style Angle(X)"] = policies.fit_xue_policy(
                tuning,
                final,
                simplex,
                device=device,
                tuning_epochs=1000,
                final_epochs=1000,
                learning_rate=0.01,
            )
            fitted["l1-PLS(X)"] = policies.fit_l1_pls_policy(
                tuning, final, 3, ratios=policies.L1_RATIO_GRID
            )
        response = evaluation["ipcw_outcome"] if survival else outcome[heldout]
        for name, fit in fitted.items():
            scores.setdefault(name, np.full(len(selected), np.nan))
            recommendations.setdefault(name, np.full(len(selected), -1, dtype=int))
            scores[name][heldout] = engine._score_policy(
                fit.recommendation, response, treatment[heldout], evaluation
            )["dr"]
            recommendations[name][heldout] = fit.recommendation
    labels = dict(LABELS)
    if survival:
        labels["Qi AD(X)"] = "Surv-AD(X)"
    table = []
    for name, values in scores.items():
        if not np.isfinite(values).all() or np.any(recommendations[name] < 0):
            raise RuntimeError("Incomplete held-out results.")
        proportions = np.bincount(recommendations[name], minlength=3) / len(selected)
        table.append(
            dict(
                method=labels[name],
                value=float(values.mean()),
                standard_error=float(values.std(ddof=1) / np.sqrt(len(values))),
                **{arm: float(100 * p) for (arm, p) in zip(ARM_ORDER, proportions)},
            )
        )
    table = (
        pd.DataFrame(table).sort_values("value", ascending=False).reset_index(drop=True)
    )
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    proximal = ["Proximal-h(X)", "DR Proximal(X)", "Proximal-q(X)"]
    agreement = {
        labels[left]
        + " | "
        + labels[right]: float(
            100 * np.mean(recommendations[left] == recommendations[right])
        )
        for (left, right) in combinations(proximal, 2)
    }
    return (table, agreement)
