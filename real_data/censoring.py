"""The fixed semi-synthetic censoring mechanism and its cross-fitted estimator."""

from __future__ import annotations
import hashlib
import numpy as np

CENSOR_SEED = 20260820
SURVIVAL_PROBABILITY = 0.7


def _uniform(subject, label):
    payload = f"{CENSOR_SEED}:{int(subject)}:{label}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (integer + 0.5) / float(2**64)


def generate_censoring(frame):
    """Generate censoring independently of outcomes, with stable subject ordering."""
    subject = frame["subject_id"].to_numpy(dtype=np.int64)
    if len(np.unique(subject)) != len(subject):
        raise ValueError("Subjects must be unique.")
    covered = np.array(
        [_uniform(i, "coverage") < SURVIVAL_PROBABILITY for i in subject]
    )
    u = np.array([_uniform(i, "time") for i in subject])
    censor_time = np.where(covered, 91.0 + 30.0 * u, 1.0 + 19.0 * u)
    death = frame["survival90_death_event"].to_numpy(dtype=int)
    time = frame["survival90_observed_time_days"].to_numpy(dtype=float)
    death_time = np.where(death == 1, time, np.inf)
    observed_death = (death == 1) & (death_time <= censor_time)
    observed_time = np.minimum(np.minimum(death_time, censor_time), 90.0)
    censored = ~observed_death & (censor_time < 90.0)
    return (
        covered.astype(int),
        observed_time,
        observed_death.astype(int),
        censored.astype(int),
    )


def _predict_g(treatment, target, train, predict):
    if np.intersect1d(train, predict).size:
        raise ValueError("Censoring training and prediction rows overlap.")
    probability = []
    for arm in range(3):
        mask = treatment[train] == arm
        count = int(mask.sum())
        if count == 0:
            raise ValueError("A censoring training block is missing a treatment arm.")
        probability.append((int(target[train][mask].sum()) + 0.5) / (count + 1.0))
    return np.asarray(probability)[treatment[predict]]


def fit_strict_g_block(
    frame,
    treatment,
    coverage_reaches_horizon,
    master_fold_id,
    train,
    predict,
    base_numeric_x,
    *,
    lower_clip=0.05,
    **kwargs,
):
    """Estimate arm-specific G with Jeffreys smoothing inside each training block."""
    treatment = np.asarray(treatment, dtype=int)
    target = np.asarray(coverage_reaches_horizon, dtype=int)
    fold = np.asarray(master_fold_id, dtype=int)
    train = np.asarray(train, dtype=int)
    predict = np.asarray(predict, dtype=int)
    included = sorted(np.unique(fold[train]).tolist())
    if len(included) < 2:
        raise ValueError(
            "Censoring cross-fitting requires at least two training folds."
        )
    raw_train = np.full(len(train), np.nan)
    for heldout in included:
        mask = fold[train] == heldout
        raw_train[mask] = _predict_g(treatment, target, train[~mask], train[mask])
    raw_predict = _predict_g(treatment, target, train, predict)
    if not np.isfinite(raw_train).all():
        raise ValueError("Incomplete censoring predictions.")
    return {
        "train_oof_raw": raw_train,
        "train_oof_clipped": np.clip(raw_train, lower_clip, 1.0),
        "predict_raw": raw_predict,
        "predict_clipped": np.clip(raw_predict, lower_clip, 1.0),
    }
