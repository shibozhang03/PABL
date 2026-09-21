from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, MultiTaskLasso
from sklearn.preprocessing import OneHotEncoder

GROUP_RATIO_GRID = tuple((float(2.0 ** (-index)) for index in range(11)))
L1_RATIO_GRID = tuple((float(10.0 ** (-index / 4.0)) for index in range(13)))
XUE_LAMBDA_GRID = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)
XUE_B_GRID = tuple((float(index) / 2.0 for index in range(1, 21)))
SURVIVAL_LAMBDA_GRID = XUE_LAMBDA_GRID
SURVIVAL_TUNING_EPOCHS = 250
SURVIVAL_FINAL_EPOCHS = 400
SURVIVAL_TUNING_LEARNING_RATE = 0.1
SURVIVAL_FINAL_LEARNING_RATE = 0.05
SURVIVAL_HORIZON_DAYS = 90.0
GROUP_MAX_ITER = 100000
GROUP_TOL = 1e-07
L1_MAX_ITER = 100000
L1_TOL = 1e-07
PROPENSITY_FLOOR = 0.05
IPW_CAP = 20.0


@dataclass(frozen=True)
class PolicyFold:
    features_train: dict[str, np.ndarray]
    features_validation: dict[str, np.ndarray]
    outcome_train: np.ndarray
    outcome_validation: np.ndarray
    treatment_train: np.ndarray
    treatment_validation: np.ndarray
    observed_time_train: np.ndarray
    observed_time_validation: np.ndarray
    death_event_train: np.ndarray
    death_event_validation: np.ndarray
    coverage_censor_event_train: np.ndarray
    coverage_censor_event_validation: np.ndarray
    h_train: np.ndarray
    h_validation: np.ndarray
    q_train: np.ndarray
    q_validation: np.ndarray
    propensity_x_train: np.ndarray
    propensity_x_validation: np.ndarray
    propensity_xwz_train: np.ndarray
    propensity_xwz_validation: np.ndarray
    policy_fold: int

    def validate(self, arm_count: int) -> None:
        train_n = len(self.outcome_train)
        validation_n = len(self.outcome_validation)
        for name, values, expected in (
            ("outcome_train", self.outcome_train, train_n),
            ("treatment_train", self.treatment_train, train_n),
            ("observed_time_train", self.observed_time_train, train_n),
            ("death_event_train", self.death_event_train, train_n),
            ("coverage_censor_event_train", self.coverage_censor_event_train, train_n),
            ("h_train", self.h_train, train_n),
            ("q_train", self.q_train, train_n),
            ("propensity_x_train", self.propensity_x_train, train_n),
            ("propensity_xwz_train", self.propensity_xwz_train, train_n),
            ("outcome_validation", self.outcome_validation, validation_n),
            ("treatment_validation", self.treatment_validation, validation_n),
            ("observed_time_validation", self.observed_time_validation, validation_n),
            ("death_event_validation", self.death_event_validation, validation_n),
            (
                "coverage_censor_event_validation",
                self.coverage_censor_event_validation,
                validation_n,
            ),
            ("h_validation", self.h_validation, validation_n),
            ("q_validation", self.q_validation, validation_n),
            ("propensity_x_validation", self.propensity_x_validation, validation_n),
            ("propensity_xwz_validation", self.propensity_xwz_validation, validation_n),
        ):
            if len(values) != expected:
                raise ValueError(f"{name} has the wrong row count")
            if not np.isfinite(np.asarray(values, dtype=float)).all():
                raise ValueError(f"{name} contains nonfinite values")
        if (
            self.h_train.shape[1] != arm_count
            or self.h_validation.shape[1] != arm_count
        ):
            raise ValueError("h nuisance has the wrong treatment dimension")
        for name, event in (
            ("death_event_train", self.death_event_train),
            ("death_event_validation", self.death_event_validation),
            ("coverage_censor_event_train", self.coverage_censor_event_train),
            ("coverage_censor_event_validation", self.coverage_censor_event_validation),
        ):
            if not np.isin(np.asarray(event, dtype=int), (0, 1)).all():
                raise ValueError(f"{name} must be binary")
        if np.any(
            np.asarray(self.death_event_train, dtype=int)
            + np.asarray(self.coverage_censor_event_train, dtype=int)
            > 1
        ) or np.any(
            np.asarray(self.death_event_validation, dtype=int)
            + np.asarray(self.coverage_censor_event_validation, dtype=int)
            > 1
        ):
            raise ValueError("death and censoring events cannot coincide")
        for name, time in (
            ("observed_time_train", self.observed_time_train),
            ("observed_time_validation", self.observed_time_validation),
        ):
            values = np.asarray(time, dtype=float)
            if np.any(values < 0) or np.any(values > SURVIVAL_HORIZON_DAYS):
                raise ValueError(f"{name} is outside [0, 90]")
        for name, propensity in (
            ("propensity_x_train", self.propensity_x_train),
            ("propensity_x_validation", self.propensity_x_validation),
            ("propensity_xwz_train", self.propensity_xwz_train),
            ("propensity_xwz_validation", self.propensity_xwz_validation),
        ):
            probability = np.asarray(propensity, dtype=float)
            if np.any(probability <= 0) or np.any(probability > 1):
                raise ValueError(f"{name} is outside (0, 1]")
        for name, assigned in (
            ("treatment_train", self.treatment_train),
            ("treatment_validation", self.treatment_validation),
        ):
            assigned = np.asarray(assigned, dtype=int)
            if np.any(assigned < 0) or np.any(assigned >= arm_count):
                raise ValueError(f"{name} contains an invalid arm index")
        for scope, matrices, expected in (
            ("train", self.features_train, train_n),
            ("validation", self.features_validation, validation_n),
        ):
            for spec, matrix in matrices.items():
                matrix = np.asarray(matrix, dtype=float)
                if (
                    matrix.ndim != 2
                    or len(matrix) != expected
                    or (not np.isfinite(matrix).all())
                ):
                    raise ValueError(f"invalid {scope} policy matrix for {spec}")


@dataclass(frozen=True)
class FittedPolicy:
    name: str
    recommendation: np.ndarray
    coefficient: np.ndarray
    intercept: np.ndarray


def inverse_probability(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=float).reshape(-1)
    return np.minimum(1.0 / np.clip(probability, PROPENSITY_FLOOR, 1.0), IPW_CAP)


def _censoring_branch(folds: list[PolicyFold], final_fold: PolicyFold) -> str:
    arrays = [final_fold.coverage_censor_event_train]
    arrays.extend((fold.coverage_censor_event_train for fold in folds))
    return (
        "strict_cross_fitted_IPCW"
        if any((np.any(np.asarray(values, dtype=int) == 1) for values in arrays))
        else "G_equals_1"
    )


def dr_potential(
    outcome: np.ndarray,
    treatment: np.ndarray,
    h_all: np.ndarray,
    q_observed: np.ndarray,
) -> np.ndarray:
    outcome = np.asarray(outcome, dtype=float).reshape(-1)
    treatment = np.asarray(treatment, dtype=int).reshape(-1)
    h_all = np.asarray(h_all, dtype=float)
    q_observed = np.asarray(q_observed, dtype=float).reshape(-1)
    result = h_all.copy()
    rows = np.arange(len(outcome))
    result[rows, treatment] += q_observed * (outcome - h_all[rows, treatment])
    if not np.isfinite(result).all():
        raise RuntimeError("proximal DR score contains nonfinite values")
    return result


def _weighted_group_lambda_max(
    features: np.ndarray, response: np.ndarray, sample_weight: np.ndarray | None
) -> float:
    x = np.asarray(features, dtype=float)
    y = np.asarray(response, dtype=float)
    if sample_weight is None:
        weight = np.ones(len(x), dtype=float)
    else:
        weight = np.asarray(sample_weight, dtype=float).reshape(-1)
    if (
        x.ndim != 2
        or y.ndim != 2
        or len(x) != len(y)
        or (len(weight) != len(x))
        or np.any(weight < 0)
        or (not np.isfinite(weight).all())
        or (float(np.sum(weight)) <= 0)
    ):
        raise ValueError("invalid weighted group-lasso inputs")
    x_mean = np.sum(weight[:, None] * x, axis=0) / np.sum(weight)
    y_mean = np.sum(weight[:, None] * y, axis=0) / np.sum(weight)
    gradient = (x - x_mean).T @ (weight[:, None] * (y - y_mean)) / len(x)
    value = float(np.max(np.linalg.norm(gradient, axis=1)))
    return max(value, np.finfo(float).eps)


def _group_path(
    features: np.ndarray,
    response: np.ndarray,
    alphas: np.ndarray,
    sample_weight: np.ndarray | None = None,
):
    x = np.asarray(features, dtype=float)
    y = np.asarray(response, dtype=float)
    alphas = np.asarray(alphas, dtype=float).reshape(-1)
    if np.any(alphas <= 0) or not np.isfinite(alphas).all():
        raise ValueError("group-lasso alphas must be finite and positive")
    if sample_weight is None:
        weight = np.ones(len(x), dtype=float)
    else:
        weight = np.asarray(sample_weight, dtype=float).reshape(-1)
    weight_sum = float(np.sum(weight))
    x_mean = np.sum(weight[:, None] * x, axis=0) / weight_sum
    y_mean = np.sum(weight[:, None] * y, axis=0) / weight_sum
    root_weight = np.sqrt(weight)[:, None]
    centered_x = (x - x_mean) * root_weight
    centered_y = (y - y_mean) * root_weight
    model = MultiTaskLasso(
        alpha=float(alphas[0]),
        fit_intercept=False,
        max_iter=GROUP_MAX_ITER,
        tol=GROUP_TOL,
        warm_start=True,
        selection="cyclic",
    )
    coefficients = np.empty((len(alphas), y.shape[1], x.shape[1]), dtype=float)
    intercepts = np.empty((len(alphas), y.shape[1]), dtype=float)
    for index, alpha in enumerate(alphas):
        model.set_params(alpha=float(alpha))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(centered_x, centered_y)
        convergence_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, ConvergenceWarning)
        ]
        coefficient = np.asarray(model.coef_, dtype=float)
        intercept = y_mean - x_mean @ coefficient.T
        dual_gap = float(np.max(np.abs(np.asarray(model.dual_gap_))))
        iteration = int(model.n_iter_)
        if (
            convergence_warnings
            or iteration >= GROUP_MAX_ITER
            or (not np.isfinite(coefficient).all())
            or (not np.isfinite(intercept).all())
            or (not np.isfinite(dual_gap))
        ):
            raise RuntimeError(
                f"group-lasso coordinate descent failed: alpha={alpha}; iterations={iteration}; dual_gap={dual_gap}"
            )
        coefficients[index] = coefficient
        intercepts[index] = intercept
    return (coefficients, intercepts)


def _recommend_group(
    features: np.ndarray,
    coefficient: np.ndarray,
    intercept: np.ndarray,
    simplex: np.ndarray,
) -> np.ndarray:
    decision = np.asarray(features, dtype=float) @ coefficient.T + intercept
    return np.argmax(decision @ simplex.T, axis=1).astype(np.int64)


def _group_payload(
    fold: PolicyFold, method: str, feature_spec: str, simplex: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, object]:
    a_train = fold.treatment_train
    a_validation = fold.treatment_validation
    y_train = fold.outcome_train
    y_validation = fold.outcome_validation
    vertices = simplex[a_train]
    sample_weight = None
    if method == "h":
        response = fold.h_train @ simplex

        def score(recommendation: np.ndarray) -> float:
            return float(
                np.mean(
                    fold.h_validation[np.arange(len(recommendation)), recommendation]
                )
            )

    elif method == "q":
        response = y_train[:, None] * fold.q_train[:, None] * vertices

        def score(recommendation: np.ndarray) -> float:
            return float(
                np.mean(
                    y_validation * (recommendation == a_validation) * fold.q_validation
                )
            )

    elif method == "dr":
        response = dr_potential(y_train, a_train, fold.h_train, fold.q_train) @ simplex
        validation = dr_potential(
            y_validation, a_validation, fold.h_validation, fold.q_validation
        )

        def score(recommendation: np.ndarray) -> float:
            return float(
                np.mean(validation[np.arange(len(recommendation)), recommendation])
            )

    elif method == "qi":
        response = simplex.shape[0] * y_train[:, None] * vertices
        sample_weight = inverse_probability(fold.propensity_xwz_train)
        validation_weight = inverse_probability(fold.propensity_xwz_validation)

        def score(recommendation: np.ndarray) -> float:
            matched = (recommendation == a_validation) * validation_weight
            denominator = float(np.sum(matched))
            if denominator <= 0:
                return float("-inf")
            return float(np.sum(y_validation * matched) / denominator)

    else:
        raise ValueError(f"unknown group-policy method: {method}")
    return (
        fold.features_train[feature_spec],
        fold.features_validation[feature_spec],
        response,
        sample_weight,
        score,
    )


def fit_group_policy(
    name: str,
    method: str,
    feature_spec: str,
    folds: list[PolicyFold],
    final_fold: PolicyFold,
    simplex: np.ndarray,
    ratios: tuple[float, ...] = GROUP_RATIO_GRID,
    selection_rule: str = "maximum_mean",
    lcb_standard_error_multiplier: float = 1.0,
    reference_arm: int = 1,
) -> FittedPolicy:
    arm_count = simplex.shape[0]
    for fold in folds:
        fold.validate(arm_count)
    final_fold.validate(arm_count)
    if not 0 <= int(reference_arm) < arm_count:
        raise ValueError("reference_arm is outside the treatment range")
    ratios_array = np.asarray(ratios, dtype=float)
    if (
        len(ratios_array) == 0
        or np.any(ratios_array <= 0)
        or (not np.isfinite(ratios_array).all())
        or np.any(np.diff(ratios_array) >= 0)
    ):
        raise ValueError("penalty ratios must be finite, positive and decreasing")
    fold_values = np.empty((len(folds), len(ratios_array)), dtype=float)
    fold_active_groups = np.empty_like(fold_values, dtype=int)
    fold_recommendation_counts = np.empty(
        (len(folds), len(ratios_array), arm_count), dtype=int
    )
    fold_reference_values = np.empty(len(folds), dtype=float)
    fold_lambda_max: list[float] = []
    for fold_index, fold in enumerate(folds):
        (x_train, x_validation, response, weight, score) = _group_payload(
            fold, method, feature_spec, simplex
        )
        lambda_max = _weighted_group_lambda_max(x_train, response, weight)
        alphas = ratios_array * lambda_max
        (coefficient_path, intercept_path) = _group_path(
            x_train, response, alphas, weight
        )
        fold_lambda_max.append(lambda_max)
        fold_reference_values[fold_index] = score(
            np.full(len(fold.outcome_validation), reference_arm, dtype=int)
        )
        for ratio_index in range(len(ratios_array)):
            recommendation = _recommend_group(
                x_validation,
                coefficient_path[ratio_index],
                intercept_path[ratio_index],
                simplex,
            )
            fold_values[fold_index, ratio_index] = score(recommendation)
            fold_active_groups[fold_index, ratio_index] = int(
                np.sum(np.linalg.norm(coefficient_path[ratio_index], axis=0) > 1e-10)
            )
            fold_recommendation_counts[fold_index, ratio_index] = np.bincount(
                recommendation, minlength=arm_count
            )
    mean_values = np.mean(fold_values, axis=0)
    fold_paired_differences = fold_values - fold_reference_values[:, None]
    mean_paired_differences = np.mean(fold_paired_differences, axis=0)
    if len(folds) > 1:
        standard_errors = np.std(fold_values, axis=0, ddof=1) / np.sqrt(len(folds))
        paired_standard_errors = np.std(
            fold_paired_differences, axis=0, ddof=1
        ) / np.sqrt(len(folds))
    else:
        standard_errors = np.zeros(len(ratios_array), dtype=float)
        paired_standard_errors = np.zeros(len(ratios_array), dtype=float)
    one_se_best_index = None
    one_se_threshold = None
    one_se_eligible_indices = None
    if selection_rule == "maximum_mean":
        selection_values = mean_values
        selection_standard_errors = standard_errors
    elif selection_rule in ("one_standard_error", "one_standard_error_paired_static"):
        paired = selection_rule.endswith("paired_static")
        selection_mean = mean_paired_differences if paired else mean_values
        selection_standard_errors = (
            paired_standard_errors if paired else standard_errors
        )
        best_mean = float(np.max(selection_mean))
        best_tied = np.flatnonzero(
            np.isclose(selection_mean, best_mean, rtol=0.0, atol=1e-12)
        )
        one_se_best_index = int(best_tied[np.argmax(ratios_array[best_tied])])
        one_se_threshold = float(
            selection_mean[one_se_best_index]
            - selection_standard_errors[one_se_best_index]
        )
        one_se_eligible_indices = np.flatnonzero(
            selection_mean >= one_se_threshold - 1e-12
        )
        if len(one_se_eligible_indices) == 0:
            raise RuntimeError("one-standard-error rule has no eligible ratio")
        selection_values = selection_mean
    else:
        raise ValueError(f"unknown group-policy selection rule: {selection_rule}")
    if selection_rule.startswith("one_standard_error"):
        selected_index = int(
            one_se_eligible_indices[np.argmax(ratios_array[one_se_eligible_indices])]
        )
    else:
        maximum = float(np.max(selection_values))
        tied = np.flatnonzero(
            np.isclose(selection_values, maximum, rtol=0.0, atol=1e-12)
        )
        selected_index = int(tied[np.argmax(ratios_array[tied])])
    selected_ratio = float(ratios_array[selected_index])
    (x_train, x_heldout, response, weight, outer_score) = _group_payload(
        final_fold, method, feature_spec, simplex
    )
    lambda_max = _weighted_group_lambda_max(x_train, response, weight)
    selected_lambda = float(selected_ratio * lambda_max)
    (coefficient_path, intercept_path) = _group_path(
        x_train, response, np.asarray([selected_lambda]), weight
    )
    coefficient = coefficient_path[0]
    intercept = intercept_path[0]
    recommendation = _recommend_group(x_heldout, coefficient, intercept, simplex)
    return FittedPolicy(
        name=name,
        recommendation=recommendation,
        coefficient=coefficient,
        intercept=intercept,
    )


def _survival_hajek_value(
    outcome: np.ndarray,
    treatment: np.ndarray,
    recommendation: np.ndarray,
    propensity: np.ndarray,
) -> float:
    inverse = inverse_probability(propensity)
    matched = (np.asarray(treatment, dtype=int) == recommendation) * inverse
    denominator = float(np.sum(matched))
    if denominator <= 0:
        return float("-inf")
    return float(np.sum(np.asarray(outcome, dtype=float) * matched) / denominator)


def _survival_angle_recommendation(
    features: np.ndarray,
    coefficient_with_intercept: np.ndarray,
    simplex: np.ndarray,
    *,
    reverse: bool,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(features)), features])
    scores = design @ coefficient_with_intercept @ simplex.T
    if reverse:
        scores = -scores
    return np.argmax(scores, axis=1).astype(np.int64)


def _time_block_indices(time: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(time, dtype=float).reshape(-1)
    order = np.argsort(-values, kind="stable")
    ordered = values[order]
    block_start = np.r_[True, ordered[1:] != ordered[:-1]]
    block_index = np.cumsum(block_start) - 1
    block_end = np.r_[ordered[1:] != ordered[:-1], True]
    ends = np.flatnonzero(block_end)
    row_end = ends[block_index]
    return (
        order.astype(np.int64),
        block_index.astype(np.int64),
        row_end.astype(np.int64),
    )


def _fit_qi_survival_coefficients(
    features: np.ndarray,
    treatment: np.ndarray,
    observed_time: np.ndarray,
    death_event: np.ndarray,
    inverse_propensity: np.ndarray,
    simplex: np.ndarray,
    *,
    candidates: list[float],
    epochs: int,
    learning_rate: float,
    device: torch.device,
):
    if not candidates:
        raise ValueError("Qi survival candidate grid must not be empty")
    design = torch.as_tensor(
        np.column_stack([np.ones(len(features)), features]),
        dtype=torch.float64,
        device=device,
    )
    vertex = torch.as_tensor(
        simplex[np.asarray(treatment, dtype=int)], dtype=torch.float64, device=device
    )
    event = torch.as_tensor(death_event, dtype=torch.float64, device=device)
    inverse = torch.as_tensor(inverse_propensity, dtype=torch.float64, device=device)
    (order, _, row_end) = _time_block_indices(observed_time)
    order_tensor = torch.as_tensor(order, dtype=torch.long, device=device)
    row_end_tensor = torch.as_tensor(row_end, dtype=torch.long, device=device)
    event_ordered = event[order_tensor]
    inverse_ordered = inverse[order_tensor]
    lambdas = torch.as_tensor(candidates, dtype=torch.float64, device=device)
    coefficient = torch.zeros(
        (len(candidates), design.shape[1], simplex.shape[1]),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    for iteration in range(1, int(epochs) + 1):
        risk = torch.einsum("np,cpd,nd->cn", design, coefficient, vertex)
        risk_ordered = risk[:, order_tensor]
        cumulative_log_risk = torch.logcumsumexp(risk_ordered, dim=1)
        log_risk_set = cumulative_log_risk[:, row_end_tensor]
        data_loss = -torch.mean(
            inverse_ordered[None, :]
            * event_ordered[None, :]
            * (risk_ordered - log_risk_set),
            dim=1,
        )
        if not torch.isfinite(data_loss).all():
            raise FloatingPointError("Qi survival Cox loss is nonfinite")
        torch.sum(data_loss).backward()
        with torch.no_grad():
            coefficient -= float(learning_rate) * coefficient.grad
            row_norm = torch.linalg.vector_norm(coefficient[:, 1:, :], dim=2)
            threshold = float(learning_rate) * lambdas[:, None]
            scale = torch.clamp(
                1.0 - threshold / torch.clamp(row_norm, min=1e-15), min=0.0
            )
            coefficient[:, 1:, :] *= scale[:, :, None]
            coefficient.grad.zero_()
    output = coefficient.detach().cpu().numpy()
    return output


def _fit_xue_survival_coefficients(
    features: np.ndarray,
    treatment: np.ndarray,
    observed_time: np.ndarray,
    death_event: np.ndarray,
    inverse_propensity: np.ndarray,
    simplex: np.ndarray,
    *,
    candidates: list[tuple[float, float]],
    epochs: int,
    learning_rate: float,
    device: torch.device,
):
    if not candidates:
        raise ValueError("Xue survival candidate grid must not be empty")
    design = torch.as_tensor(
        np.column_stack([np.ones(len(features)), features]),
        dtype=torch.float64,
        device=device,
    )
    vertex = torch.as_tensor(
        simplex[np.asarray(treatment, dtype=int)], dtype=torch.float64, device=device
    )
    event = torch.as_tensor(death_event, dtype=torch.float64, device=device)
    inverse = torch.as_tensor(inverse_propensity, dtype=torch.float64, device=device)
    (order, block_index, row_end) = _time_block_indices(observed_time)
    ordered_time = np.asarray(observed_time, dtype=float)[order]
    order_tensor = torch.as_tensor(order, dtype=torch.long, device=device)
    block_tensor = torch.as_tensor(block_index, dtype=torch.long, device=device)
    ends = np.unique(row_end)
    end_tensor = torch.as_tensor(ends, dtype=torch.long, device=device)
    event_ordered = event[order_tensor]
    block_count = int(block_index.max() + 1)
    event_count_by_block = np.bincount(
        block_index,
        weights=np.asarray(death_event, dtype=float)[order],
        minlength=block_count,
    )
    block_time = ordered_time[ends]
    failure_block = torch.as_tensor(
        (event_count_by_block > 0) & (block_time <= SURVIVAL_HORIZON_DAYS),
        dtype=torch.bool,
        device=device,
    )
    lambdas = torch.as_tensor(
        [candidate[0] for candidate in candidates], dtype=torch.float64, device=device
    )
    b_values = torch.as_tensor(
        [candidate[1] for candidate in candidates], dtype=torch.float64, device=device
    )
    coefficient = torch.zeros(
        (len(candidates), design.shape[1], simplex.shape[1]),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([coefficient], lr=float(learning_rate))
    for iteration in range(1, int(epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        margin = torch.einsum("np,cpd,nd->cn", design, coefficient, vertex)
        policy_weight = (
            torch.sigmoid(b_values[:, None] * (margin + 1.0)) * inverse[None, :]
        )
        ordered_weight = policy_weight[:, order_tensor]
        denominator = torch.cumsum(ordered_weight, dim=1)[:, end_tensor]
        numerator = torch.zeros(
            (len(candidates), block_count), dtype=torch.float64, device=device
        ).scatter_add(
            1,
            block_tensor[None, :].expand(len(candidates), -1),
            ordered_weight * event_ordered[None, :],
        )
        hazard = numerator / torch.clamp(denominator, min=1e-12)
        hazard = torch.clamp(hazard, min=0.0, max=1.0 - 1e-08)
        data_loss = -torch.sum(torch.log1p(-hazard[:, failure_block]), dim=1)
        penalty = lambdas * torch.sum(coefficient[:, 1:, :] ** 2, dim=(1, 2))
        candidate_loss = data_loss + penalty
        if not torch.isfinite(candidate_loss).all():
            raise FloatingPointError("Xue survival weighted-KM loss is nonfinite")
        torch.sum(candidate_loss).backward()
        optimizer.step()
    output = coefficient.detach().cpu().numpy()
    return output


def fit_qi_survival_policy(
    folds: list[PolicyFold],
    final_fold: PolicyFold,
    simplex: np.ndarray,
    *,
    device: torch.device,
) -> FittedPolicy:
    candidates = list(SURVIVAL_LAMBDA_GRID)
    fold_values = np.empty((len(folds), len(candidates)), dtype=float)
    for fold_index, fold in enumerate(folds):
        coefficient_path = _fit_qi_survival_coefficients(
            fold.features_train["x"],
            fold.treatment_train,
            fold.observed_time_train,
            fold.death_event_train,
            inverse_probability(fold.propensity_xwz_train),
            simplex,
            candidates=candidates,
            epochs=SURVIVAL_TUNING_EPOCHS,
            learning_rate=SURVIVAL_TUNING_LEARNING_RATE,
            device=device,
        )
        for candidate_index, coefficient in enumerate(coefficient_path):
            recommendation = _survival_angle_recommendation(
                fold.features_validation["x"], coefficient, simplex, reverse=True
            )
            fold_values[fold_index, candidate_index] = _survival_hajek_value(
                fold.outcome_validation,
                fold.treatment_validation,
                recommendation,
                fold.propensity_xwz_validation,
            )
    mean_values = np.mean(fold_values, axis=0)
    maximum = float(np.max(mean_values))
    tied = np.flatnonzero(np.isclose(mean_values, maximum, rtol=0.0, atol=1e-12))
    selected_index = max(tied.tolist(), key=lambda index: candidates[index])
    coefficient_path = _fit_qi_survival_coefficients(
        final_fold.features_train["x"],
        final_fold.treatment_train,
        final_fold.observed_time_train,
        final_fold.death_event_train,
        inverse_probability(final_fold.propensity_xwz_train),
        simplex,
        candidates=candidates,
        epochs=SURVIVAL_FINAL_EPOCHS,
        learning_rate=SURVIVAL_FINAL_LEARNING_RATE,
        device=device,
    )
    coefficient = coefficient_path[selected_index]
    return FittedPolicy(
        name="Qi AD(X)",
        recommendation=_survival_angle_recommendation(
            final_fold.features_validation["x"], coefficient, simplex, reverse=True
        ),
        coefficient=coefficient[1:].T,
        intercept=coefficient[0],
    )


def fit_xue_survival_policy(
    folds: list[PolicyFold],
    final_fold: PolicyFold,
    simplex: np.ndarray,
    *,
    device: torch.device,
) -> FittedPolicy:
    candidates = [
        (float(lambda_value), float(b_value))
        for lambda_value in SURVIVAL_LAMBDA_GRID
        for b_value in XUE_B_GRID
    ]
    fold_values = np.empty((len(folds), len(candidates)), dtype=float)
    for fold_index, fold in enumerate(folds):
        coefficient_path = _fit_xue_survival_coefficients(
            fold.features_train["x"],
            fold.treatment_train,
            fold.observed_time_train,
            fold.death_event_train,
            inverse_probability(fold.propensity_xwz_train),
            simplex,
            candidates=candidates,
            epochs=SURVIVAL_TUNING_EPOCHS,
            learning_rate=SURVIVAL_TUNING_LEARNING_RATE,
            device=device,
        )
        for candidate_index, coefficient in enumerate(coefficient_path):
            recommendation = _survival_angle_recommendation(
                fold.features_validation["x"], coefficient, simplex, reverse=False
            )
            fold_values[fold_index, candidate_index] = _survival_hajek_value(
                fold.outcome_validation,
                fold.treatment_validation,
                recommendation,
                fold.propensity_xwz_validation,
            )
    mean_values = np.mean(fold_values, axis=0)
    maximum = float(np.max(mean_values))
    tied = np.flatnonzero(np.isclose(mean_values, maximum, rtol=0.0, atol=1e-12))
    selected_index = sorted(
        tied.tolist(), key=lambda index: (-candidates[index][0], candidates[index][1])
    )[0]
    coefficient_path = _fit_xue_survival_coefficients(
        final_fold.features_train["x"],
        final_fold.treatment_train,
        final_fold.observed_time_train,
        final_fold.death_event_train,
        inverse_probability(final_fold.propensity_xwz_train),
        simplex,
        candidates=candidates,
        epochs=SURVIVAL_FINAL_EPOCHS,
        learning_rate=SURVIVAL_FINAL_LEARNING_RATE,
        device=device,
    )
    coefficient = coefficient_path[selected_index]
    (selected_lambda, selected_b) = candidates[selected_index]
    return FittedPolicy(
        name="Xue survival Angle(X)",
        recommendation=_survival_angle_recommendation(
            final_fold.features_validation["x"], coefficient, simplex, reverse=False
        ),
        coefficient=coefficient[1:].T,
        intercept=coefficient[0],
    )


def _km_survival_before(
    fit_time: np.ndarray, fit_censor_event: np.ndarray, query_time: np.ndarray
) -> np.ndarray:
    time = np.asarray(fit_time, dtype=float).reshape(-1)
    event = np.asarray(fit_censor_event, dtype=int).reshape(-1)
    query = np.asarray(query_time, dtype=float).reshape(-1)
    event_times = np.unique(time[event == 1])
    survival = 1.0
    values: list[float] = []
    for event_time in event_times:
        risk = int(np.sum(time >= event_time))
        failures = int(np.sum((time == event_time) & (event == 1)))
        if risk <= 0:
            raise RuntimeError("invalid Goldberg marginal-KM risk set")
        survival *= 1.0 - failures / risk
        values.append(survival)
    if not values:
        return np.ones(len(query), dtype=float)
    position = np.searchsorted(event_times, query, side="right") - 1
    output = np.ones(len(query), dtype=float)
    selected = position >= 0
    output[selected] = np.asarray(values, dtype=float)[position[selected]]
    return output


def _goldberg_marginal_km_weights(
    observed_time: np.ndarray,
    death_event: np.ndarray,
    censor_event: np.ndarray,
    treatment: np.ndarray,
    *,
    seed: int,
    n_splits: int,
):
    from sklearn.model_selection import StratifiedKFold

    time = np.asarray(observed_time, dtype=float)
    death = np.asarray(death_event, dtype=int)
    censor = np.asarray(censor_event, dtype=int)
    treatment = np.asarray(treatment, dtype=int)
    strata = treatment * 2 + censor
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    target = np.minimum(time, SURVIVAL_HORIZON_DAYS)
    query = np.maximum(target - 1e-08, 0.0)
    g = np.full(len(time), np.nan, dtype=float)
    for fold, (fit, predict) in enumerate(splitter.split(np.zeros(len(time)), strata)):
        g[predict] = _km_survival_before(time[fit], censor[fit], query[predict])
    if not np.isfinite(g).all():
        raise RuntimeError("Goldberg marginal-KM G is incomplete")
    g = np.clip(g, 0.001, 1.0)
    valid = (death == 1) | (time >= SURVIVAL_HORIZON_DAYS)
    weight = np.zeros(len(time), dtype=float)
    weight[valid] = 1.0 / g[valid]
    return (weight, valid, target)


def _goldberg_design(
    features: np.ndarray, treatment: np.ndarray, encoder: OneHotEncoder
) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    dummy = encoder.transform(np.asarray(treatment, dtype=int).reshape(-1, 1))
    interaction = (features[:, :, None] * dummy[:, None, :]).reshape(len(features), -1)
    return np.column_stack([features, dummy, interaction])


def _goldberg_alpha_max(
    design: np.ndarray, target: np.ndarray, sample_weight: np.ndarray
) -> float:
    matrix = np.asarray(design, dtype=float)
    response = np.asarray(target, dtype=float).reshape(-1)
    weight = np.asarray(sample_weight, dtype=float).reshape(-1)
    if np.any(weight <= 0) or float(np.sum(weight)) <= 0.0:
        raise ValueError("Goldberg requires positive sample weights")
    normalized = weight / np.sum(weight)
    centered_matrix = matrix - np.sum(normalized[:, None] * matrix, axis=0)
    centered_response = response - float(np.sum(normalized * response))
    value = float(np.max(np.abs(centered_matrix.T @ (normalized * centered_response))))
    return max(value, np.finfo(float).eps)


def _fit_goldberg_path(
    features: np.ndarray,
    treatment: np.ndarray,
    observed_time: np.ndarray,
    death_event: np.ndarray,
    censor_event: np.ndarray,
    encoder: OneHotEncoder,
    ratios: tuple[float, ...],
    *,
    seed: int,
):
    (km_weight, valid, target) = _goldberg_marginal_km_weights(
        observed_time, death_event, censor_event, treatment, seed=seed, n_splits=5
    )
    design = _goldberg_design(features[valid], treatment[valid], encoder)
    valid_target = target[valid]
    valid_weight = km_weight[valid]
    alpha_max = _goldberg_alpha_max(design, valid_target, valid_weight)
    models: list[Lasso] = []
    for ratio in ratios:
        alpha = float(ratio * alpha_max)
        model = Lasso(
            alpha=alpha,
            fit_intercept=True,
            max_iter=L1_MAX_ITER,
            tol=L1_TOL,
            selection="cyclic",
        )
        model.fit(design, valid_target, sample_weight=valid_weight)
        models.append(model)
    return models


def _goldberg_recommendations(
    models: list[Lasso], features: np.ndarray, encoder: OneHotEncoder, arm_count: int
) -> np.ndarray:
    result = np.empty((len(models), len(features)), dtype=np.int64)
    for index, model in enumerate(models):
        prediction = np.column_stack(
            [
                model.predict(
                    _goldberg_design(
                        features, np.full(len(features), arm, dtype=int), encoder
                    )
                )
                for arm in range(arm_count)
            ]
        )
        result[index] = np.argmax(prediction, axis=1)
    return result


def fit_goldberg_survival_policy(
    folds: list[PolicyFold], final_fold: PolicyFold, arm_count: int
) -> FittedPolicy:
    ratios = L1_RATIO_GRID
    encoder = OneHotEncoder(
        drop="first", sparse_output=False, categories=[range(arm_count)]
    ).fit(final_fold.treatment_train.reshape(-1, 1))
    fold_values = np.empty((len(folds), len(ratios)), dtype=float)
    for fold_index, fold in enumerate(folds):
        models = _fit_goldberg_path(
            fold.features_train["x"],
            fold.treatment_train,
            fold.observed_time_train,
            fold.death_event_train,
            fold.coverage_censor_event_train,
            encoder,
            ratios,
            seed=20260817 + fold_index,
        )
        recommendation_path = _goldberg_recommendations(
            models, fold.features_validation["x"], encoder, arm_count
        )
        for ratio_index, recommendation in enumerate(recommendation_path):
            fold_values[fold_index, ratio_index] = _survival_hajek_value(
                fold.outcome_validation,
                fold.treatment_validation,
                recommendation,
                fold.propensity_x_validation,
            )
    mean_values = np.mean(fold_values, axis=0)
    maximum = float(np.max(mean_values))
    tied = np.flatnonzero(np.isclose(mean_values, maximum, rtol=0.0, atol=1e-12))
    selected_index = min(tied.tolist())
    models = _fit_goldberg_path(
        final_fold.features_train["x"],
        final_fold.treatment_train,
        final_fold.observed_time_train,
        final_fold.death_event_train,
        final_fold.coverage_censor_event_train,
        encoder,
        ratios,
        seed=20261817 + int(final_fold.policy_fold),
    )
    model = models[selected_index]
    p = final_fold.features_train["x"].shape[1]
    contrast_intercept = model.coef_[p : p + arm_count - 1]
    contrast_coefficient = model.coef_[p + arm_count - 1 :].reshape(p, arm_count - 1).T
    return FittedPolicy(
        name="Goldberg censored Q(X)",
        recommendation=_goldberg_recommendations(
            [model], final_fold.features_validation["x"], encoder, arm_count
        )[0],
        coefficient=contrast_coefficient,
        intercept=contrast_intercept,
    )
