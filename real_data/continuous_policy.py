from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import MultiTaskLasso, lasso_path
from sklearn.preprocessing import OneHotEncoder

GROUP_RATIO_GRID = tuple((float(2.0 ** (-index)) for index in range(11)))
L1_RATIO_GRID = tuple((float(10.0 ** (-index / 4.0)) for index in range(13)))
XUE_LAMBDA_GRID = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)
XUE_B_GRID = tuple((float(index) / 2.0 for index in range(1, 21)))
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
            ("h_train", self.h_train, train_n),
            ("q_train", self.q_train, train_n),
            ("propensity_x_train", self.propensity_x_train, train_n),
            ("propensity_xwz_train", self.propensity_xwz_train, train_n),
            ("outcome_validation", self.outcome_validation, validation_n),
            ("treatment_validation", self.treatment_validation, validation_n),
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


def _fit_xue_coefficients(
    features: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    inverse_propensity: np.ndarray,
    simplex: np.ndarray,
    *,
    candidates: list[tuple[float, float]],
    epochs: int,
    learning_rate: float,
    device: torch.device,
):
    if not candidates:
        raise ValueError("Xue candidate grid must not be empty")
    x = torch.as_tensor(
        np.column_stack([np.ones(len(features)), features]),
        dtype=torch.float64,
        device=device,
    )
    y = torch.as_tensor(outcome, dtype=torch.float64, device=device)
    inverse = torch.as_tensor(inverse_propensity, dtype=torch.float64, device=device)
    vertex = torch.as_tensor(
        simplex[np.asarray(treatment, dtype=int)], dtype=torch.float64, device=device
    )
    lambdas = torch.as_tensor(
        [candidate[0] for candidate in candidates], dtype=torch.float64, device=device
    )
    b_values = torch.as_tensor(
        [candidate[1] for candidate in candidates], dtype=torch.float64, device=device
    )
    coefficient = torch.zeros(
        (len(candidates), x.shape[1], simplex.shape[1]),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([coefficient], lr=float(learning_rate))
    for iteration in range(1, int(epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        margin = torch.einsum("np,cpm,nm->cn", x, coefficient, vertex)
        data_loss = -torch.mean(
            (y * inverse)[None, :] * torch.sigmoid(b_values[:, None] * margin), dim=1
        )
        penalty = lambdas * torch.sum(coefficient[:, 1:, :] ** 2, dim=(1, 2))
        candidate_loss = data_loss + penalty
        if not torch.isfinite(candidate_loss).all():
            raise FloatingPointError("Xue policy loss is nonfinite")
        loss = torch.sum(candidate_loss)
        loss.backward()
        optimizer.step()
    return coefficient.detach().cpu().numpy()


def _recommend_matrix(
    features: np.ndarray, coefficient_with_intercept: np.ndarray, simplex: np.ndarray
) -> np.ndarray:
    design = np.column_stack([np.ones(len(features)), features])
    return np.argmax(design @ coefficient_with_intercept @ simplex.T, axis=1).astype(
        int
    )


def fit_xue_policy(
    folds: list[PolicyFold],
    final_fold: PolicyFold,
    simplex: np.ndarray,
    *,
    device: torch.device,
    tuning_epochs: int = 1000,
    final_epochs: int = 1000,
    learning_rate: float = 0.01,
) -> FittedPolicy:
    candidates = [
        (float(lambda_value), float(b_value))
        for lambda_value in XUE_LAMBDA_GRID
        for b_value in XUE_B_GRID
    ]
    fold_values = np.empty((len(folds), len(candidates)), dtype=float)
    for fold_index, fold in enumerate(folds):
        inverse_train = inverse_probability(fold.propensity_xwz_train)
        inverse_validation = inverse_probability(fold.propensity_xwz_validation)
        coefficient_path = _fit_xue_coefficients(
            fold.features_train["x"],
            fold.outcome_train,
            fold.treatment_train,
            inverse_train,
            simplex,
            candidates=candidates,
            epochs=tuning_epochs,
            learning_rate=learning_rate,
            device=device,
        )
        for candidate_index, (lambda_value, b_value) in enumerate(candidates):
            recommendation = _recommend_matrix(
                fold.features_validation["x"],
                coefficient_path[candidate_index],
                simplex,
            )
            matched = (recommendation == fold.treatment_validation) * inverse_validation
            denominator = float(np.sum(matched))
            fold_values[fold_index, candidate_index] = (
                float("-inf")
                if denominator <= 0
                else float(np.sum(fold.outcome_validation * matched) / denominator)
            )
    mean_values = np.mean(fold_values, axis=0)
    maximum = float(np.max(mean_values))
    tied = np.flatnonzero(np.isclose(mean_values, maximum, rtol=0.0, atol=1e-12))
    selected_index = sorted(
        tied.tolist(), key=lambda index: (-candidates[index][0], candidates[index][1])
    )[0]
    (selected_lambda, selected_b) = candidates[selected_index]
    coefficient_path = _fit_xue_coefficients(
        final_fold.features_train["x"],
        final_fold.outcome_train,
        final_fold.treatment_train,
        inverse_probability(final_fold.propensity_xwz_train),
        simplex,
        candidates=[(selected_lambda, selected_b)],
        epochs=final_epochs,
        learning_rate=learning_rate,
        device=device,
    )
    coefficient = coefficient_path[0]
    recommendation = _recommend_matrix(
        final_fold.features_validation["x"], coefficient, simplex
    )
    return FittedPolicy(
        name="Xue-style Angle(X)",
        recommendation=recommendation,
        coefficient=coefficient[1:].T,
        intercept=coefficient[0],
    )


def _l1_design(
    features: np.ndarray, treatment: np.ndarray, encoder: OneHotEncoder
) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    treatment = np.asarray(treatment, dtype=int).reshape(-1)
    dummies = encoder.transform(treatment.reshape(-1, 1))
    interactions = (features[:, :, None] * dummies[:, None, :]).reshape(
        len(features), -1
    )
    return np.column_stack([features, dummies, interactions])


def _l1_alpha_max(design: np.ndarray, outcome: np.ndarray) -> float:
    design = np.asarray(design, dtype=float)
    outcome = np.asarray(outcome, dtype=float).reshape(-1)
    scale = np.sqrt(np.mean(np.square(design), axis=0))
    active = scale > np.finfo(float).eps
    if not np.any(active):
        return float(np.finfo(float).eps)
    scaled = design[:, active] / scale[active]
    centered_x = scaled - np.mean(scaled, axis=0)
    centered_y = outcome - np.mean(outcome)
    value = float(np.max(np.abs(centered_x.T @ centered_y)) / len(outcome))
    return max(value, np.finfo(float).eps)


def _fit_l1_path(
    design: np.ndarray, outcome: np.ndarray, alphas: np.ndarray
) -> list[dict[str, object]]:
    design = np.asarray(design, dtype=float)
    outcome = np.asarray(outcome, dtype=float).reshape(-1)
    alphas = np.asarray(alphas, dtype=float).reshape(-1)
    scale = np.sqrt(np.mean(np.square(design), axis=0))
    active = scale > np.finfo(float).eps
    models: list[dict[str, object]] = []
    if not np.any(active):
        return [
            {
                "intercept": float(np.mean(outcome)),
                "coefficient": np.zeros(design.shape[1]),
                "n_iter": 0,
                "dual_gap": 0.0,
            }
            for _ in alphas
        ]
    scaled = design[:, active] / scale[active]
    x_mean = np.mean(scaled, axis=0)
    y_mean = float(np.mean(outcome))
    descending = np.argsort(-alphas, kind="stable")
    descending_alphas = alphas[descending]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        (returned_alphas, path, gaps, iterations) = lasso_path(
            scaled - x_mean,
            outcome - y_mean,
            alphas=descending_alphas,
            max_iter=L1_MAX_ITER,
            tol=L1_TOL,
            return_n_iter=True,
        )
    if any((issubclass(w.category, ConvergenceWarning) for w in caught)):
        raise RuntimeError("l1-PLS path produced a convergence warning")
    if not np.array_equal(returned_alphas, descending_alphas):
        raise RuntimeError("unexpected l1-PLS alpha path")
    models = [None] * len(alphas)
    for path_index, original_index in enumerate(descending):
        coefficient = np.zeros(design.shape[1], dtype=float)
        coefficient[active] = path[:, path_index] / scale[active]
        models[int(original_index)] = {
            "intercept": float(y_mean - x_mean @ path[:, path_index]),
            "coefficient": coefficient,
            "n_iter": int(iterations[path_index]),
            "dual_gap": float(gaps[path_index]),
        }
    return models


def _predict_l1(model: dict[str, object], design: np.ndarray) -> np.ndarray:
    return float(model["intercept"]) + np.asarray(design) @ np.asarray(
        model["coefficient"], dtype=float
    )


def _l1_active_rule_variables(
    model: dict[str, object], feature_count: int, arm_count: int
) -> int:
    start = feature_count + arm_count - 1
    interaction = np.asarray(model["coefficient"])[start:].reshape(
        feature_count, arm_count - 1
    )
    return int(np.sum(np.any(np.abs(interaction) > 1e-12, axis=1)))


def fit_l1_pls_policy(
    folds: list[PolicyFold],
    final_fold: PolicyFold,
    arm_count: int,
    ratios: tuple[float, ...] = L1_RATIO_GRID,
) -> FittedPolicy:
    ratios_array = np.asarray(ratios, dtype=float)
    encoder = OneHotEncoder(
        drop="first", sparse_output=False, categories=[range(arm_count)]
    ).fit(final_fold.treatment_train.reshape(-1, 1))
    values = np.empty((len(folds), len(ratios_array)), dtype=float)
    errors = np.empty_like(values)
    fold_active_rule_variables = np.empty_like(values, dtype=int)
    fold_recommendation_counts = np.empty(
        (len(folds), len(ratios_array), arm_count), dtype=int
    )
    convergence: dict[str, list[dict[str, object]]] = {
        str(float(ratio)): [] for ratio in ratios_array
    }
    for fold_index, fold in enumerate(folds):
        train_design = _l1_design(
            fold.features_train["x"], fold.treatment_train, encoder
        )
        alpha_max = _l1_alpha_max(train_design, fold.outcome_train)
        models = _fit_l1_path(
            train_design, fold.outcome_train, ratios_array * alpha_max
        )
        observed_design = _l1_design(
            fold.features_validation["x"], fold.treatment_validation, encoder
        )
        all_arm_design = [
            _l1_design(
                fold.features_validation["x"],
                np.full(len(fold.outcome_validation), arm, dtype=int),
                encoder,
            )
            for arm in range(arm_count)
        ]
        validation_weight = inverse_probability(fold.propensity_x_validation)
        for ratio_index, (ratio, model) in enumerate(zip(ratios_array, models)):
            observed_prediction = _predict_l1(model, observed_design)
            errors[fold_index, ratio_index] = float(
                np.mean(np.square(fold.outcome_validation - observed_prediction))
            )
            potential = np.column_stack(
                [_predict_l1(model, design) for design in all_arm_design]
            )
            recommendation = np.argmax(potential, axis=1)
            fold_active_rule_variables[fold_index, ratio_index] = (
                _l1_active_rule_variables(
                    model, fold.features_train["x"].shape[1], arm_count
                )
            )
            fold_recommendation_counts[fold_index, ratio_index] = np.bincount(
                recommendation, minlength=arm_count
            )
            matched = (recommendation == fold.treatment_validation) * validation_weight
            denominator = float(np.sum(matched))
            values[fold_index, ratio_index] = (
                float("-inf")
                if denominator <= 0
                else float(np.sum(fold.outcome_validation * matched) / denominator)
            )
            convergence[str(float(ratio))].append(
                {
                    "policy_fold": int(fold.policy_fold),
                    "alpha_max": float(alpha_max),
                    "absolute_alpha": float(ratio * alpha_max),
                    "n_iter": int(model["n_iter"]),
                    "dual_gap": float(model["dual_gap"]),
                }
            )
    mean_values = np.mean(values, axis=0)
    mean_errors = np.mean(errors, axis=0)
    maximum = float(np.max(mean_values))
    value_tied = np.flatnonzero(np.isclose(mean_values, maximum, rtol=0.0, atol=1e-12))
    full_design = _l1_design(
        final_fold.features_train["x"], final_fold.treatment_train, encoder
    )
    full_alpha_max = _l1_alpha_max(full_design, final_fold.outcome_train)
    tied_models = _fit_l1_path(
        full_design, final_fold.outcome_train, ratios_array[value_tied] * full_alpha_max
    )
    active_by_index = {
        int(index): _l1_active_rule_variables(
            model, final_fold.features_train["x"].shape[1], arm_count
        )
        for (index, model) in zip(value_tied, tied_models)
    }
    minimum_active = min(active_by_index.values())
    sparsity_tied = np.asarray(
        [
            index
            for index in value_tied
            if active_by_index[int(index)] == minimum_active
        ],
        dtype=int,
    )
    minimum_error = float(np.min(mean_errors[sparsity_tied]))
    error_tied = np.asarray(
        [
            index
            for index in sparsity_tied
            if np.isclose(mean_errors[index], minimum_error, rtol=0.0, atol=1e-12)
        ],
        dtype=int,
    )
    selected_index = int(error_tied[np.argmax(ratios_array[error_tied])])
    selected_ratio = float(ratios_array[selected_index])
    selected_alpha = float(selected_ratio * full_alpha_max)
    model = _fit_l1_path(
        full_design, final_fold.outcome_train, np.asarray([selected_alpha])
    )[0]
    potential = np.column_stack(
        [
            _predict_l1(
                model,
                _l1_design(
                    final_fold.features_validation["x"],
                    np.full(len(final_fold.outcome_validation), arm, dtype=int),
                    encoder,
                ),
            )
            for arm in range(arm_count)
        ]
    )
    recommendation = np.argmax(potential, axis=1).astype(int)
    p = final_fold.features_train["x"].shape[1]
    arm_terms = arm_count - 1
    coefficient_vector = np.asarray(model["coefficient"], dtype=float)
    contrast = np.zeros((arm_terms, p), dtype=float)
    contrast[:] = coefficient_vector[p + arm_terms :].reshape(p, arm_terms).T
    intercept = coefficient_vector[p : p + arm_terms]
    return FittedPolicy(
        name="l1-PLS(X)",
        recommendation=recommendation,
        coefficient=contrast,
        intercept=intercept,
    )
