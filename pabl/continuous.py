import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Lasso, lasso_path
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from pabl.dgp import ContinuousDGP
from pabl.common import (
    fit_proximal_itr_linear_fista,
    fit_weighted_proximal_itr_linear_fista,
    inverse_probability_weights as inverse_probability_weights,
)

Q_CROSSFIT_SPLITS = 3
Q_BRIDGE_CV = 3
H_CROSSFIT_SPLITS = 3
H_BRIDGE_CV = 3
PI_CROSSFIT_SPLITS = 3
LAMBDA_CV_SPLITS = 3
L1_PLS_RATIO_GRID = tuple((float(10.0 ** (-j / 4.0)) for j in range(13)))
L1_PLS_ALPHA_GRID = L1_PLS_RATIO_GRID
L1_PLS_TUNING_MAX_ITER = 50000
L1_PLS_FINAL_MAX_ITER = 50000


def build_full_arm_h_pseudo_response(h_cf_tr, V_matrix, dtype=None, device=None):
    if device is None:
        device = h_cf_tr.device if torch.is_tensor(h_cf_tr) else torch.device("cpu")
    if dtype is None:
        dtype = (
            h_cf_tr.dtype
            if torch.is_tensor(h_cf_tr) and h_cf_tr.is_floating_point()
            else torch.float64
        )
    h_all = torch.as_tensor(h_cf_tr, dtype=dtype, device=device)
    V_tensor_local = torch.as_tensor(V_matrix, dtype=dtype, device=device)
    if h_all.ndim != 2:
        raise ValueError(
            "h_cf_tr must be a 2D matrix with one column per treatment arm"
        )
    if h_all.shape[1] != V_tensor_local.shape[0]:
        raise ValueError("h_cf_tr columns must match the number of simplex vertices")
    if not torch.isfinite(h_all).all():
        raise ValueError("h_cf_tr contains non-finite values")
    return h_all @ V_tensor_local


def build_full_arm_dr_pseudo_response(
    Y_tr, A_tr, q_tr, h_obs_tr, h_cf_tr, V_matrix, dtype=None, device=None
):
    if device is None:
        for obj in (Y_tr, A_tr, q_tr, h_obs_tr, h_cf_tr):
            if torch.is_tensor(obj):
                device = obj.device
                break
        if device is None:
            device = torch.device("cpu")
    if dtype is None:
        for obj in (Y_tr, q_tr, h_obs_tr, h_cf_tr):
            if torch.is_tensor(obj) and obj.is_floating_point():
                dtype = obj.dtype
                break
        if dtype is None:
            dtype = torch.float64
    Y_vec = torch.as_tensor(Y_tr, dtype=dtype, device=device).view(-1)
    A_vec = torch.as_tensor(A_tr, dtype=torch.long, device=device).view(-1)
    q_vec = torch.as_tensor(q_tr, dtype=dtype, device=device).view(-1)
    h_obs = torch.as_tensor(h_obs_tr, dtype=dtype, device=device).view(-1)
    h_all = torch.as_tensor(h_cf_tr, dtype=dtype, device=device)
    V_tensor_local = torch.as_tensor(V_matrix, dtype=dtype, device=device)
    K_arms = V_tensor_local.shape[0]
    if h_all.shape != (len(Y_vec), K_arms):
        raise ValueError("h_cf_tr must have shape (n, K_arms)")
    if not len(Y_vec) == len(A_vec) == len(q_vec) == len(h_obs):
        raise ValueError(
            "Y_tr, A_tr, q_tr, h_obs_tr, and h_cf_tr must have matching lengths"
        )
    if torch.any((A_vec < 0) | (A_vec >= K_arms)):
        raise ValueError("A_tr contains treatment labels outside 0, ..., K-1")
    if not torch.isfinite(h_all).all():
        raise ValueError("h_cf_tr contains non-finite values")
    pseudo_all = h_all.clone()
    rows = torch.arange(len(Y_vec), device=device)
    pseudo_all[rows, A_vec] += q_vec * (Y_vec - h_obs)
    return (pseudo_all @ V_tensor_local, pseudo_all)


def build_qi_ad_training_target(Y_tr, A_tr, pi_tr, V_matrix):
    device = Y_tr.device if torch.is_tensor(Y_tr) else torch.device("cpu")
    dtype = Y_tr.dtype if torch.is_tensor(Y_tr) else torch.float64
    Y_vec = torch.as_tensor(Y_tr, dtype=dtype, device=device).view(-1)
    A_vec = torch.as_tensor(A_tr, dtype=torch.long, device=device).view(-1)
    pi_vec = torch.as_tensor(pi_tr, dtype=dtype, device=device).view(-1)
    V_tensor_local = torch.as_tensor(V_matrix, dtype=dtype, device=device)
    K_arms = V_tensor_local.shape[0]
    if not len(Y_vec) == len(A_vec) == len(pi_vec):
        raise ValueError("Qi AD outcome, treatment, and propensity lengths must match")
    if torch.any((A_vec < 0) | (A_vec >= K_arms)):
        raise ValueError("Qi AD treatment labels are outside the simplex arm range")
    if not torch.isfinite(Y_vec).all() or not torch.isfinite(pi_vec).all():
        raise ValueError("Qi AD inputs must be finite")
    response = K_arms * Y_vec.view(-1, 1) * V_tensor_local[A_vec]
    sample_weight = inverse_probability_weights(pi_vec).to(dtype=dtype, device=device)
    return (response, sample_weight)


def generate_data_wrapper(
    n,
    seed,
    linearity="linear",
    U_inter=True,
    device="cpu",
    preprocessors=None,
    fit_preprocess=True,
):
    sim = ContinuousDGP(samp_size=n, add_noise=True, seed=seed, K=4)
    df = sim.gen_Y(linearity=linearity, U_inter=U_inter)
    W = df["W"].values.reshape(-1, 1)
    Z = df["Z"].values.reshape(-1, 1)
    U = df["U"].values.reshape(-1, 1)
    A = df["A"].values.reshape(-1, 1)
    Y = df["Y"].values.reshape(-1, 1)
    A_opt = df["GOR"].values
    X_cols = [
        col
        for col in df.columns
        if col.startswith("X") and (not col.startswith("X_irre"))
    ]
    X_irre_cols = [col for col in df.columns if col.startswith("X_irre")]
    X_raw = df[X_cols + X_irre_cols].values
    raw_names = [f"X_{i + 1}" for i in range(X_raw.shape[1])]
    if fit_preprocess:
        preprocessors = {}
        if linearity == "linear":
            X_processed = X_raw
            feature_names = raw_names
            preprocessors["spline"] = None
        else:
            spline = SplineTransformer(n_knots=4, degree=3, include_bias=False)
            X_processed = spline.fit_transform(X_raw)
            feature_names = list(spline.get_feature_names_out(input_features=raw_names))
            preprocessors["spline"] = spline
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_processed)
        preprocessors["scaler"] = scaler
        preprocessors["feature_names"] = feature_names
        preprocessors["linearity"] = linearity
    else:
        if preprocessors is None:
            raise ValueError("preprocessors must be provided when fit_preprocess=False")
        if preprocessors.get("linearity") != linearity:
            raise ValueError(
                "The training preprocessors were fitted under a different linearity setting"
            )
        spline = preprocessors["spline"]
        if spline is None:
            X_processed = X_raw
        else:
            X_processed = spline.transform(X_raw)
        X_scaled = preprocessors["scaler"].transform(X_processed)
        feature_names = preprocessors["feature_names"]
    if hasattr(sim, "Y_all"):
        true_outcomes = sim.Y_all
    elif hasattr(sim, "delta_all"):
        true_outcomes = sim.delta_all
    else:
        raise AttributeError(
            "Invalid preprocessing input: use fitted training transformations for prediction."
        )
    X_raw_tensor = torch.tensor(X_raw, dtype=torch.float64, device=device)
    (oracle_x, oracle_xw, oracle_full, delta_x, delta_xw, delta_full) = (
        sim.compute_oracle_regimes(return_delta=True)
    )
    (oracle_xz, delta_xz) = sim.compute_oracle_xz(return_delta=True)
    (oracle_xwz, delta_xwz) = sim.compute_oracle_xwz(return_delta=True)
    return (
        torch.tensor(W, dtype=torch.float64, device=device),
        torch.tensor(X_scaled, dtype=torch.float64, device=device),
        torch.tensor(Z, dtype=torch.float64, device=device),
        torch.tensor(U, dtype=torch.float64, device=device),
        torch.tensor(A, dtype=torch.float64, device=device),
        torch.tensor(Y, dtype=torch.float64, device=device),
        torch.tensor(A_opt, dtype=torch.long, device=device),
        torch.tensor(true_outcomes, dtype=torch.float64, device=device),
        feature_names,
        X_raw_tensor,
        preprocessors,
        torch.tensor(oracle_x, dtype=torch.long, device=device),
        torch.tensor(oracle_xz, dtype=torch.long, device=device),
        torch.tensor(oracle_xw, dtype=torch.long, device=device),
        torch.tensor(oracle_xwz, dtype=torch.long, device=device),
        torch.tensor(oracle_full, dtype=torch.long, device=device),
        torch.tensor(delta_x, dtype=torch.float64, device=device),
        torch.tensor(delta_xz, dtype=torch.float64, device=device),
        torch.tensor(delta_xw, dtype=torch.float64, device=device),
        torch.tensor(delta_xwz, dtype=torch.float64, device=device),
        torch.tensor(delta_full, dtype=torch.float64, device=device),
    )


def _to_numpy_1d(x, dtype=None):
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy().reshape(-1)
    else:
        arr = np.asarray(x).reshape(-1)
    return arr.astype(dtype) if dtype is not None else arr


def _to_numpy_2d(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def summarize_policy_performance(
    d_pred,
    potential_outcomes_te,
    oracle_own_te,
    oracle_full_te,
    margin_delta_own_te=None,
):
    d = _to_numpy_1d(d_pred, dtype=int)
    potential = _to_numpy_2d(potential_outcomes_te)
    return {"true_val": float(np.mean(potential[np.arange(len(d)), d]))}


def evaluate_model(
    name,
    Feat_tr,
    Feat_te,
    Y_tr,
    A_tr,
    weight_tr,
    V_mat,
    V_tensor,
    oracle_own_te,
    oracle_full_te,
    delta_all_te,
    margin_delta_own_te,
    l_f,
    feat_names,
    device,
):
    U_tr = torch.cat([torch.ones((len(Y_tr), 1), device=device), Feat_tr], dim=1)
    V_enc = torch.tensor(
        V_mat[A_tr.long().view(-1).cpu()], dtype=torch.float64, device=device
    )
    Y_ps = Y_tr.view(-1, 1) * weight_tr.view(-1, 1) * V_enc
    B = fit_proximal_itr_linear_fista(U_tr, Y_ps, l_f)
    U_te = torch.cat(
        [torch.ones((len(delta_all_te), 1), device=device), Feat_te], dim=1
    )
    d_pred = torch.argmax(U_te @ B @ V_tensor.T, dim=1)
    perf = summarize_policy_performance(
        d_pred, delta_all_te, oracle_own_te, oracle_full_te, margin_delta_own_te
    )
    return perf


def evaluate_full_arm_h_model(
    name,
    Feat_tr,
    Feat_te,
    h_cf_tr,
    V_mat,
    V_tensor,
    oracle_own_te,
    oracle_full_te,
    delta_all_te,
    margin_delta_own_te,
    l_f,
    feat_names,
    device,
):
    dtype = Feat_tr.dtype
    Y_ps = build_full_arm_h_pseudo_response(h_cf_tr, V_mat, dtype=dtype, device=device)
    U_tr = torch.cat(
        [torch.ones((len(Feat_tr), 1), dtype=dtype, device=device), Feat_tr], dim=1
    )
    B = fit_proximal_itr_linear_fista(U_tr, Y_ps, l_f)
    U_te = torch.cat(
        [torch.ones((len(delta_all_te), 1), dtype=dtype, device=device), Feat_te], dim=1
    )
    d_pred = torch.argmax(U_te @ B @ V_tensor.T, dim=1)
    perf = summarize_policy_performance(
        d_pred, delta_all_te, oracle_own_te, oracle_full_te, margin_delta_own_te
    )
    return perf


def evaluate_qi_ad_model(
    name,
    Feat_tr,
    Feat_te,
    Y_tr,
    A_tr,
    pi_tr,
    V_mat,
    V_tensor,
    oracle_own_te,
    oracle_full_te,
    delta_all_te,
    margin_delta_own_te,
    l_f,
    feat_names,
    device,
):
    dtype = Feat_tr.dtype
    (response, sample_weight) = build_qi_ad_training_target(Y_tr, A_tr, pi_tr, V_mat)
    U_tr = torch.cat(
        [torch.ones((len(Y_tr), 1), dtype=dtype, device=device), Feat_tr], dim=1
    )
    B = fit_weighted_proximal_itr_linear_fista(
        U_tr,
        response.to(dtype=dtype, device=device),
        sample_weight.to(dtype=dtype, device=device),
        l_f,
    )
    U_te = torch.cat(
        [torch.ones((len(delta_all_te), 1), dtype=dtype, device=device), Feat_te], dim=1
    )
    d_pred = torch.argmax(U_te @ B @ V_tensor.T, dim=1)
    perf = summarize_policy_performance(
        d_pred, delta_all_te, oracle_own_te, oracle_full_te, margin_delta_own_te
    )
    return perf


def evaluate_full_arm_dr_model(
    name,
    Feat_tr,
    Feat_te,
    Y_tr,
    A_tr,
    q_tr,
    h_obs_tr,
    h_cf_tr,
    V_mat,
    V_tensor,
    oracle_own_te,
    oracle_full_te,
    delta_all_te,
    margin_delta_own_te,
    l_f,
    feat_names,
    device,
):
    dtype = Feat_tr.dtype
    (Y_ps, _pseudo_all) = build_full_arm_dr_pseudo_response(
        Y_tr, A_tr, q_tr, h_obs_tr, h_cf_tr, V_mat, dtype=dtype, device=device
    )
    U_tr = torch.cat(
        [torch.ones((len(A_tr), 1), dtype=dtype, device=device), Feat_tr], dim=1
    )
    B = fit_proximal_itr_linear_fista(U_tr, Y_ps, l_f)
    U_te = torch.cat(
        [torch.ones((len(delta_all_te), 1), dtype=dtype, device=device), Feat_te], dim=1
    )
    d_pred = torch.argmax(U_te @ B @ V_tensor.T, dim=1)
    perf = summarize_policy_performance(
        d_pred, delta_all_te, oracle_own_te, oracle_full_te, margin_delta_own_te
    )
    return perf


def _build_l1_pls_design(X, treatment, encoder):
    X = np.asarray(X, dtype=float)
    treatment = np.asarray(treatment).reshape(-1).astype(int)
    dummies = encoder.transform(treatment.reshape(-1, 1))
    interactions = (X[:, :, None] * dummies[:, None, :]).reshape(X.shape[0], -1)
    return np.hstack([X, dummies, interactions])


def _fit_l1_pls_empirical_rms(Phi, Y, alpha, max_iter, random_state=42):
    Phi = np.asarray(Phi, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    if Phi.ndim != 2 or Phi.shape[0] != len(Y):
        raise ValueError("Invalid l1-PLS design/target shapes")
    penalty_scales = np.sqrt(np.mean(np.square(Phi), axis=0))
    nonconstant = penalty_scales > np.finfo(float).eps
    coef = np.zeros(Phi.shape[1], dtype=float)
    if np.any(nonconstant):
        scaled_design = Phi[:, nonconstant] / penalty_scales[nonconstant]
        lasso = Lasso(
            alpha=float(alpha),
            fit_intercept=True,
            random_state=int(random_state),
            max_iter=int(max_iter),
        )
        lasso.fit(scaled_design, Y)
        coef[nonconstant] = lasso.coef_ / penalty_scales[nonconstant]
        intercept = float(lasso.intercept_)
        n_iter = int(lasso.n_iter_)
        dual_gap = float(lasso.dual_gap_)
    else:
        intercept = float(np.mean(Y))
        n_iter = 0
        dual_gap = 0.0
    return {
        "intercept": intercept,
        "coef": coef,
        "penalty_scales": penalty_scales,
        "n_iter": n_iter,
        "dual_gap": dual_gap,
        "max_iter": int(max_iter),
        "converged_before_max_iter": bool(n_iter < int(max_iter)),
    }


def _fit_l1_pls_empirical_rms_path(Phi, Y, alphas, max_iter):
    Phi = np.asarray(Phi, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    alphas = np.asarray(alphas, dtype=float).reshape(-1)
    if Phi.ndim != 2 or Phi.shape[0] != len(Y) or len(alphas) == 0:
        raise ValueError("Invalid l1-PLS path inputs")
    if np.any(alphas <= 0.0):
        raise ValueError("Every l1-PLS alpha must be positive")
    penalty_scales = np.sqrt(np.mean(np.square(Phi), axis=0))
    nonconstant = penalty_scales > np.finfo(float).eps
    models = [None] * len(alphas)
    if not np.any(nonconstant):
        for index in range(len(alphas)):
            models[index] = {
                "intercept": float(np.mean(Y)),
                "coef": np.zeros(Phi.shape[1], dtype=float),
                "penalty_scales": penalty_scales,
                "n_iter": 0,
                "dual_gap": 0.0,
                "max_iter": int(max_iter),
                "converged_before_max_iter": True,
            }
        return models
    scaled_design = Phi[:, nonconstant] / penalty_scales[nonconstant]
    design_mean = np.mean(scaled_design, axis=0)
    response_mean = float(np.mean(Y))
    centered_design = scaled_design - design_mean
    centered_response = Y - response_mean
    descending_indices = np.argsort(-alphas, kind="stable")
    descending_alphas = alphas[descending_indices]
    (path_alphas, scaled_coefs, dual_gaps, n_iters) = lasso_path(
        centered_design,
        centered_response,
        alphas=descending_alphas,
        max_iter=int(max_iter),
        tol=0.0001,
        return_n_iter=True,
    )
    if not np.array_equal(path_alphas, descending_alphas):
        raise RuntimeError("sklearn returned an unexpected l1-PLS alpha path")
    for path_position, original_index in enumerate(descending_indices):
        scaled_coef = scaled_coefs[:, path_position]
        coef = np.zeros(Phi.shape[1], dtype=float)
        coef[nonconstant] = scaled_coef / penalty_scales[nonconstant]
        n_iter = int(n_iters[path_position])
        models[int(original_index)] = {
            "intercept": float(response_mean - design_mean @ scaled_coef),
            "coef": coef,
            "penalty_scales": penalty_scales,
            "n_iter": n_iter,
            "dual_gap": float(dual_gaps[path_position]),
            "max_iter": int(max_iter),
            "converged_before_max_iter": bool(n_iter < int(max_iter)),
        }
    return models


def _l1_pls_alpha_max(Phi, Y):
    Phi = np.asarray(Phi, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    penalty_scales = np.sqrt(np.mean(np.square(Phi), axis=0))
    nonconstant = penalty_scales > np.finfo(float).eps
    if not np.any(nonconstant):
        return np.finfo(float).eps
    scaled = Phi[:, nonconstant] / penalty_scales[nonconstant]
    centered_x = scaled - np.mean(scaled, axis=0)
    centered_y = Y - np.mean(Y)
    alpha_max = np.max(np.abs(centered_x.T @ centered_y)) / len(Y)
    return float(max(alpha_max, np.finfo(float).eps))


def _predict_l1_pls_empirical_rms(model, Phi):
    Phi = np.asarray(Phi, dtype=float)
    return float(model["intercept"]) + Phi @ np.asarray(model["coef"], dtype=float)


def _l1_pls_hajek_value(Y, A, d_pred, pi):
    Y = np.asarray(Y, dtype=float).reshape(-1)
    A = np.asarray(A).reshape(-1).astype(int)
    d_pred = np.asarray(d_pred).reshape(-1).astype(int)
    inv_pi = np.asarray(inverse_probability_weights(pi), dtype=float).reshape(-1)
    matched_weight = (A == d_pred).astype(float) * inv_pi
    denominator = float(np.sum(matched_weight))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("l1-PLS Hajek value has a nonpositive denominator")
    return float(np.sum(Y * matched_weight) / denominator)


def _l1_pls_active_rule_variables(model, p_features, K):
    interaction_start = p_features + (K - 1)
    interactions = np.asarray(model["coef"][interaction_start:], dtype=float).reshape(
        p_features, K - 1
    )
    return int(np.sum(np.any(interactions != 0.0, axis=1)))


def tune_l1_pls_ratio_honest(
    X_full_np, Y_full_np, A_full_np, honest_payloads, K=4, ratio_grid=L1_PLS_RATIO_GRID
):
    ratios = np.asarray(ratio_grid, dtype=float)
    cv_values = np.zeros(len(ratios), dtype=float)
    cv_mses = np.zeros(len(ratios), dtype=float)
    fold_alpha_max = []
    convergence = {str(float(ratio)): [] for ratio in ratios}
    encoder = OneHotEncoder(
        drop="first", sparse_output=False, categories=[range(K)]
    ).fit(np.asarray(A_full_np).reshape(-1, 1))
    for fold, payload in enumerate(honest_payloads):
        (X_sub, X_val) = payload["features"]["X"]
        Y_sub = np.asarray(payload["Y_train"], dtype=float)
        A_sub = np.asarray(payload["A_train"], dtype=int)
        Y_val = np.asarray(payload["Y_validation"], dtype=float)
        A_val = np.asarray(payload["A_validation"], dtype=int)
        pi_val = np.asarray(payload["pi"]["X"][1], dtype=float)
        Phi_sub = _build_l1_pls_design(X_sub, A_sub, encoder)
        Phi_val_observed = _build_l1_pls_design(X_val, A_val, encoder)
        Phi_val_all_arms = [
            _build_l1_pls_design(X_val, np.full(len(X_val), arm, dtype=int), encoder)
            for arm in range(K)
        ]
        alpha_max = _l1_pls_alpha_max(Phi_sub, Y_sub)
        fold_alpha_max.append(alpha_max)
        fold_alphas = ratios * alpha_max
        fold_models = _fit_l1_pls_empirical_rms_path(
            Phi_sub, Y_sub, fold_alphas, max_iter=L1_PLS_TUNING_MAX_ITER
        )
        for index, (ratio, model) in enumerate(zip(ratios, fold_models)):
            observed_prediction = _predict_l1_pls_empirical_rms(model, Phi_val_observed)
            cv_mses[index] += np.mean(np.square(Y_val - observed_prediction)) / len(
                honest_payloads
            )
            potential_predictions = np.column_stack(
                [
                    _predict_l1_pls_empirical_rms(model, Phi_arm)
                    for Phi_arm in Phi_val_all_arms
                ]
            )
            d_pred = np.argmax(potential_predictions, axis=1)
            cv_values[index] += _l1_pls_hajek_value(Y_val, A_val, d_pred, pi_val) / len(
                honest_payloads
            )
            convergence[str(float(ratio))].append(
                {
                    "fold": int(fold),
                    "alpha_max": float(alpha_max),
                    "absolute_alpha": float(fold_alphas[index]),
                    "n_iter": int(model["n_iter"]),
                    "dual_gap": float(model["dual_gap"]),
                    "converged_before_max_iter": bool(
                        model["converged_before_max_iter"]
                    ),
                }
            )
    X_full_np = np.asarray(X_full_np, dtype=float)
    Y_full_np = np.asarray(Y_full_np, dtype=float).reshape(-1)
    A_full_np = np.asarray(A_full_np).reshape(-1).astype(int)
    Phi_full = _build_l1_pls_design(X_full_np, A_full_np, encoder)
    full_alpha_max = _l1_pls_alpha_max(Phi_full, Y_full_np)
    full_alphas = ratios * full_alpha_max
    maximum_value = float(np.max(cv_values))
    value_tied_indices = np.flatnonzero(
        np.isclose(cv_values, maximum_value, rtol=0.0, atol=1e-12)
    )
    tied_models = _fit_l1_pls_empirical_rms_path(
        Phi_full,
        Y_full_np,
        full_alphas[value_tied_indices],
        max_iter=L1_PLS_FINAL_MAX_ITER,
    )
    active_rule_variables = {
        str(float(ratios[index])): _l1_pls_active_rule_variables(
            model, X_full_np.shape[1], K
        )
        for (index, model) in zip(value_tied_indices, tied_models)
    }
    minimum_active = min(active_rule_variables.values())
    sparsity_tied = np.asarray(
        [
            index
            for index in value_tied_indices
            if active_rule_variables[str(float(ratios[index]))] == minimum_active
        ],
        dtype=int,
    )
    minimum_mse = float(np.min(cv_mses[sparsity_tied]))
    prediction_tied = np.asarray(
        [
            index
            for index in sparsity_tied
            if np.isclose(cv_mses[index], minimum_mse, rtol=0.0, atol=1e-12)
        ],
        dtype=int,
    )
    selected_index = int(prediction_tied[0])
    selected_ratio = float(ratios[selected_index])
    selected_alpha = float(full_alphas[selected_index])
    return (
        selected_alpha,
        {
            "search": "deterministic_grid",
            "grid_scale": "alpha_over_fold_specific_alpha_max",
            "ratio_grid": ratios.tolist(),
            "fold_alpha_max": [float(value) for value in fold_alpha_max],
            "full_alpha_max": float(full_alpha_max),
            "cv_hajek_value_by_ratio": {
                str(float(ratio)): float(value)
                for (ratio, value) in zip(ratios, cv_values)
            },
            "cv_prediction_mse_by_ratio": {
                str(float(ratio)): float(value)
                for (ratio, value) in zip(ratios, cv_mses)
            },
            "active_rule_variables_by_value_tied_ratio": active_rule_variables,
            "minimum_active_rule_variables": int(minimum_active),
            "selected_ratio": selected_ratio,
            "selected_alpha": selected_alpha,
            "tie_rule": "value_then_rule_sparsity_then_prediction_mse_then_larger_ratio",
            "convergence_by_ratio_and_fold": convergence,
        },
    )


def evaluate_l1_pls_baseline(
    name,
    Feat_tr,
    Feat_te,
    Y_tr,
    A_tr,
    pi_tr,
    oracle_own_te,
    oracle_full_te,
    delta_all_te,
    margin_delta_own_te,
    feat_names,
    device,
    selected_alpha=None,
):
    X_tr_np = Feat_tr.cpu().numpy()
    X_te_np = Feat_te.cpu().numpy()
    Y_tr_np = Y_tr.cpu().numpy().flatten()
    A_tr_np = A_tr.cpu().numpy().flatten().astype(int)
    (n_samples_tr, p_features) = X_tr_np.shape
    K = 4
    enc = OneHotEncoder(drop="first", sparse_output=False, categories=[range(K)])
    enc.fit(A_tr_np.reshape(-1, 1))
    if selected_alpha is None:
        raise ValueError("A penalty selected by nested tuning is required.")
    best_alpha = float(selected_alpha)
    Phi_tr = _build_l1_pls_design(X_tr_np, A_tr_np, enc)
    final_model = _fit_l1_pls_empirical_rms(
        Phi_tr,
        Y_tr_np,
        alpha=best_alpha,
        max_iter=L1_PLS_FINAL_MAX_ITER,
        random_state=42,
    )
    K_minus_1 = K - 1
    coefficient = np.asarray(final_model["coef"], dtype=float)
    coef_X = coefficient[0:p_features]
    coef_A = coefficient[p_features : p_features + K_minus_1]
    coef_XA = coefficient[p_features + K_minus_1 :].reshape(p_features, K_minus_1)
    intercept = float(final_model["intercept"])
    B_equiv = np.zeros((1 + p_features, K))
    for k in range(K):
        if k == 0:
            B_equiv[0, k] = intercept
            B_equiv[1:, k] = coef_X
        else:
            B_equiv[0, k] = intercept + coef_A[k - 1]
            B_equiv[1:, k] = coef_X + coef_XA[:, k - 1]
    preds = np.zeros((X_te_np.shape[0], K))
    for k in range(K):
        preds[:, k] = B_equiv[0, k] + X_te_np @ B_equiv[1:, k]
    d_pred = np.argmax(preds, axis=1)
    decision_contrasts = np.zeros((1 + p_features, K_minus_1))
    decision_contrasts[0, :] = coef_A
    decision_contrasts[1:, :] = coef_XA
    perf = summarize_policy_performance(
        d_pred, delta_all_te, oracle_own_te, oracle_full_te, margin_delta_own_te
    )
    return perf


def fit_xue_angle_based_itr(
    U_tr, Y_weighted, V_enc, lambda_f, b=1.0, u0=0.0, max_iter=1000, lr=0.01
):
    p_features = U_tr.shape[1]
    K_minus_1 = V_enc.shape[1]
    B = torch.zeros(
        (p_features, K_minus_1),
        dtype=U_tr.dtype,
        device=U_tr.device,
        requires_grad=True,
    )
    optimizer = optim.Adam([B], lr=lr)
    for _ in range(max_iter):
        optimizer.zero_grad()
        f_X = U_tr @ B
        inner_prod = torch.sum(f_X * V_enc, dim=1)
        surrogate = torch.sigmoid(b * (inner_prod - u0))
        loss_main = -torch.mean(Y_weighted * surrogate)
        l2_penalty = lambda_f * torch.sum(B[1:, :] ** 2)
        loss = loss_main + l2_penalty
        loss.backward()
        optimizer.step()
    return B.detach()


def evaluate_xue_model(
    name,
    Feat_tr,
    Feat_te,
    Y_tr,
    A_tr,
    weight_tr,
    V_mat,
    V_tensor,
    oracle_own_te,
    oracle_full_te,
    delta_all_te,
    margin_delta_own_te,
    l_f,
    b_val,
    feat_names,
    device,
):
    U_tr = torch.cat([torch.ones((len(Y_tr), 1), device=device), Feat_tr], dim=1)
    V_enc = torch.tensor(
        V_mat[A_tr.long().view(-1).cpu().numpy()], dtype=torch.float64, device=device
    )
    Y_weighted = Y_tr.view(-1) * weight_tr.view(-1)
    B = fit_xue_angle_based_itr(U_tr, Y_weighted, V_enc, l_f, b=b_val, u0=0.0)
    U_te = torch.cat(
        [torch.ones((len(delta_all_te), 1), device=device), Feat_te], dim=1
    )
    d_pred = torch.argmax(U_te @ B @ V_tensor.T, dim=1)
    perf = summarize_policy_performance(
        d_pred, delta_all_te, oracle_own_te, oracle_full_te, margin_delta_own_te
    )
    return perf
