import torch
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import Lasso
import warnings
from sklearn.preprocessing import OneHotEncoder
from lifelines import KaplanMeierFitter
from pabl.tuning import goldberg_alpha_max
from pabl.common import fit_proximal_itr_linear_fista

Q_CROSSFIT_SPLITS = 3
Q_BRIDGE_CV = 3
H_CROSSFIT_SPLITS = 3
H_BRIDGE_CV = 3
PI_CROSSFIT_SPLITS = 3
IPCW_CROSSFIT_SPLITS = 3
IPCW_SURV_CLIP_LOWER = 0.001
USE_CROSSFIT_GOLDBERG_IPCW = True
GOLDBERG_IPCW_CROSSFIT_SPLITS = 3


def _call_gen_survival_with_milestone(sim, linearity, U_inter, t_milestone):
    try:
        return sim.gen_survival(
            linearity=linearity, U_inter=U_inter, milestone_t=t_milestone
        )
    except TypeError:
        try:
            return sim.gen_survival(
                linearity=linearity, U_inter=U_inter, t_milestone=t_milestone
            )
        except TypeError:
            warnings.warn(
                "Invalid preprocessing input: use fitted training transformations for prediction.",
                RuntimeWarning,
            )
            return sim.gen_survival(linearity=linearity, U_inter=U_inter)


def _extract_surv_prob_all(sim, df, K_arms):
    candidate_attrs = [
        "surv_prob_all",
        "S_all",
        "survival_prob_all",
        "true_surv_prob_all",
        "S_t_all",
    ]
    for attr in candidate_attrs:
        if hasattr(sim, attr):
            arr = np.asarray(getattr(sim, attr), dtype=float)
            if arr.ndim == 2 and arr.shape[1] == K_arms:
                return arr
    column_patterns = [
        [f"surv_prob_{k}" for k in range(K_arms)],
        [f"S_{k}" for k in range(K_arms)],
        [f"S{k}" for k in range(K_arms)],
        [f"true_surv_{k}" for k in range(K_arms)],
    ]
    for cols in column_patterns:
        if all((c in df.columns for c in cols)):
            return df[cols].values.astype(float)
    raise AttributeError(
        "Invalid preprocessing input: use fitted training transformations for prediction."
    )


def compute_true_policy_survival_value_torch(
    d_pred, true_T_all_te, surv_prob_all_te, t_milestone
):
    if surv_prob_all_te is not None:
        idx = torch.arange(len(d_pred), device=d_pred.device)
        return torch.mean(surv_prob_all_te[idx, d_pred.long()]).item()
    idx = torch.arange(len(d_pred), device=d_pred.device)
    true_surv_time = true_T_all_te[idx, d_pred.long()]
    return torch.mean((true_surv_time > t_milestone).float()).item()


def compute_true_policy_survival_value_numpy(
    d_pred, true_T_all_te, surv_prob_all_te, t_milestone
):
    if torch.is_tensor(surv_prob_all_te):
        surv_prob_np = surv_prob_all_te.detach().cpu().numpy()
    else:
        surv_prob_np = surv_prob_all_te
    if surv_prob_np is not None:
        return float(np.mean(surv_prob_np[np.arange(len(d_pred)), d_pred.astype(int)]))
    if torch.is_tensor(true_T_all_te):
        true_T_np = true_T_all_te.detach().cpu().numpy()
    else:
        true_T_np = true_T_all_te
    true_surv_time = true_T_np[np.arange(len(d_pred)), d_pred.astype(int)]
    return float(np.mean((true_surv_time > t_milestone).astype(float)))


def prepare_ad_learning_data_tensor(X_tensor, A_tensor, W_tensor):
    (n_samples, p) = X_tensor.shape
    K_minus_1 = W_tensor.shape[1]
    W_A = W_tensor[A_tensor.long().view(-1)]
    X_star_vec = torch.einsum("ni,nj->nij", X_tensor, W_A).reshape(
        n_samples, p * K_minus_1
    )
    group_indices = torch.repeat_interleave(torch.arange(p), K_minus_1).to(
        X_tensor.device
    )
    return (X_star_vec, group_indices)


def weighted_cox_loss(beta, X_star, events, times, weights):
    risk_scores = torch.matmul(X_star, beta)
    sort_idx = torch.argsort(times.view(-1), descending=True)
    events_sorted = events.view(-1)[sort_idx]
    risk_scores_sorted = risk_scores.view(-1)[sort_idx]
    weights_sorted = weights.view(-1)[sort_idx]
    exp_risk = torch.exp(risk_scores_sorted)
    cumsum_exp_risk = torch.cumsum(exp_risk, dim=0)
    log_risk_set = torch.log(cumsum_exp_risk + 1e-10)
    log_lik = weights_sorted * events_sorted * (risk_scores_sorted - log_risk_set)
    return -torch.mean(log_lik)


def proximal_group_lasso_step_tensor(beta, lr, lambda_reg, group_indices, K_minus_1):
    beta_prox = torch.zeros_like(beta)
    p = len(beta) // K_minus_1
    for j in range(p):
        idx = group_indices == j
        beta_group = beta[idx]
        if j == 0:
            beta_prox[idx] = beta_group
        else:
            l2_norm = torch.norm(beta_group, p=2)
            threshold = lr * lambda_reg
            if l2_norm > threshold:
                scaling = 1.0 - threshold / l2_norm
                beta_prox[idx] = beta_group * scaling
            else:
                beta_prox[idx] = 0.0
    return beta_prox


def fit_qi_cox_ad_learning(
    U_tr, A_tr, Y_time, Delta_event, weights, V_tensor, lambda_f, lr=0.05, epochs=400
):
    (X_star, group_indices) = prepare_ad_learning_data_tensor(U_tr, A_tr, V_tensor)
    (p, K_minus_1) = (U_tr.shape[1], V_tensor.shape[1])
    beta = torch.zeros(
        p * K_minus_1, dtype=torch.float64, device=U_tr.device, requires_grad=True
    )
    optimizer = torch.optim.Adam([beta], lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = weighted_cox_loss(beta, X_star, Delta_event, Y_time, weights)
        loss.backward()
        with torch.no_grad():
            beta -= lr * beta.grad
            beta.copy_(
                proximal_group_lasso_step_tensor(
                    beta, lr, lambda_f, group_indices, K_minus_1
                )
            )
    return beta.detach().view(p, K_minus_1)


def xue_weighted_km_loss(
    theta_flat,
    U_tensor,
    A_tensor,
    Y_tensor,
    Delta_tensor,
    inv_pi_tensor,
    V_tensor,
    t_g,
    b=1.0,
    u0=-1.0,
    lam=0.01,
):
    p = U_tensor.shape[1]
    K_minus_1 = V_tensor.shape[1]
    theta = theta_flat.view(p, K_minus_1)
    f_X = torch.matmul(U_tensor, theta)
    V_A = V_tensor[A_tensor.long().view(-1)]
    u = torch.sum(V_A * f_X, dim=1)
    l_u = 1.0 / (1.0 + torch.exp(-b * (u - u0)))
    W = l_u * inv_pi_tensor.view(-1)
    failure_mask = (Delta_tensor.view(-1) == 1) & (Y_tensor.view(-1) <= t_g)
    failure_times = torch.unique(Y_tensor[failure_mask])
    if len(failure_times) == 0:
        return lam * torch.sum(theta[1:, :] ** 2)
    Y_flat = Y_tensor.view(-1, 1)
    Delta_flat = Delta_tensor.view(-1, 1)
    fail_t_row = failure_times.view(1, -1)
    mask_eq = (Y_flat == fail_t_row) & (Delta_flat == 1)
    mask_ge = Y_flat >= fail_t_row
    W_col = W.view(-1, 1)
    num = torch.sum(W_col * mask_eq, dim=0)
    den = torch.sum(W_col * mask_ge, dim=0)
    valid_mask = den > 1e-08
    hazard = torch.zeros_like(den)
    hazard[valid_mask] = num[valid_mask] / den[valid_mask]
    hazard = torch.clamp(hazard, min=0.0, max=1.0 - 1e-08)
    log_S = torch.sum(torch.log(1.0 - hazard))
    penalty = lam * torch.sum(theta[1:, :] ** 2)
    return -log_S + penalty


def fit_xue_angle_based(
    U_tr,
    A_tr,
    Y_tr,
    Delta_tr,
    prop_weights,
    V_tensor,
    t_g,
    lambda_f,
    b_val=1.0,
    lr=0.05,
    epochs=300,
):
    (p, K_minus_1) = (U_tr.shape[1], V_tensor.shape[1])
    theta = torch.zeros(
        p * K_minus_1, dtype=torch.float64, device=U_tr.device, requires_grad=True
    )
    optimizer = torch.optim.Adam([theta], lr=lr)
    prev_loss = float("inf")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = xue_weighted_km_loss(
            theta,
            U_tr,
            A_tr,
            Y_tr,
            Delta_tr,
            prop_weights,
            V_tensor,
            t_g,
            b=b_val,
            lam=lambda_f,
        )
        loss.backward()
        optimizer.step()
        current_loss = loss.item()
        if abs(prev_loss - current_loss) < 1e-05:
            break
        prev_loss = current_loss
    return theta.detach().view(p, K_minus_1)


def compute_goldberg_ipcw_numpy(
    Y_time,
    Delta_event,
    t_milestone,
    A_t=None,
    use_crossfit_ipcw=USE_CROSSFIT_GOLDBERG_IPCW,
    n_splits=GOLDBERG_IPCW_CROSSFIT_SPLITS,
    random_state=42,
):
    Y_np = (
        Y_time.cpu().numpy().flatten() if torch.is_tensor(Y_time) else Y_time.flatten()
    )
    Delta_np = (
        Delta_event.cpu().numpy().flatten()
        if torch.is_tensor(Delta_event)
        else Delta_event.flatten()
    )
    A_np = (
        A_t.detach().cpu().numpy().flatten().astype(int)
        if torch.is_tensor(A_t)
        else (
            np.asarray(A_t).flatten().astype(int)
            if A_t is not None
            else np.zeros_like(Delta_np, dtype=int)
        )
    )
    Y_target = np.clip(Y_np, a_min=None, a_max=t_milestone)
    epsilon = 1e-05
    Y_query = np.clip(Y_target - epsilon, a_min=0.0, a_max=None)
    S_c = np.zeros_like(Y_np, dtype=float)
    if use_crossfit_ipcw:
        splits = make_ipcw_crossfit_splits(
            A_np, Delta_np, n_splits=n_splits, random_state=random_state
        )
        for train_idx, val_idx in splits:
            kmf_c = KaplanMeierFitter()
            kmf_c.fit(Y_np[train_idx], 1 - Delta_np[train_idx])
            S_c[val_idx] = kmf_c.survival_function_at_times(Y_query[val_idx]).values
    else:
        kmf_c = KaplanMeierFitter()
        kmf_c.fit(Y_np, 1 - Delta_np)
        S_c = kmf_c.survival_function_at_times(Y_query).values
    S_c = np.clip(S_c, IPCW_SURV_CLIP_LOWER, 1.0)
    valid_mask = (Delta_np == 1) | (Y_np >= t_milestone)
    ipcw = np.zeros_like(Y_np)
    ipcw[valid_mask] = 1.0 / S_c[valid_mask]
    return (ipcw, valid_mask, Y_target)


def evaluate_goldberg_l1_pls(
    name,
    Feat_tr,
    Feat_te,
    Y_time_tr,
    Delta_tr,
    A_tr,
    A_opt_te,
    true_T_all_te,
    surv_prob_all_te,
    feat_names,
    t_milestone,
    device,
    use_crossfit_ipcw=USE_CROSSFIT_GOLDBERG_IPCW,
    ipcw_crossfit_splits=GOLDBERG_IPCW_CROSSFIT_SPLITS,
    selected_ratio=None,
):
    X_tr_np = Feat_tr.cpu().numpy()
    X_te_np = Feat_te.cpu().numpy()
    A_tr_np = A_tr.cpu().numpy().flatten().astype(int)
    K = 4
    (ipcw_weights, valid_mask, Y_target) = compute_goldberg_ipcw_numpy(
        Y_time_tr,
        Delta_tr,
        t_milestone,
        A_t=A_tr,
        use_crossfit_ipcw=use_crossfit_ipcw,
        n_splits=ipcw_crossfit_splits,
    )
    X_valid = X_tr_np[valid_mask]
    A_valid = A_tr_np[valid_mask]
    Y_valid = Y_target[valid_mask]
    w_valid = ipcw_weights[valid_mask]
    (n_samples_valid, p_features) = X_valid.shape
    enc = OneHotEncoder(drop="first", sparse_output=False, categories=[range(K)])
    A_dummies_valid = enc.fit_transform(A_valid.reshape(-1, 1))
    K_minus_1 = A_dummies_valid.shape[1]
    interactions_valid = (X_valid[:, :, None] * A_dummies_valid[:, None, :]).reshape(
        n_samples_valid, -1
    )
    Phi_valid = np.hstack([X_valid, A_dummies_valid, interactions_valid])
    if selected_ratio is None:
        raise ValueError(
            "Goldberg requires a preselected honest-grid alpha/alpha_max ratio"
        )
    alpha_max = goldberg_alpha_max(Phi_valid, Y_valid, w_valid)
    selected_alpha = float(selected_ratio) * alpha_max
    model = Lasso(
        alpha=selected_alpha, fit_intercept=True, random_state=42, max_iter=50000
    )
    model.fit(Phi_valid, Y_valid, sample_weight=w_valid)
    coef_X = model.coef_[0:p_features]
    coef_A = model.coef_[p_features : p_features + K_minus_1]
    coef_XA = model.coef_[p_features + K_minus_1 :].reshape(p_features, K_minus_1)
    intercept = model.intercept_
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
    true_prob = compute_true_policy_survival_value_numpy(
        d_pred=d_pred,
        true_T_all_te=true_T_all_te,
        surv_prob_all_te=surv_prob_all_te,
        t_milestone=t_milestone,
    )
    return {"true_val": true_prob}


def make_ipcw_crossfit_splits(
    A_t, Delta_event, n_splits=IPCW_CROSSFIT_SPLITS, random_state=42
):
    A_np = (
        A_t.detach().cpu().numpy().ravel().astype(int)
        if torch.is_tensor(A_t)
        else np.asarray(A_t).ravel().astype(int)
    )
    Delta_np = (
        Delta_event.detach().cpu().numpy().ravel().astype(int)
        if torch.is_tensor(Delta_event)
        else np.asarray(Delta_event).ravel().astype(int)
    )
    censor_event_np = 1 - Delta_np
    n = len(A_np)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for cross-fitted IPCW")
    if n_splits > n:
        raise ValueError("n_splits cannot exceed sample size for cross-fitted IPCW")
    joint_strata = A_np * 2 + censor_event_np
    for strata in (joint_strata, A_np):
        counts = np.bincount(strata)
        positive_counts = counts[counts > 0]
        if positive_counts.size > 0 and np.min(positive_counts) >= n_splits:
            skf = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            return list(skf.split(np.zeros(n), strata))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(np.zeros(n)))


def evaluate_model(
    name,
    Feat_tr,
    Feat_te,
    Y_tr,
    A_tr,
    weight_tr,
    V_mat,
    V_tensor,
    A_opt_te,
    true_T_all_te,
    surv_prob_all_te,
    l_f,
    feat_names,
    t_milestone,
    device,
):
    U_tr = torch.cat([torch.ones((len(Y_tr), 1), device=device), Feat_tr], dim=1)
    I_Y_greater_t = (Y_tr.view(-1, 1) > t_milestone).float()
    raw_weights = I_Y_greater_t * weight_tr.view(-1, 1)
    V_enc_tr = torch.tensor(
        V_mat[A_tr.long().view(-1).cpu()], dtype=torch.float64, device=device
    )
    Y_ps = raw_weights * V_enc_tr
    B = fit_proximal_itr_linear_fista(U_tr, Y_ps, l_f)
    U_te = torch.cat([torch.ones((len(A_opt_te), 1), device=device), Feat_te], dim=1)
    d_pred = torch.argmax(U_te @ B @ V_tensor.T, dim=1)
    (unique_arms, counts) = torch.unique(d_pred, return_counts=True)
    true_prob = compute_true_policy_survival_value_torch(
        d_pred=d_pred,
        true_T_all_te=true_T_all_te,
        surv_prob_all_te=surv_prob_all_te,
        t_milestone=t_milestone,
    )
    return {"true_val": true_prob}


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
    if h_all.ndim != 2 or h_all.shape[1] != V_tensor_local.shape[0]:
        raise ValueError("h_cf_tr must have one column per treatment arm")
    if not torch.isfinite(h_all).all():
        raise ValueError("h_cf_tr contains non-finite values")
    return h_all @ V_tensor_local


def build_full_arm_dr_pseudo_response(
    Y_fixed_tr, A_tr, q_tr, h_obs_tr, h_cf_tr, V_matrix, dtype=None, device=None
):
    if device is None:
        for obj in (Y_fixed_tr, A_tr, q_tr, h_obs_tr, h_cf_tr):
            if torch.is_tensor(obj):
                device = obj.device
                break
        if device is None:
            device = torch.device("cpu")
    if dtype is None:
        for obj in (Y_fixed_tr, q_tr, h_obs_tr, h_cf_tr):
            if torch.is_tensor(obj) and obj.is_floating_point():
                dtype = obj.dtype
                break
        if dtype is None:
            dtype = torch.float64
    Y_vec = torch.as_tensor(Y_fixed_tr, dtype=dtype, device=device).view(-1)
    A_vec = torch.as_tensor(A_tr, dtype=torch.long, device=device).view(-1)
    q_vec = torch.as_tensor(q_tr, dtype=dtype, device=device).view(-1)
    h_obs = torch.as_tensor(h_obs_tr, dtype=dtype, device=device).view(-1)
    h_all = torch.as_tensor(h_cf_tr, dtype=dtype, device=device)
    V_tensor_local = torch.as_tensor(V_matrix, dtype=dtype, device=device)
    K_arms = V_tensor_local.shape[0]
    if h_all.shape != (len(Y_vec), K_arms):
        raise ValueError("h_cf_tr must have shape (n, K_arms)")
    if not len(Y_vec) == len(A_vec) == len(q_vec) == len(h_obs):
        raise ValueError("Y_fixed_tr, A_tr, q_tr, h_obs_tr, and h_cf_tr must match")
    if torch.any((A_vec < 0) | (A_vec >= K_arms)):
        raise ValueError("A_tr contains treatment labels outside 0, ..., K-1")
    if not torch.isfinite(h_all).all():
        raise ValueError("h_cf_tr contains non-finite values")
    pseudo_all = h_all.clone()
    rows = torch.arange(len(Y_vec), device=device)
    pseudo_all[rows, A_vec] += q_vec * (Y_vec - h_obs)
    return (pseudo_all @ V_tensor_local, pseudo_all)


def evaluate_full_arm_h_model(
    name,
    Feat_tr,
    Feat_te,
    h_cf_tr,
    V_mat,
    V_tensor,
    A_opt_te,
    true_T_all_te,
    surv_prob_all_te,
    l_f,
    feat_names,
    t_milestone,
    device,
):
    dtype = Feat_tr.dtype
    Y_ps = build_full_arm_h_pseudo_response(h_cf_tr, V_mat, dtype=dtype, device=device)
    U_tr = torch.cat(
        [torch.ones((len(Feat_tr), 1), dtype=dtype, device=device), Feat_tr], dim=1
    )
    B = fit_proximal_itr_linear_fista(U_tr, Y_ps, l_f)
    U_te = torch.cat(
        [torch.ones((len(A_opt_te), 1), dtype=dtype, device=device), Feat_te], dim=1
    )
    d_pred = torch.argmax(U_te @ B @ V_tensor.T, dim=1)
    true_prob = compute_true_policy_survival_value_torch(
        d_pred=d_pred,
        true_T_all_te=true_T_all_te,
        surv_prob_all_te=surv_prob_all_te,
        t_milestone=t_milestone,
    )
    return {"true_val": true_prob}


def evaluate_full_arm_dr_model(
    name,
    Feat_tr,
    Feat_te,
    Y_fixed_tr,
    A_tr,
    q_tr,
    h_obs_tr,
    h_cf_tr,
    V_mat,
    V_tensor,
    A_opt_te,
    true_T_all_te,
    surv_prob_all_te,
    l_f,
    feat_names,
    t_milestone,
    device,
):
    dtype = Feat_tr.dtype
    (Y_ps, _pseudo_all) = build_full_arm_dr_pseudo_response(
        Y_fixed_tr, A_tr, q_tr, h_obs_tr, h_cf_tr, V_mat, dtype=dtype, device=device
    )
    U_tr = torch.cat(
        [torch.ones((len(A_tr), 1), dtype=dtype, device=device), Feat_tr], dim=1
    )
    B = fit_proximal_itr_linear_fista(U_tr, Y_ps, l_f)
    U_te = torch.cat(
        [torch.ones((len(A_opt_te), 1), dtype=dtype, device=device), Feat_te], dim=1
    )
    d_pred = torch.argmax(U_te @ B @ V_tensor.T, dim=1)
    true_prob = compute_true_policy_survival_value_torch(
        d_pred=d_pred,
        true_T_all_te=true_T_all_te,
        surv_prob_all_te=surv_prob_all_te,
        t_milestone=t_milestone,
    )
    return {"true_val": true_prob}


def evaluate_baseline_model(
    name,
    baseline_type,
    Feat_tr,
    Feat_te,
    Y_time_tr,
    Delta_tr,
    A_tr,
    prop_weights,
    V_tensor,
    A_opt_te,
    true_T_all_te,
    surv_prob_all_te,
    l_f,
    feat_names,
    t_milestone,
    device,
    b_val=1.0,
):
    U_tr = torch.cat([torch.ones((len(Y_time_tr), 1), device=device), Feat_tr], dim=1)
    if baseline_type == "qi":
        B = fit_qi_cox_ad_learning(
            U_tr,
            A_tr,
            Y_time_tr,
            Delta_tr,
            prop_weights,
            V_tensor,
            l_f,
            lr=0.05,
            epochs=400,
        )
    elif baseline_type == "xue":
        B = fit_xue_angle_based(
            U_tr,
            A_tr,
            Y_time_tr,
            Delta_tr,
            prop_weights,
            V_tensor,
            t_milestone,
            l_f,
            b_val=b_val,
            lr=0.05,
            epochs=400,
        )
    U_te = torch.cat([torch.ones((len(A_opt_te), 1), device=device), Feat_te], dim=1)
    if baseline_type == "qi":
        d_pred = torch.argmax(-(U_te @ B) @ V_tensor.T, dim=1)
    elif baseline_type == "xue":
        d_pred = torch.argmax(U_te @ B @ V_tensor.T, dim=1)
    true_prob = compute_true_policy_survival_value_torch(
        d_pred=d_pred,
        true_T_all_te=true_T_all_te,
        surv_prob_all_te=surv_prob_all_te,
        t_milestone=t_milestone,
    )
    return {"true_val": true_prob}
