import numpy as np
import torch

PI_WEIGHT_CLIP_UPPER = 20.0
PI_PROB_CLIP_LOWER = 1.0 / PI_WEIGHT_CLIP_UPPER


def generate_simplex(K):
    W = np.zeros((K, K - 1))
    W[0, :] = 1.0 / np.sqrt(K - 1)
    const1 = -(1 + np.sqrt(K)) / (K - 1) ** 1.5
    const2 = np.sqrt(K / (K - 1))
    for j in range(1, K):
        w_j = np.full(K - 1, const1)
        w_j[j - 1] += const2
        W[j, :] = w_j
    return W


def block_soft_thresholding_with_intercept(B, lambda_t):
    B_new = B.clone()
    row_norms = torch.norm(B[1:, :], p=2, dim=1, keepdim=True)
    scaling = torch.clamp(1.0 - lambda_t / (row_norms + 1e-08), min=0.0)
    B_new[1:, :] = B[1:, :] * scaling
    return B_new


def fit_proximal_itr_linear_fista(U_tr, Y_pseudo, lambda_f, max_iter=2000, tol=1e-06):
    n_train = U_tr.shape[0]
    p_features = U_tr.shape[1]
    K_minus_1 = Y_pseudo.shape[1]
    UtU = U_tr.T @ U_tr / n_train
    UtY = U_tr.T @ Y_pseudo / n_train
    L_lipschitz = torch.linalg.eigvalsh(UtU).max().item() + 1e-07
    step_size = 1.0 / L_lipschitz
    B = torch.zeros((p_features, K_minus_1), dtype=U_tr.dtype, device=U_tr.device)
    Yk = B.clone()
    t = 1.0
    for _ in range(max_iter):
        B_old = B.clone()
        grad = UtU @ Yk - UtY
        B_next = block_soft_thresholding_with_intercept(
            Yk - step_size * grad, lambda_f * step_size
        )
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
        Yk = B_next + (t - 1.0) / t_next * (B_next - B_old)
        B = B_next
        t = t_next
        if torch.norm(B - B_old, p="fro") < tol:
            break
    return B


def fit_weighted_proximal_itr_linear_fista(
    U_tr, Y_pseudo, sample_weight, lambda_f, max_iter=2000, tol=1e-06
):
    n_train = U_tr.shape[0]
    p_features = U_tr.shape[1]
    K_minus_1 = Y_pseudo.shape[1]
    weights = sample_weight.view(-1, 1).to(dtype=U_tr.dtype, device=U_tr.device)
    U_weighted = U_tr * torch.sqrt(weights)
    Y_weighted = Y_pseudo * torch.sqrt(weights)
    UtU = U_weighted.T @ U_weighted / n_train
    UtY = U_weighted.T @ Y_weighted / n_train
    L_lipschitz = torch.linalg.eigvalsh(UtU).max().item() + 1e-07
    step_size = 1.0 / L_lipschitz
    B = torch.zeros((p_features, K_minus_1), dtype=U_tr.dtype, device=U_tr.device)
    Yk = B.clone()
    t = 1.0
    for _ in range(max_iter):
        B_old = B.clone()
        grad = UtU @ Yk - UtY
        B_next = block_soft_thresholding_with_intercept(
            Yk - step_size * grad, lambda_f * step_size
        )
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
        Yk = B_next + (t - 1.0) / t_next * (B_next - B_old)
        B = B_next
        t = t_next
        if torch.norm(B - B_old, p="fro") < tol:
            break
    return B


def inverse_probability_weights(pi, max_weight=PI_WEIGHT_CLIP_UPPER):
    if torch.is_tensor(pi):
        return torch.clamp(1.0 / pi.view(-1), max=max_weight)
    pi_np = np.asarray(pi, dtype=float).reshape(-1)
    return np.minimum(1.0 / np.clip(pi_np, PI_PROB_CLIP_LOWER, 1.0), max_weight)
