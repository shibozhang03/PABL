from pabl.common import generate_simplex, inverse_probability_weights
from typing import Any, Dict, List
import numpy as np
import torch
from pabl.continuous import (
    Q_CROSSFIT_SPLITS,
    H_CROSSFIT_SPLITS,
    Q_BRIDGE_CV,
    H_BRIDGE_CV,
    PI_CROSSFIT_SPLITS,
    generate_data_wrapper,
    evaluate_model,
    evaluate_qi_ad_model,
    evaluate_full_arm_h_model,
    evaluate_full_arm_dr_model,
    evaluate_l1_pls_baseline,
    tune_l1_pls_ratio_honest,
    fit_xue_angle_based_itr,
    evaluate_xue_model,
)
from pabl.tuning import (
    SIM_NUISANCE_FOLDS,
    build_continuous_honest_payloads,
    fit_transform_policy_features,
    prepare_final_continuous_group_fit,
    select_continuous_group_ratio,
    select_continuous_xue_grid,
)


def result_rows_from_results(
    results: Dict[str, Dict[str, Any]],
    rep: int,
    train_seed: int,
    test_seed: int,
    n_train: int,
    n_test: int,
    linearity: str,
    U_inter: bool,
) -> List[Dict[str, Any]]:
    return [
        {
            "rep": int(rep),
            "train_seed": int(train_seed),
            "test_seed": int(test_seed),
            "n_train": int(n_train),
            "n_test": int(n_test),
            "linearity": linearity,
            "U_inter": bool(U_inter),
            "method_full": name,
            "true_val": float(result["true_val"]),
        }
        for (name, result) in results.items()
    ]


def run_one_rep(
    rep: int,
    train_seed: int,
    test_seed: int,
    n_train: int = 5000,
    n_test: int = 50000,
    linearity: str = "linear",
    U_inter: bool = False,
    K_arms: int = 4,
    device="cpu",
    q_crossfit_splits: int = Q_CROSSFIT_SPLITS,
    q_bridge_cv: int = Q_BRIDGE_CV,
    h_crossfit_splits: int = H_CROSSFIT_SPLITS,
    h_bridge_cv: int = H_BRIDGE_CV,
    bridge_n_components: int = 50,
    pi_crossfit_splits: int = PI_CROSSFIT_SPLITS,
):
    if isinstance(device, str):
        device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
    if K_arms != 4:
        raise ValueError(
            "The current continuous DGP is configured for exactly 4 treatment arms; set K_arms=4."
        )
    if int(bridge_n_components) < 2:
        raise ValueError("bridge_n_components must be at least 2.")
    V_mat = generate_simplex(K_arms)
    V_tensor = torch.tensor(V_mat, dtype=torch.float64, device=device)
    train = generate_data_wrapper(
        n=n_train,
        seed=train_seed,
        linearity=linearity,
        U_inter=U_inter,
        device=device,
        fit_preprocess=True,
    )
    (
        W_tr,
        X_tr,
        Z_tr,
        U_tr,
        A_tr,
        Y_tr,
        _Aopt_tr,
        delta_all_tr,
        names_tr,
        X_raw_tr,
        preprocessors,
        oracle_x_tr,
        oracle_xz_tr,
        oracle_xw_tr,
        oracle_xwz_tr,
        oracle_full_tr,
        delta_x_tr,
        delta_xz_tr,
        delta_xw_tr,
        delta_xwz_tr,
        delta_full_tr,
    ) = train
    test = generate_data_wrapper(
        n=n_test,
        seed=test_seed,
        linearity=linearity,
        U_inter=U_inter,
        device=device,
        preprocessors=preprocessors,
        fit_preprocess=False,
    )
    (
        W_te,
        X_te,
        Z_te,
        U_te,
        A_te,
        Y_te,
        A_opt_te,
        delta_all_te,
        _names_te,
        X_raw_te,
        _prep_te,
        oracle_x_te,
        oracle_xz_te,
        oracle_xw_te,
        oracle_xwz_te,
        oracle_full_te,
        delta_x_te,
        delta_xz_te,
        delta_xw_te,
        delta_xwz_te,
        delta_full_te,
    ) = test
    raw_train = {
        "X": X_raw_tr.detach().cpu().numpy(),
        "W": W_tr.detach().cpu().numpy(),
        "Z": Z_tr.detach().cpu().numpy(),
        "U": U_tr.detach().cpu().numpy(),
    }
    raw_test = {
        "X": X_raw_te.detach().cpu().numpy(),
        "W": W_te.detach().cpu().numpy(),
        "Z": Z_te.detach().cpu().numpy(),
        "U": U_te.detach().cpu().numpy(),
    }
    final_feature_np = {
        spec: fit_transform_policy_features(
            raw_train, raw_test, spec=spec, linearity=linearity
        )
        for spec in ("X", "XU", "XW", "XZ", "XWZ")
    }
    final_feature_t = {
        spec: (
            torch.tensor(train_np, dtype=torch.float64, device=device),
            torch.tensor(test_np, dtype=torch.float64, device=device),
        )
        for (spec, (train_np, test_np)) in final_feature_np.items()
    }
    (Feat_X_tr, Feat_X_te) = final_feature_t["X"]
    (Feat_XU_tr, Feat_XU_te) = final_feature_t["XU"]
    (Feat_XW_tr, Feat_XW_te) = final_feature_t["XW"]
    (Feat_XZ_tr, Feat_XZ_te) = final_feature_t["XZ"]
    (Feat_XWZ_tr, Feat_XWZ_te) = final_feature_t["XWZ"]
    names_XU = ["Intercept"] + names_tr + ["U_True"]
    names_XW = ["Intercept"] + names_tr + ["W_Proxy"]
    names_XZ = ["Intercept"] + names_tr + ["Z_Proxy"]
    names_X = ["Intercept"] + names_tr
    A_tr_np_flat = A_tr.detach().cpu().numpy().ravel().astype(int)
    if not (
        int(pi_crossfit_splits) == SIM_NUISANCE_FOLDS
        and (not True or int(q_crossfit_splits) == SIM_NUISANCE_FOLDS)
        and (not True or int(h_crossfit_splits) == SIM_NUISANCE_FOLDS)
    ):
        raise ValueError("Simulation nuisance cross-fitting is frozen at 3 folds")
    (_split_plan, nested_policy_payloads, final_nuisance_payload) = (
        build_continuous_honest_payloads(
            raw=raw_train,
            treatment=A_tr_np_flat,
            outcome=Y_tr.detach().cpu().numpy().reshape(-1),
            linearity=linearity,
            policy_specs=("X", "XU", "XW", "XZ", "XWZ"),
            propensity_specs=("X", "XU", "XW", "XZ", "XWZ"),
            need_q=True,
            need_h=True,
            bridge_cv=q_bridge_cv,
            n_components=int(bridge_n_components),
            seed=42,
        )
    )

    def final_pi(spec):
        return torch.tensor(
            final_nuisance_payload["pi"][spec][0], dtype=torch.float64, device=device
        )

    pi_oracle_tr = final_pi("XU")
    pi_naive_xwz_tr = final_pi("XWZ")
    pi_naive_x_tr = final_pi("X")
    q_policy_np = np.asarray(final_nuisance_payload["q"][0], dtype=float)
    q_tr = torch.tensor(q_policy_np, dtype=torch.float64, device=device)
    h_obs_np = np.asarray(final_nuisance_payload["h"][0], dtype=float)
    h_cf_np = np.asarray(final_nuisance_payload["h"][1], dtype=float)
    h_obs_tr = torch.tensor(h_obs_np, dtype=torch.float64, device=device)
    h_cf_tr = torch.tensor(h_cf_np, dtype=torch.float64, device=device)
    results: Dict[str, Dict[str, Any]] = {}

    def choose_group_penalty(label, feature_spec, method, pi_spec="X"):
        (ratio, grid_diagnostics) = select_continuous_group_ratio(
            nested_policy_payloads,
            feature_spec=feature_spec,
            method=method,
            V=V_mat,
            pi_spec=pi_spec,
        )
        (_response, _weight, penalty) = prepare_final_continuous_group_fit(
            final_nuisance_payload,
            features=final_feature_np[feature_spec][0],
            method=method,
            V=V_mat,
            pi_spec=pi_spec,
            selected_ratio=ratio,
        )
        return (float(penalty), float(ratio))

    if U_inter:
        (l_f_or, rho_or) = choose_group_penalty("Oracle(X,U)", "XU", "qi", pi_spec="XU")
        results["Oracle(X, U)"] = evaluate_qi_ad_model(
            "Oracle(X, U)",
            Feat_XU_tr,
            Feat_XU_te,
            Y_tr,
            A_tr,
            pi_oracle_tr,
            V_mat,
            V_tensor,
            oracle_full_te,
            oracle_full_te,
            delta_all_te,
            delta_full_te,
            l_f_or,
            names_XU,
            device,
        )
    else:
        (l_f_or, rho_or) = choose_group_penalty("Oracle(X)", "X", "qi", pi_spec="XU")
        results["Oracle(X)"] = evaluate_qi_ad_model(
            "Oracle(X)",
            Feat_X_tr,
            Feat_X_te,
            Y_tr,
            A_tr,
            pi_oracle_tr,
            V_mat,
            V_tensor,
            oracle_x_te,
            oracle_full_te,
            delta_all_te,
            delta_x_te,
            l_f_or,
            names_X,
            device,
        )
    (l_f_qi_x, rho_qi_x) = choose_group_penalty("Qi AD(X)", "X", "qi", pi_spec="XWZ")
    results["Qi AD(X)"] = evaluate_qi_ad_model(
        "Qi AD(X)",
        Feat_X_tr,
        Feat_X_te,
        Y_tr,
        A_tr,
        pi_naive_xwz_tr,
        V_mat,
        V_tensor,
        oracle_x_te,
        oracle_full_te,
        delta_all_te,
        delta_x_te,
        l_f_qi_x,
        names_X,
        device,
    )
    (l1_alpha, l1_tuning) = tune_l1_pls_ratio_honest(
        Feat_X_tr.detach().cpu().numpy(),
        Y_tr.detach().cpu().numpy().reshape(-1),
        A_tr_np_flat,
        nested_policy_payloads,
        K=K_arms,
    )
    l1_pls_result = evaluate_l1_pls_baseline(
        "l1-PLS(X)",
        Feat_X_tr,
        Feat_X_te,
        Y_tr,
        A_tr,
        pi_naive_x_tr,
        oracle_x_te,
        oracle_full_te,
        delta_all_te,
        delta_x_te,
        names_X,
        device,
        selected_alpha=l1_alpha,
    )
    results["l1-PLS(X)"] = l1_pls_result
    (l_f_xue_x, b_xue_x, xue_grid) = select_continuous_xue_grid(
        nested_policy_payloads,
        V=V_mat,
        fit_function=fit_xue_angle_based_itr,
        feature_spec="X",
        propensity_spec="XWZ",
    )
    weight_xue_x = inverse_probability_weights(pi_naive_xwz_tr)
    results["Xue(X)"] = evaluate_xue_model(
        "Xue(X)",
        Feat_X_tr,
        Feat_X_te,
        Y_tr,
        A_tr,
        weight_xue_x,
        V_mat,
        V_tensor,
        oracle_x_te,
        oracle_full_te,
        delta_all_te,
        delta_x_te,
        l_f_xue_x,
        b_xue_x,
        names_X,
        device,
    )
    (l_f_fah_zx, rho_fah_zx) = choose_group_penalty(
        "Proximal(Z,X)", "XZ", "full_h", pi_spec="X"
    )
    results["Proximal(Z, X)"] = evaluate_full_arm_h_model(
        "Proximal(Z, X)",
        Feat_XZ_tr,
        Feat_XZ_te,
        h_cf_tr,
        V_mat,
        V_tensor,
        oracle_xz_te,
        oracle_full_te,
        delta_all_te,
        delta_xz_te,
        l_f_fah_zx,
        names_XZ,
        device,
    )
    (l_f_fah_x, rho_fah_x) = choose_group_penalty(
        "Proximal-h(X)", "X", "full_h", pi_spec="X"
    )
    results["Proximal-h(X)"] = evaluate_full_arm_h_model(
        "Proximal-h(X)",
        Feat_X_tr,
        Feat_X_te,
        h_cf_tr,
        V_mat,
        V_tensor,
        oracle_x_te,
        oracle_full_te,
        delta_all_te,
        delta_x_te,
        l_f_fah_x,
        names_X,
        device,
    )
    (l_f_px_xw, rho_px_xw) = choose_group_penalty(
        "Proximal(X,W)", "XW", "prox_q", pi_spec="X"
    )
    results["Proximal(X, W)"] = evaluate_model(
        "Proximal(X, W)",
        Feat_XW_tr,
        Feat_XW_te,
        Y_tr,
        A_tr,
        q_tr,
        V_mat,
        V_tensor,
        oracle_xw_te,
        oracle_full_te,
        delta_all_te,
        delta_xw_te,
        l_f_px_xw,
        names_XW,
        device,
    )
    (l_f_px_x, rho_px_x) = choose_group_penalty(
        "Proximal-q(X)", "X", "prox_q", pi_spec="X"
    )
    results["Proximal-q(X)"] = evaluate_model(
        "Proximal-q(X)",
        Feat_X_tr,
        Feat_X_te,
        Y_tr,
        A_tr,
        q_tr,
        V_mat,
        V_tensor,
        oracle_x_te,
        oracle_full_te,
        delta_all_te,
        delta_x_te,
        l_f_px_x,
        names_X,
        device,
    )
    (l_f_fadr_x, rho_fadr_x) = choose_group_penalty(
        "DR Proximal(X)", "X", "full_dr", pi_spec="X"
    )
    results["DR Proximal(X)"] = evaluate_full_arm_dr_model(
        "DR Proximal(X)",
        Feat_X_tr,
        Feat_X_te,
        Y_tr,
        A_tr,
        q_tr,
        h_obs_tr,
        h_cf_tr,
        V_mat,
        V_tensor,
        oracle_x_te,
        oracle_full_te,
        delta_all_te,
        delta_x_te,
        l_f_fadr_x,
        names_X,
        device,
    )
    rows = result_rows_from_results(
        results, rep, train_seed, test_seed, n_train, n_test, linearity, U_inter
    )
    return rows
