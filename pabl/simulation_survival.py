from pabl.common import generate_simplex, inverse_probability_weights
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, SplineTransformer
from pabl.dgp_survival import WeibullSurvivalDGP
from pabl.survival import (
    GOLDBERG_IPCW_CROSSFIT_SPLITS,
    H_BRIDGE_CV,
    H_CROSSFIT_SPLITS,
    IPCW_CROSSFIT_SPLITS,
    PI_CROSSFIT_SPLITS,
    Q_BRIDGE_CV,
    Q_CROSSFIT_SPLITS,
    USE_CROSSFIT_GOLDBERG_IPCW,
    _call_gen_survival_with_milestone,
    _extract_surv_prob_all,
    evaluate_baseline_model,
    evaluate_full_arm_dr_model,
    evaluate_full_arm_h_model,
    evaluate_goldberg_l1_pls,
    evaluate_model,
    fit_qi_cox_ad_learning,
    fit_xue_angle_based,
)
from pabl.tuning import (
    SIM_BRIDGE_COMPONENTS,
    SIM_NUISANCE_FOLDS,
    build_survival_honest_payloads,
    fit_transform_policy_features,
    make_simulation_split_plan,
    prepare_final_continuous_group_fit,
    select_continuous_group_ratio,
    select_goldberg_ratio_grid,
    select_survival_qi_ratio_grid,
    select_survival_xue_grid,
    survival_cox_group_lambda_max,
)
from typing import Dict, List, Tuple, Any


def compute_surv_prob_from_weibull_scale(
    scale_all: np.ndarray, t_eval: float, weibull_shape: float
) -> np.ndarray:
    t_eval = float(t_eval)
    weibull_shape = float(weibull_shape)
    if t_eval <= 0:
        raise ValueError("t_eval must be positive")
    if weibull_shape <= 0:
        raise ValueError("weibull_shape must be positive")
    scale_all = np.asarray(scale_all, dtype=float)
    if np.any(scale_all <= 0) or not np.isfinite(scale_all).all():
        raise ValueError("Weibull scales must be finite and positive")
    return np.exp(-((t_eval / scale_all) ** weibull_shape))


def generate_survival_bundle(
    n: int,
    seed: int,
    linearity: str,
    U_inter: bool,
    reference_t: float,
    device="cpu",
    K_arms: int = 4,
    spline_transformer=None,
    scaler=None,
    fit_preprocessor: bool = True,
    default_logtime_sigma: float = 1.0,
    survival_dgp: str = "weibull",
) -> Dict[str, Any]:
    dgp_class = WeibullSurvivalDGP
    sim = dgp_class(samp_size=n, add_noise=True, seed=seed, K=K_arms)
    df = _call_gen_survival_with_milestone(sim, linearity, U_inter, reference_t)
    W = df["W"].values.reshape(-1, 1)
    Z = df["Z"].values.reshape(-1, 1)
    U = df["U"].values.reshape(-1, 1)
    A = df["A"].values.reshape(-1, 1)
    Y_obs = (
        df["Y_obs"].values.reshape(-1, 1)
        if "Y_obs" in df.columns
        else df["Y"].values.reshape(-1, 1)
    )
    Delta = df["Delta"].values.reshape(-1, 1)
    X_cols = [
        col
        for col in df.columns
        if col.startswith("X") and (not col.startswith("X_irre"))
    ]
    X_irre_cols = [col for col in df.columns if col.startswith("X_irre")]
    X_raw = df[X_cols + X_irre_cols].values
    raw_names = [f"X_{i + 1}" for i in range(X_raw.shape[1])]
    fitted_spline = spline_transformer
    if linearity == "linear":
        X_processed = X_raw
        feature_names = raw_names
        fitted_spline = None
    else:
        if fit_preprocessor:
            fitted_spline = SplineTransformer(n_knots=4, degree=3, include_bias=False)
            X_processed = fitted_spline.fit_transform(X_raw)
        else:
            if fitted_spline is None:
                raise ValueError(
                    "Invalid preprocessing input: use fitted training transformations for prediction."
                )
            X_processed = fitted_spline.transform(X_raw)
        feature_names = list(
            fitted_spline.get_feature_names_out(input_features=raw_names)
        )
    fitted_scaler = scaler
    if fit_preprocessor:
        fitted_scaler = StandardScaler()
        X_scaled = fitted_scaler.fit_transform(X_processed)
    else:
        if fitted_scaler is None:
            raise ValueError(
                "Invalid preprocessing input: use fitted training transformations for prediction."
            )
        X_scaled = fitted_scaler.transform(X_processed)
    true_T_all = np.asarray(sim.T_all, dtype=float)
    log_time_mean_all = None
    sigma_t = None
    scale_all = None
    weibull_shape = None
    scale_all = np.asarray(sim.scale_all, dtype=float)
    weibull_shape = float(sim.weibull_shape)
    surv_prob_ref = _extract_surv_prob_all(sim, df, K_arms=K_arms)
    return {
        "W": torch.tensor(W, dtype=torch.float64, device=device),
        "X": torch.tensor(X_scaled, dtype=torch.float64, device=device),
        "Z": torch.tensor(Z, dtype=torch.float64, device=device),
        "U": torch.tensor(U, dtype=torch.float64, device=device),
        "A": torch.tensor(A, dtype=torch.float64, device=device),
        "Y_obs": torch.tensor(Y_obs, dtype=torch.float64, device=device),
        "Delta": torch.tensor(Delta, dtype=torch.float64, device=device),
        "true_T_all": torch.tensor(true_T_all, dtype=torch.float64, device=device),
        "surv_prob_ref": torch.tensor(
            surv_prob_ref, dtype=torch.float64, device=device
        ),
        "X_raw_tensor": torch.tensor(X_raw, dtype=torch.float64, device=device),
        "feature_names": feature_names,
        "spline": fitted_spline,
        "scaler": fitted_scaler,
        "log_time_mean_all_np": log_time_mean_all,
        "scale_all_np": scale_all,
        "weibull_shape": weibull_shape,
        "true_T_all_np": true_T_all,
        "surv_prob_ref_np": surv_prob_ref,
        "sigma_t": sigma_t,
        "survival_dgp": "weibull",
        "dgp_eta_clipping_applied": bool(getattr(sim, "eta_clipping_applied", False)),
        "dgp_censor_lp_clipping_applied": bool(
            getattr(sim, "censor_lp_clipping_applied", False)
        ),
        "observed_censoring_rate": float(1.0 - np.mean(np.asarray(sim.Delta))),
        "reference_t": float(reference_t),
        "seed": int(seed),
    }


def make_eval_targets(
    bundle: Dict[str, Any], t_eval: float, K_arms: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = bundle["A"].device
    surv_prob_np = compute_surv_prob_from_weibull_scale(
        bundle["scale_all_np"], t_eval, bundle["weibull_shape"]
    )
    a_opt_np = np.argmax(surv_prob_np, axis=1)
    return (
        torch.tensor(a_opt_np, dtype=torch.long, device=device),
        torch.tensor(surv_prob_np, dtype=torch.float64, device=device),
    )


def result_rows_from_results(
    results: Dict[str, Dict[str, Any]],
    rep: int,
    t0: float,
    train_seed: int,
    test_seed: int,
    n_train: int,
    n_test: int,
    linearity: str,
    U_inter: bool,
    reference_t: float,
    K_arms: int = 4,
    survival_dgp: str = "weibull",
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
            "t0": float(t0),
        }
        for (name, result) in results.items()
    ]


def run_goldberg_oracle_xu_multi_t(
    rep: int,
    train_seed: int,
    test_seed: int,
    t_grid: List[float],
    reference_t: float = 1.0,
    n_train: int = 2000,
    n_test: int = 20000,
    linearity: str = "linear",
    U_inter: bool = False,
    K_arms: int = 4,
    device="cpu",
    default_logtime_sigma: float = 1.0,
    survival_dgp: str = "weibull",
    use_crossfit_goldberg_ipcw: bool = USE_CROSSFIT_GOLDBERG_IPCW,
    goldberg_ipcw_crossfit_splits: int = GOLDBERG_IPCW_CROSSFIT_SPLITS,
):
    if isinstance(device, str):
        device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
    train = generate_survival_bundle(
        n=n_train,
        seed=train_seed,
        linearity=linearity,
        U_inter=U_inter,
        reference_t=reference_t,
        device=device,
        K_arms=K_arms,
        fit_preprocessor=True,
        default_logtime_sigma=default_logtime_sigma,
        survival_dgp=survival_dgp,
    )
    test = generate_survival_bundle(
        n=n_test,
        seed=test_seed,
        linearity=linearity,
        U_inter=U_inter,
        reference_t=reference_t,
        device=device,
        K_arms=K_arms,
        spline_transformer=train["spline"],
        scaler=train["scaler"],
        fit_preprocessor=False,
        default_logtime_sigma=default_logtime_sigma,
        survival_dgp=survival_dgp,
    )
    U_tr = train["U"] if train["U"].ndim > 1 else train["U"].view(-1, 1)
    raw_train = {
        "X": train["X_raw_tensor"].detach().cpu().numpy(),
        "W": train["W"].detach().cpu().numpy(),
        "Z": train["Z"].detach().cpu().numpy(),
        "U": U_tr.detach().cpu().numpy(),
    }
    raw_test = {
        "X": test["X_raw_tensor"].detach().cpu().numpy(),
        "W": test["W"].detach().cpu().numpy(),
        "Z": test["Z"].detach().cpu().numpy(),
        "U": test["U"].detach().cpu().numpy(),
    }
    (xu_train_np, xu_test_np) = fit_transform_policy_features(
        raw_train, raw_test, spec="XU", linearity=linearity
    )
    Feat_XU_tr = torch.tensor(xu_train_np, dtype=torch.float64, device=device)
    Feat_XU_te = torch.tensor(xu_test_np, dtype=torch.float64, device=device)
    A_train_np = train["A"].detach().cpu().numpy().reshape(-1).astype(int)
    split_plan = make_simulation_split_plan(A_train_np, seed=42)
    u_dim = U_tr.shape[1]
    u_names = ["U"] if u_dim == 1 else [f"U_{j + 1}" for j in range(u_dim)]
    names_XU = ["Intercept"] + train["feature_names"] + u_names
    all_rows: List[Dict[str, Any]] = []
    for t0 in t_grid:
        t0 = float(t0)
        (A_opt_te_t, surv_prob_all_te_t) = make_eval_targets(test, t0, K_arms=K_arms)
        (goldberg_ratio, goldberg_grid) = select_goldberg_ratio_grid(
            raw_train,
            A_train_np,
            train["Y_obs"].detach().cpu().numpy().reshape(-1),
            train["Delta"].detach().cpu().numpy().reshape(-1),
            linearity,
            t0,
            split_plan,
            feature_spec="XU",
        )
        res = evaluate_goldberg_l1_pls(
            name="Goldberg Q (X,U)",
            Feat_tr=Feat_XU_tr,
            Feat_te=Feat_XU_te,
            Y_time_tr=train["Y_obs"],
            Delta_tr=train["Delta"],
            A_tr=train["A"],
            A_opt_te=A_opt_te_t,
            true_T_all_te=test["true_T_all"],
            surv_prob_all_te=surv_prob_all_te_t,
            feat_names=names_XU,
            t_milestone=t0,
            device=device,
            use_crossfit_ipcw=use_crossfit_goldberg_ipcw,
            ipcw_crossfit_splits=goldberg_ipcw_crossfit_splits,
            selected_ratio=goldberg_ratio,
        )
        rows = result_rows_from_results(
            results={"Goldberg Q (X,U)": res},
            rep=rep,
            t0=t0,
            train_seed=train_seed,
            test_seed=test_seed,
            n_train=n_train,
            n_test=n_test,
            linearity=linearity,
            U_inter=U_inter,
            reference_t=reference_t,
            K_arms=K_arms,
            survival_dgp=survival_dgp,
        )
        all_rows.extend(rows)
    return all_rows


def run_one_rep_multi_t(
    rep: int,
    train_seed: int,
    test_seed: int,
    t_grid: List[float],
    reference_t: float = 1.0,
    n_train: int = 2000,
    n_test: int = 20000,
    linearity: str = "linear",
    U_inter: bool = False,
    K_arms: int = 4,
    device="cpu",
    default_logtime_sigma: float = 1.0,
    q_crossfit_splits: int = Q_CROSSFIT_SPLITS,
    q_bridge_cv: int = Q_BRIDGE_CV,
    h_crossfit_splits: int = H_CROSSFIT_SPLITS,
    h_bridge_cv: int = H_BRIDGE_CV,
    pi_crossfit_splits: int = PI_CROSSFIT_SPLITS,
    ipcw_crossfit_splits: int = IPCW_CROSSFIT_SPLITS,
    goldberg_ipcw_crossfit_splits: int = GOLDBERG_IPCW_CROSSFIT_SPLITS,
):
    if isinstance(device, str):
        device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
    V_mat = generate_simplex(K_arms)
    V_tensor = torch.tensor(V_mat, dtype=torch.float64, device=device)
    train = generate_survival_bundle(
        n=n_train,
        seed=train_seed,
        linearity=linearity,
        U_inter=U_inter,
        reference_t=reference_t,
        device=device,
        K_arms=K_arms,
        fit_preprocessor=True,
        default_logtime_sigma=default_logtime_sigma,
        survival_dgp="weibull",
    )
    test = generate_survival_bundle(
        n=n_test,
        seed=test_seed,
        linearity=linearity,
        U_inter=U_inter,
        reference_t=reference_t,
        device=device,
        K_arms=K_arms,
        spline_transformer=train["spline"],
        scaler=train["scaler"],
        fit_preprocessor=False,
        default_logtime_sigma=default_logtime_sigma,
        survival_dgp="weibull",
    )
    raw_train = {
        "X": train["X_raw_tensor"].detach().cpu().numpy(),
        "W": train["W"].detach().cpu().numpy(),
        "Z": train["Z"].detach().cpu().numpy(),
        "U": train["U"].detach().cpu().numpy(),
    }
    raw_test = {
        "X": test["X_raw_tensor"].detach().cpu().numpy(),
        "W": test["W"].detach().cpu().numpy(),
        "Z": test["Z"].detach().cpu().numpy(),
        "U": test["U"].detach().cpu().numpy(),
    }
    final_feature_np = {
        spec: fit_transform_policy_features(
            raw_train, raw_test, spec=spec, linearity=linearity
        )
        for spec in ("X", "XW", "XZ", "XWZ")
    }
    final_feature_t = {
        spec: (
            torch.tensor(train_np, dtype=torch.float64, device=device),
            torch.tensor(test_np, dtype=torch.float64, device=device),
        )
        for (spec, (train_np, test_np)) in final_feature_np.items()
    }
    (Feat_X_tr, Feat_X_te) = final_feature_t["X"]
    (Feat_XW_tr, Feat_XW_te) = final_feature_t["XW"]
    (Feat_XZ_tr, Feat_XZ_te) = final_feature_t["XZ"]
    (Feat_XWZ_tr, Feat_XWZ_te) = final_feature_t["XWZ"]
    names_XW = ["Intercept"] + train["feature_names"] + ["W_Proxy"]
    names_XZ = ["Intercept"] + train["feature_names"] + ["Z_Proxy"]
    names_X = ["Intercept"] + train["feature_names"]
    A_tr_np = train["A"].detach().cpu().numpy().ravel().astype(int)
    need_h_any = bool(True or True)
    if not all(
        (
            int(value) == SIM_NUISANCE_FOLDS
            for value in (
                q_crossfit_splits,
                h_crossfit_splits,
                pi_crossfit_splits,
                ipcw_crossfit_splits,
                goldberg_ipcw_crossfit_splits,
            )
        )
    ):
        raise ValueError(
            "Simulation nuisance/censoring cross-fitting is frozen at 3 folds"
        )
    all_rows: List[Dict[str, Any]] = []
    for t0 in t_grid:
        t0 = float(t0)
        (A_opt_te_t, surv_prob_all_te_t) = make_eval_targets(test, t0, K_arms=K_arms)
        (split_plan, nested_policy_payloads, final_nuisance_payload) = (
            build_survival_honest_payloads(
                raw=raw_train,
                treatment=A_tr_np,
                y_time=train["Y_obs"].detach().cpu().numpy().reshape(-1),
                delta=train["Delta"].detach().cpu().numpy().reshape(-1),
                linearity=linearity,
                t_eval=t0,
                policy_specs=("X", "XW", "XZ", "XWZ"),
                propensity_specs=("X", "XW", "XWZ"),
                need_q=True,
                need_h=need_h_any,
                bridge_cv=q_bridge_cv,
                n_components=SIM_BRIDGE_COMPONENTS,
                seed=42,
            )
        )
        q_tr_np = np.asarray(final_nuisance_payload["q"][0], dtype=float)
        q_tr = torch.tensor(q_tr_np, dtype=torch.float64, device=device)
        Y_fixed_np = np.asarray(final_nuisance_payload["Y_train"], dtype=float)
        Y_fixed_tr = torch.tensor(Y_fixed_np, dtype=torch.float64, device=device)
        pi_naive_xwz_tr = torch.tensor(
            final_nuisance_payload["pi"]["XWZ"][0], dtype=torch.float64, device=device
        )
        propensity_weights = inverse_probability_weights(pi_naive_xwz_tr)
        weight_px = q_tr * Y_fixed_tr
        need_h = need_h_any
        if need_h:
            h_obs_np = np.asarray(final_nuisance_payload["h"][0], dtype=float)
            h_cf_np = np.asarray(final_nuisance_payload["h"][1], dtype=float)
            h_obs_tr = torch.tensor(h_obs_np, dtype=torch.float64, device=device)
            h_cf_tr = torch.tensor(h_cf_np, dtype=torch.float64, device=device)
        else:
            h_obs_np = h_cf_np = None
            h_obs_tr = h_cf_tr = None
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

        (qi_ratio, qi_grid) = select_survival_qi_ratio_grid(
            nested_policy_payloads,
            V_mat,
            fit_qi_cox_ad_learning,
            t_eval=t0,
            feature_spec="X",
            propensity_spec="XWZ",
        )
        qi_lambda_max = survival_cox_group_lambda_max(
            final_feature_np["X"][0],
            A_tr_np,
            train["Y_obs"].detach().cpu().numpy().reshape(-1),
            train["Delta"].detach().cpu().numpy().reshape(-1),
            propensity_weights.detach().cpu().numpy(),
            V_mat,
        )
        l_f_qi_x = float(qi_ratio * qi_lambda_max)
        results["Qi AD (X)"] = evaluate_baseline_model(
            "Qi AD (X)",
            "qi",
            Feat_X_tr,
            Feat_X_te,
            train["Y_obs"],
            train["Delta"],
            train["A"],
            propensity_weights,
            V_tensor,
            A_opt_te_t,
            test["true_T_all"],
            surv_prob_all_te_t,
            l_f_qi_x,
            names_X,
            t0,
            device,
        )
        (l_f_xue_x, b_xue_x, xue_grid) = select_survival_xue_grid(
            nested_policy_payloads,
            V_mat,
            fit_xue_angle_based,
            t_eval=t0,
            feature_spec="X",
            propensity_spec="XWZ",
        )
        results["Xue Angle (X)"] = evaluate_baseline_model(
            "Xue Angle (X)",
            "xue",
            Feat_X_tr,
            Feat_X_te,
            train["Y_obs"],
            train["Delta"],
            train["A"],
            propensity_weights,
            V_tensor,
            A_opt_te_t,
            test["true_T_all"],
            surv_prob_all_te_t,
            l_f_xue_x,
            names_X,
            t0,
            device,
            b_val=b_xue_x,
        )
        (goldberg_ratio, goldberg_grid) = select_goldberg_ratio_grid(
            raw_train,
            A_tr_np,
            train["Y_obs"].detach().cpu().numpy().reshape(-1),
            train["Delta"].detach().cpu().numpy().reshape(-1),
            linearity,
            t0,
            split_plan,
            feature_spec="X",
        )
        results["Goldberg Q (X)"] = evaluate_goldberg_l1_pls(
            name="Goldberg Q (X)",
            Feat_tr=Feat_X_tr,
            Feat_te=Feat_X_te,
            Y_time_tr=train["Y_obs"],
            Delta_tr=train["Delta"],
            A_tr=train["A"],
            A_opt_te=A_opt_te_t,
            true_T_all_te=test["true_T_all"],
            surv_prob_all_te=surv_prob_all_te_t,
            feat_names=names_X,
            t_milestone=t0,
            device=device,
            use_crossfit_ipcw=True,
            ipcw_crossfit_splits=goldberg_ipcw_crossfit_splits,
            selected_ratio=goldberg_ratio,
        )
        (l_f_fah_xz, rho_fah_xz) = choose_group_penalty(
            "Proximal(Z,X)", "XZ", "full_h", pi_spec="X"
        )
        results["Proximal(Z, X)"] = evaluate_full_arm_h_model(
            "Proximal(Z, X)",
            Feat_XZ_tr,
            Feat_XZ_te,
            h_cf_tr,
            V_mat,
            V_tensor,
            A_opt_te_t,
            test["true_T_all"],
            surv_prob_all_te_t,
            l_f_fah_xz,
            names_XZ,
            t0,
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
            A_opt_te_t,
            test["true_T_all"],
            surv_prob_all_te_t,
            l_f_fah_x,
            names_X,
            t0,
            device,
        )
        (l_f_px_xw, rho_px_xw) = choose_group_penalty(
            "Proximal(X,W)", "XW", "prox_q", pi_spec="X"
        )
        results["Proximal(X, W)"] = evaluate_model(
            "Proximal(X, W)",
            Feat_XW_tr,
            Feat_XW_te,
            train["Y_obs"],
            train["A"],
            weight_px,
            V_mat,
            V_tensor,
            A_opt_te_t,
            test["true_T_all"],
            surv_prob_all_te_t,
            l_f_px_xw,
            names_XW,
            t0,
            device,
        )
        (l_f_px_x, rho_px_x) = choose_group_penalty(
            "Proximal-q(X)", "X", "prox_q", pi_spec="X"
        )
        results["Proximal-q(X)"] = evaluate_model(
            "Proximal-q(X)",
            Feat_X_tr,
            Feat_X_te,
            train["Y_obs"],
            train["A"],
            weight_px,
            V_mat,
            V_tensor,
            A_opt_te_t,
            test["true_T_all"],
            surv_prob_all_te_t,
            l_f_px_x,
            names_X,
            t0,
            device,
        )
        (l_f_fadr_x, rho_fadr_x) = choose_group_penalty(
            "DR Proximal(X)", "X", "full_dr", pi_spec="X"
        )
        results["DR Proximal(X)"] = evaluate_full_arm_dr_model(
            "DR Proximal(X)",
            Feat_X_tr,
            Feat_X_te,
            Y_fixed_tr,
            train["A"],
            q_tr,
            h_obs_tr,
            h_cf_tr,
            V_mat,
            V_tensor,
            A_opt_te_t,
            test["true_T_all"],
            surv_prob_all_te_t,
            l_f_fadr_x,
            names_X,
            t0,
            device,
        )
        rows = result_rows_from_results(
            results=results,
            rep=rep,
            t0=t0,
            train_seed=train_seed,
            test_seed=test_seed,
            n_train=n_train,
            n_test=n_test,
            linearity=linearity,
            U_inter=U_inter,
            reference_t=reference_t,
            K_arms=K_arms,
            survival_dgp="weibull",
        )
        all_rows.extend(rows)
    return all_rows
