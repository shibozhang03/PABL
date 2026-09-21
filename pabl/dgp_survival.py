from __future__ import annotations
import numpy as np
import pandas as pd
from pabl.dgp import SurvivalCovariates as SurvivalCovariates


class WeibullSurvivalDGP(SurvivalCovariates):

    def survival_delta(self, U_value=None, linearity=None, U_inter=None):
        linearity = (
            getattr(self, "linearity", "linear") if linearity is None else linearity
        )
        U_inter = getattr(self, "U_inter", False) if U_inter is None else U_inter
        (x0, x1, x2) = (self.X[:, 0], self.X[:, 1], self.X[:, 2])
        if U_value is None:
            u = self.U
        else:
            u = np.asarray(U_value).reshape(-1)
            if u.shape[0] != self.X.shape[0]:
                raise ValueError("U_value must have the same length as X")
        delta = np.zeros((self.X.shape[0], self.K))
        if linearity == "linear":
            delta[:, 0] = 0.2 + 0.8 * x0 - 0.35 * x1 + 0.2 * x2
            delta[:, 1] = -0.1 - 0.45 * x0 + 0.75 * x1 + 0.15 * x2
            delta[:, 2] = 0.05 + 0.35 * x0 + 0.2 * x1 - 0.75 * x2
            delta[:, 3] = -0.15 - 0.65 * x0 - 0.3 * x1 + 0.65 * x2
        elif linearity == "nonlinear":
            delta[:, 0] = 0.75 * np.sin(x0) - 0.45 * x1 + 0.2 * x2
            delta[:, 1] = -0.4 * x0 + 0.75 * np.tanh(x1) + 0.25 * x2
            delta[:, 2] = 0.35 * x0 - 0.2 * x1 - 0.7 * np.sin(x2)
            delta[:, 3] = -0.55 * np.tanh(x0 + x2) - 0.25 * x1 + 0.6 * x2
        else:
            raise ValueError("linearity must be 'linear' or 'nonlinear'")
        if U_inter:
            delta[:, 0] += 0.4 * u
            delta[:, 1] += -0.36 * u
            delta[:, 2] += 0.28 * u
            delta[:, 3] += -0.32 * u
        delta -= delta.mean(axis=1, keepdims=True)
        return delta

    def survival_main_effect(self, linearity=None, U_inter=None):
        linearity = (
            getattr(self, "linearity", "linear") if linearity is None else linearity
        )
        (x0, x1, x2) = (self.X[:, 0], self.X[:, 1], self.X[:, 2])
        (c0, c1, c2) = (
            self.X_irre[:, 0] - 0.5,
            self.X_irre[:, 1] - 0.5,
            self.X_irre[:, 2] - 0.5,
        )
        if linearity == "linear":
            return (
                0.2 * x0
                - 0.15 * x1
                + 0.1 * x2
                + 0.35 * self.W
                + 0.9 * self.U
                + 0.08 * c0
                - 0.06 * c1
                + 0.05 * c2
            )
        if linearity == "nonlinear":
            return (
                0.2 * x0
                - 0.15 * x1
                + 0.1 * x2
                + 0.45 * self.W
                + 0.9 * self.U
                + 0.18 * (self.U**2 - 1.0)
                + 0.25 * np.sin(x0 * x1)
                + 0.2 * np.cos(x2)
                + 0.12 * x0 * x2
                + 0.08 * c0
                - 0.06 * c1
                + 0.05 * c2
            )
        raise ValueError("linearity must be 'linear' or 'nonlinear'")

    def survival_probability(self, t_eval):
        if not hasattr(self, "scale_all"):
            raise AttributeError(
                "Call gen_survival(...) before requesting survival probabilities"
            )
        t_eval = float(t_eval)
        if t_eval <= 0:
            raise ValueError("t_eval must be positive")
        return np.exp(-((t_eval / self.scale_all) ** self.weibull_shape))

    def _surv_prob_from_delta(self, delta, t_eval=None):
        if not hasattr(self, "base_scale"):
            raise AttributeError(
                "Call gen_survival(...) before requesting survival probabilities"
            )
        t_eval = self.milestone_t if t_eval is None else float(t_eval)
        eta = self.main_effect.reshape(-1, 1) + self.effect_scale * np.asarray(
            delta, dtype=float
        )
        eta = np.clip(eta, -2.5, 2.5)
        scale = self.base_scale * np.exp(eta)
        return np.exp(-((t_eval / scale) ** self.weibull_shape))

    def compute_oracle_regimes(self, t_eval=None, return_surv=False):
        if not hasattr(self, "delta_all"):
            raise AttributeError(
                "Call gen_survival(...) before computing oracle regimes"
            )
        u_x = np.zeros(self.X.shape[0])
        u_xw = self.expected_U_given_XW()
        surv_x = self._surv_prob_from_delta(
            self.survival_delta(U_value=u_x), t_eval=t_eval
        )
        surv_xw = self._surv_prob_from_delta(
            self.survival_delta(U_value=u_xw), t_eval=t_eval
        )
        surv_full = self.survival_probability(
            self.milestone_t if t_eval is None else float(t_eval)
        )
        oracle_x = np.argmax(surv_x, axis=1)
        oracle_xw = np.argmax(surv_xw, axis=1)
        oracle_full = np.argmax(surv_full, axis=1)
        if return_surv:
            return (oracle_x, oracle_xw, oracle_full, surv_x, surv_xw, surv_full)
        return (oracle_x, oracle_xw, oracle_full)

    def gen_survival(
        self,
        linearity="linear",
        U_inter=False,
        milestone_t=1.0,
        weibull_shape=1.5,
        target_survival_at_zero=0.55,
        effect_scale=0.55,
        censor_base_rate=0.15,
    ):
        self.linearity = linearity
        self.U_inter = U_inter
        self.milestone_t = float(milestone_t)
        self.weibull_shape = float(weibull_shape)
        self.target_survival_at_zero = float(target_survival_at_zero)
        self.effect_scale = float(effect_scale)
        self.censor_base_rate = float(censor_base_rate)
        n = len(self.A)
        rng_t = np.random.default_rng(self.seed[5])
        rng_c = np.random.default_rng(self.seed[6])
        self.delta_all = self.survival_delta(linearity=linearity, U_inter=U_inter)
        self.main_effect = self.survival_main_effect(
            linearity=linearity, U_inter=U_inter
        )
        eta_all_unbounded = (
            self.main_effect.reshape(-1, 1) + self.effect_scale * self.delta_all
        )
        self.eta_all = np.clip(eta_all_unbounded, -2.5, 2.5)
        self.eta_clipping_applied = bool(np.any(self.eta_all != eta_all_unbounded))
        self.base_scale = self.milestone_t / (
            -np.log(self.target_survival_at_zero)
        ) ** (1.0 / self.weibull_shape)
        self.scale_all = self.base_scale * np.exp(self.eta_all)
        self.surv_prob_all = self.survival_probability(self.milestone_t)
        self.GOR = np.argmax(self.surv_prob_all, axis=1)
        self.T_all = self.scale_all * rng_t.weibull(
            self.weibull_shape, size=(n, self.K)
        )
        self.T_actual = self.T_all[np.arange(n), self.A]
        (x0, x1) = (self.X[:, 0], self.X[:, 1])
        censor_lp_unbounded = (
            0.2 * x0
            - 0.15 * x1
            + 0.1 * self.W
            + 0.12 * (self.A == 1)
            - 0.12 * (self.A == 2)
        )
        self.censor_lp = np.clip(censor_lp_unbounded, -1.5, 1.5)
        self.censor_lp_clipping_applied = bool(
            np.any(self.censor_lp != censor_lp_unbounded)
        )
        self.censor_rate = self.censor_base_rate * np.exp(self.censor_lp)
        self.C = rng_c.exponential(scale=1.0 / self.censor_rate, size=n)
        self.Y_obs = np.minimum(self.T_actual, self.C)
        self.Delta = (self.T_actual <= self.C).astype(float)
        return self.get_survival_dataframe()

    def get_survival_dataframe(self):
        df = pd.DataFrame({"A": self.A})
        for i in range(self.X.shape[1]):
            df[f"X{i}"] = self.X[:, i]
        for i in range(self.X_irre.shape[1]):
            df[f"X_irre{i}"] = self.X_irre[:, i]
        df["Z"] = self.Z
        df["W"] = self.W
        df["U"] = self.U
        df["Y_obs"] = self.Y_obs
        df["Delta"] = self.Delta
        df["T_actual"] = self.T_actual
        df["GOR"] = self.GOR
        df["S_opt"] = self.surv_prob_all[np.arange(len(self.A)), self.GOR]
        df["S_A"] = self.surv_prob_all[np.arange(len(self.A)), self.A]
        for k in range(self.K):
            df[f"T_{k}"] = self.T_all[:, k]
            df[f"scale_{k}"] = self.scale_all[:, k]
            df[f"surv_prob_{k}"] = self.surv_prob_all[:, k]
        return df
