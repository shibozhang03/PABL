import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import uniform
from scipy.stats import randint
import pandas as pd

CONTINUOUS_X_MEAN = np.array([0.0, 0.0, 0.0], dtype=float)
CONTINUOUS_X_COV = np.array(
    [[1.0, 0.25, -0.15], [0.25, 1.0, 0.2], [-0.15, 0.2, 1.0]], dtype=float
)
CONTINUOUS_TREATMENT_INTERCEPTS = np.array([0.0, 0.1, -0.05, 0.05], dtype=float)
CONTINUOUS_TREATMENT_X_COEFS = np.array(
    [[0.0, 0.0, 0.0], [0.45, -0.25, 0.1], [-0.25, 0.5, 0.2], [0.15, -0.35, 0.55]],
    dtype=float,
)
CONTINUOUS_TREATMENT_U_COEFS = np.array([0.0, -0.65, 0.2, 0.65], dtype=float)
CONTINUOUS_PROXY_PARAMS = {
    "w_intercept": 0.15,
    "w_x0_coef": 0.35,
    "w_u_coef": 0.55,
    "w_noise_sd": 0.9,
    "z_intercept": -0.1,
    "z_x1_coef": -0.3,
    "z_u_coef": 0.65,
    "z_noise_sd": 0.9,
}
CONTINUOUS_OUTCOME_NOISE_SCALE = 1.0


def continuous_treatment_logits(X, U, K=4):
    X = np.asarray(X)
    U = np.asarray(U).reshape(-1)
    if K > len(CONTINUOUS_TREATMENT_U_COEFS):
        raise ValueError("K exceeds configured continuous treatment coefficients")
    if K > CONTINUOUS_TREATMENT_X_COEFS.shape[0]:
        raise ValueError("K exceeds configured continuous treatment X coefficients")
    logits = (
        CONTINUOUS_TREATMENT_INTERCEPTS[:K].reshape(1, -1)
        + X[:, :3] @ CONTINUOUS_TREATMENT_X_COEFS[:K, :3].T
        + U.reshape(-1, 1) * CONTINUOUS_TREATMENT_U_COEFS[:K].reshape(1, -1)
    )
    logits[:, 0] = 0.0
    return logits


class ContinuousDGP:

    def __init__(self, samp_size: int, add_noise=True, seed=None, K=4):
        self.add_noise = add_noise
        self.K = K
        if seed is None:
            self.seed = [None] * 6
        else:
            self.seed = randint.rvs(0, 4294967295, size=6, random_state=seed)
        self.X = multivariate_normal.rvs(
            mean=CONTINUOUS_X_MEAN,
            cov=CONTINUOUS_X_COV,
            size=samp_size,
            random_state=self.seed[0],
        )
        self.X_irre = uniform.rvs(
            loc=0.0, scale=1.0, size=[samp_size, 7], random_state=self.seed[2]
        )
        rng_u = np.random.default_rng(self.seed[3])
        self.U = rng_u.normal(0.0, 1.0, samp_size)
        logits = continuous_treatment_logits(self.X, self.U, self.K)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        rng_a = np.random.default_rng(self.seed[1])
        self.A = np.array([rng_a.choice(self.K, p=probs[i]) for i in range(samp_size)])
        self.w_intercept = CONTINUOUS_PROXY_PARAMS["w_intercept"]
        self.w_x0_coef = CONTINUOUS_PROXY_PARAMS["w_x0_coef"]
        self.w_u_coef = CONTINUOUS_PROXY_PARAMS["w_u_coef"]
        self.w_noise_sd = CONTINUOUS_PROXY_PARAMS["w_noise_sd"]
        self.z_intercept = CONTINUOUS_PROXY_PARAMS["z_intercept"]
        self.z_x1_coef = CONTINUOUS_PROXY_PARAMS["z_x1_coef"]
        self.z_u_coef = CONTINUOUS_PROXY_PARAMS["z_u_coef"]
        self.z_noise_sd = CONTINUOUS_PROXY_PARAMS["z_noise_sd"]
        rng_proxy = np.random.default_rng(self.seed[4])
        noise_W = rng_proxy.normal(0.0, self.w_noise_sd, samp_size)
        noise_Z = rng_proxy.normal(0.0, self.z_noise_sd, samp_size)
        self.W = (
            self.w_intercept
            + self.w_x0_coef * self.X[:, 0]
            + self.w_u_coef * self.U
            + noise_W
        )
        self.Z = (
            self.z_intercept
            + self.z_x1_coef * self.X[:, 1]
            + self.z_u_coef * self.U
            + noise_Z
        )

    def continuous_delta(self, U_value=None, linearity=None, U_inter=None):
        linearity = (
            getattr(self, "linearity", "linear") if linearity is None else linearity
        )
        U_inter = getattr(self, "U_inter", True) if U_inter is None else U_inter
        (x0, x1, x2) = (self.X[:, 0], self.X[:, 1], self.X[:, 2])
        if U_value is None:
            u = self.U
        else:
            u = np.asarray(U_value).reshape(-1)
            if u.shape[0] != self.X.shape[0]:
                raise ValueError("U_value must have the same length as X")
        delta = np.zeros((self.X.shape[0], self.K))
        if linearity == "linear":
            if U_inter:
                delta[:, 0] = 0.12 + 0.62 * x0 - 0.32 * x1 + 0.12 * x2 + 0.85 * u
                delta[:, 1] = -0.1 - 0.46 * x0 + 0.66 * x1 + 0.12 * x2 - 0.8 * u
                delta[:, 2] = 0.04 + 0.22 * x0 + 0.12 * x1 - 0.62 * x2 + 0.55 * u
                delta[:, 3] = -0.06 - 0.4 * x0 - 0.24 * x1 + 0.58 * x2 - 0.6 * u
            else:
                delta[:, 0] = 0.15 + 1.0 * x0 - 0.55 * x1 + 0.25 * x2
                delta[:, 1] = -0.1 - 0.6 * x0 + 0.95 * x1 + 0.15 * x2
                delta[:, 2] = 0.05 + 0.45 * x0 + 0.25 * x1 - 0.95 * x2
                delta[:, 3] = -0.1 - 0.7 * x0 - 0.45 * x1 + 0.85 * x2
        elif linearity == "nonlinear":
            if U_inter:
                delta[:, 0] = 0.75 * np.sin(x0) - 0.55 * x1 + 0.2 * x2 + 0.55 * u
                delta[:, 1] = -0.45 * x0 + 0.8 * np.tanh(x1) + 0.3 * x2 - 0.5 * u
                delta[:, 2] = 0.35 * x0 - 0.25 * x1 - 0.7 * np.sin(x2) + 0.4 * u
                delta[:, 3] = -0.55 * np.tanh(x0 + x2) - 0.3 * x1 + 0.65 * x2 - 0.45 * u
            else:
                delta[:, 0] = 0.85 * np.sin(x0) - 0.6 * x1 + 0.25 * x2
                delta[:, 1] = -0.5 * x0 + 0.9 * np.tanh(x1) + 0.25 * x2
                delta[:, 2] = 0.45 * x0 - 0.25 * x1 - 0.8 * np.sin(x2)
                delta[:, 3] = -0.65 * np.tanh(x0 + x2) - 0.35 * x1 + 0.75 * x2
        else:
            raise ValueError("linearity must be 'linear' or 'nonlinear'")
        delta -= delta.mean(axis=1, keepdims=True)
        return 1.15 * delta

    def expected_U_given_XW(self):
        residual_w = self.W - self.w_intercept - self.w_x0_coef * self.X[:, 0]
        return self.w_u_coef / (self.w_u_coef**2 + self.w_noise_sd**2) * residual_w

    def expected_U_given_XZ(self):
        residual_z = self.Z - self.z_intercept - self.z_x1_coef * self.X[:, 1]
        return self.z_u_coef / (self.z_u_coef**2 + self.z_noise_sd**2) * residual_z

    def expected_U_given_XWZ(self):
        residual_w = self.W - self.w_intercept - self.w_x0_coef * self.X[:, 0]
        residual_z = self.Z - self.z_intercept - self.z_x1_coef * self.X[:, 1]
        precision = (
            1.0
            + self.w_u_coef**2 / self.w_noise_sd**2
            + self.z_u_coef**2 / self.z_noise_sd**2
        )
        return (
            self.w_u_coef * residual_w / self.w_noise_sd**2
            + self.z_u_coef * residual_z / self.z_noise_sd**2
        ) / precision

    def compute_oracle_xz(self, return_delta=False):
        if not hasattr(self, "delta_all"):
            raise AttributeError(
                "Invalid preprocessing input: use fitted training transformations for prediction."
            )
        u_xz = self.expected_U_given_XZ()
        delta_xz = self.continuous_delta(U_value=u_xz)
        oracle_xz = np.argmax(delta_xz, axis=1)
        if return_delta:
            return (oracle_xz, delta_xz)
        return oracle_xz

    def compute_oracle_xwz(self, return_delta=False):
        if not hasattr(self, "delta_all"):
            raise AttributeError(
                "Invalid preprocessing input: use fitted training transformations for prediction."
            )
        u_xwz = self.expected_U_given_XWZ()
        delta_xwz = self.continuous_delta(U_value=u_xwz)
        oracle_xwz = np.argmax(delta_xwz, axis=1)
        if return_delta:
            return (oracle_xwz, delta_xwz)
        return oracle_xwz

    def compute_oracle_regimes(self, return_delta=False):
        if not hasattr(self, "delta_all"):
            raise AttributeError(
                "Invalid preprocessing input: use fitted training transformations for prediction."
            )
        u_x = np.zeros(self.X.shape[0])
        u_xw = self.expected_U_given_XW()
        delta_x = self.continuous_delta(U_value=u_x)
        delta_xw = self.continuous_delta(U_value=u_xw)
        delta_full = self.delta_all
        oracle_x = np.argmax(delta_x, axis=1)
        oracle_xw = np.argmax(delta_xw, axis=1)
        oracle_full = np.argmax(delta_full, axis=1)
        if return_delta:
            return (oracle_x, oracle_xw, oracle_full, delta_x, delta_xw, delta_full)
        return (oracle_x, oracle_xw, oracle_full)

    def continuous_main_effect(self, linearity=None, U_inter=None):
        linearity = (
            getattr(self, "linearity", "linear") if linearity is None else linearity
        )
        U_inter = getattr(self, "U_inter", True) if U_inter is None else U_inter
        (x0, x1, x2) = (self.X[:, 0], self.X[:, 1], self.X[:, 2])
        z0 = self.X_irre[:, 0]
        z1 = self.X_irre[:, 1]
        z2 = self.X_irre[:, 2]
        z3 = self.X_irre[:, 3]
        z4 = self.X_irre[:, 4]
        z5 = self.X_irre[:, 5]
        z6 = self.X_irre[:, 6]
        (c0, c1, c2) = (z0 - 0.5, z1 - 0.5, z2 - 0.5)
        (c3, c4, c5, c6) = (z3 - 0.5, z4 - 0.5, z5 - 0.5, z6 - 0.5)
        if linearity == "linear":
            return (
                2.0
                + 0.25 * x0
                - 0.2 * x1
                + 0.15 * x2
                + 0.25 * self.W
                + 2.0 * self.U
                + 0.18 * c0
                - 0.15 * c1
                + 0.12 * c2
                - 0.1 * c3
                + 0.08 * c4
                - 0.06 * c5
                + 0.05 * c6
            )
        elif linearity == "nonlinear":
            return (
                2.0
                + 0.25 * x0
                - 0.2 * x1
                + 0.15 * x2
                + 0.8 * self.W
                + 1.7 * self.U
                + 0.5 * (self.U**2 - 1.0)
                + 0.45 * np.sin(x0 * x1)
                + 0.3 * np.cos(x2)
                + 0.2 * np.tanh(x0 + x2)
                + 0.15 * x0 * x2
                + 0.18 * c0
                - 0.15 * c1
                + 0.12 * c2
                - 0.1 * c3
                + 0.08 * c4
                - 0.06 * c5
                + 0.05 * c6
                + 0.12 * np.sin(2.0 * np.pi * z0)
                + 0.1 * np.cos(2.0 * np.pi * z1)
                + 0.08 * c2 * c3
                - 0.06 * c4 * c5
                + 0.05 * (c6**2 - 1.0 / 12.0)
            )
        else:
            raise ValueError("linearity must be 'linear' or 'nonlinear'")

    def gen_Y(self, linearity="linear", U_inter=True):
        self.linearity = linearity
        self.U_inter = U_inter
        n = self.A.shape[0]
        self.delta_all = np.zeros((n, self.K))
        self.delta_all = self.continuous_delta(linearity=linearity, U_inter=U_inter)
        main_effect = self.continuous_main_effect(linearity=linearity, U_inter=U_inter)
        self.GOR = np.argmax(self.delta_all, axis=1)
        self.Y_all = main_effect.reshape(-1, 1) + self.delta_all
        self.Y = self.Y_all[np.arange(n), self.A]
        if self.add_noise:
            half_width = CONTINUOUS_OUTCOME_NOISE_SCALE / 2.0
            self.Y += uniform.rvs(
                loc=-half_width,
                scale=CONTINUOUS_OUTCOME_NOISE_SCALE,
                size=n,
                random_state=self.seed[5],
            )
        return self.get_dataframe()

    def get_dataframe(self):
        df = pd.DataFrame()
        df["A"] = self.A
        for i in range(self.X.shape[1]):
            df[f"X{i}"] = self.X[:, i]
        for i in range(self.X_irre.shape[1]):
            df[f"X_irre{i}"] = self.X_irre[:, i]
        df["Z"] = self.Z
        df["W"] = self.W
        df["U"] = self.U
        df["Y"] = self.Y
        df["GOR"] = self.GOR
        return df


class SurvivalCovariates:

    def __init__(self, samp_size: int, add_noise=True, seed=None, K=4):
        self.add_noise = add_noise
        self.K = K
        if seed is None:
            self.seed = [None] * 8
        else:
            self.seed = randint.rvs(0, 4294967295, size=8, random_state=seed)
        self.X = multivariate_normal.rvs(
            mean=CONTINUOUS_X_MEAN,
            cov=CONTINUOUS_X_COV,
            size=samp_size,
            random_state=self.seed[0],
        )
        self.X_irre = uniform.rvs(
            loc=0.0, scale=1.0, size=(samp_size, 7), random_state=self.seed[2]
        )
        rng_u = np.random.default_rng(self.seed[3])
        self.U = rng_u.normal(0.0, 1.0, samp_size)
        logits = continuous_treatment_logits(self.X, self.U, self.K)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        self.treatment_prob_all = probs
        rng_a = np.random.default_rng(self.seed[1])
        self.A = np.array([rng_a.choice(self.K, p=probs[i]) for i in range(samp_size)])
        self.w_intercept = CONTINUOUS_PROXY_PARAMS["w_intercept"]
        self.w_x0_coef = CONTINUOUS_PROXY_PARAMS["w_x0_coef"]
        self.w_u_coef = CONTINUOUS_PROXY_PARAMS["w_u_coef"]
        self.w_noise_sd = CONTINUOUS_PROXY_PARAMS["w_noise_sd"]
        self.z_intercept = CONTINUOUS_PROXY_PARAMS["z_intercept"]
        self.z_x1_coef = CONTINUOUS_PROXY_PARAMS["z_x1_coef"]
        self.z_u_coef = CONTINUOUS_PROXY_PARAMS["z_u_coef"]
        self.z_noise_sd = CONTINUOUS_PROXY_PARAMS["z_noise_sd"]
        rng_proxy = np.random.default_rng(self.seed[4])
        noise_W = rng_proxy.normal(0.0, self.w_noise_sd, samp_size)
        noise_Z = rng_proxy.normal(0.0, self.z_noise_sd, samp_size)
        self.W = (
            self.w_intercept
            + self.w_x0_coef * self.X[:, 0]
            + self.w_u_coef * self.U
            + noise_W
        )
        self.Z = (
            self.z_intercept
            + self.z_x1_coef * self.X[:, 1]
            + self.z_u_coef * self.U
            + noise_Z
        )

    def expected_U_given_XW(self):
        residual_w = self.W - self.w_intercept - self.w_x0_coef * self.X[:, 0]
        return self.w_u_coef / (self.w_u_coef**2 + self.w_noise_sd**2) * residual_w

    def expected_U_given_XZ(self):
        residual_z = self.Z - self.z_intercept - self.z_x1_coef * self.X[:, 1]
        return self.z_u_coef / (self.z_u_coef**2 + self.z_noise_sd**2) * residual_z

    def expected_U_given_XWZ(self):
        residual_w = self.W - self.w_intercept - self.w_x0_coef * self.X[:, 0]
        residual_z = self.Z - self.z_intercept - self.z_x1_coef * self.X[:, 1]
        precision = (
            1.0
            + self.w_u_coef**2 / self.w_noise_sd**2
            + self.z_u_coef**2 / self.z_noise_sd**2
        )
        return (
            self.w_u_coef * residual_w / self.w_noise_sd**2
            + self.z_u_coef * residual_z / self.z_noise_sd**2
        ) / precision
