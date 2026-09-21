import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.preprocessing import RobustScaler as Scaler


class BlockDimensionNormalizedRobustScaler(BaseEstimator):

    def __init__(self, block_sizes, block_distance_weights=None):
        self.block_sizes = block_sizes
        self.block_distance_weights = block_distance_weights

    @staticmethod
    def _validate_block_sizes(block_sizes, n_features):
        if isinstance(block_sizes, (str, bytes)):
            raise TypeError("block_sizes must be a nonempty sequence of integers")
        try:
            values = tuple(block_sizes)
        except TypeError as exc:
            raise TypeError(
                "block_sizes must be a nonempty sequence of integers"
            ) from exc
        if not values:
            raise ValueError("block_sizes must contain at least one block")
        normalized = []
        for value in values:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError("Every block size must be a positive integer")
            value = int(value)
            if value <= 0:
                raise ValueError("Every block size must be a positive integer")
            normalized.append(value)
        normalized = tuple(normalized)
        if sum(normalized) != int(n_features):
            raise ValueError(
                "block_sizes must sum to the transformed feature dimension"
            )
        return normalized

    @staticmethod
    def _validate_block_distance_weights(block_distance_weights, n_blocks):
        if block_distance_weights is None:
            return tuple((1.0 for _ in range(int(n_blocks))))
        if isinstance(block_distance_weights, (str, bytes)):
            raise TypeError(
                "block_distance_weights must be a sequence of positive numbers"
            )
        try:
            values = tuple(block_distance_weights)
        except TypeError as exc:
            raise TypeError(
                "block_distance_weights must be a sequence of positive numbers"
            ) from exc
        if len(values) != int(n_blocks):
            raise ValueError("block_distance_weights must have one value per block")
        normalized = []
        for value in values:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError("Every block distance weight must be a positive number")
            value = float(value)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    "Every block distance weight must be finite and positive"
                )
            normalized.append(value)
        return tuple(normalized)

    def fit(self, X, y=None):
        self.robust_scaler_ = Scaler()
        self.robust_scaler_.fit(X, y)
        self.n_features_in_ = int(self.robust_scaler_.n_features_in_)
        self.block_sizes_ = self._validate_block_sizes(
            self.block_sizes, self.n_features_in_
        )
        self.block_distance_weights_ = self._validate_block_distance_weights(
            self.block_distance_weights, len(self.block_sizes_)
        )
        self.block_weights_ = np.concatenate(
            [
                np.full(
                    size,
                    np.sqrt(float(distance_weight)) / np.sqrt(float(size)),
                    dtype=float,
                )
                for (size, distance_weight) in zip(
                    self.block_sizes_, self.block_distance_weights_
                )
            ]
        )
        return self

    def transform(self, X):
        check_is_fitted(
            self,
            attributes=(
                "robust_scaler_",
                "block_sizes_",
                "block_distance_weights_",
                "block_weights_",
            ),
        )
        scaled = self.robust_scaler_.transform(X)
        return scaled * self.block_weights_

    def fit_transform(self, X, y=None, **fit_params):
        if fit_params:
            raise TypeError("Unexpected fit parameters for block scaler")
        return self.fit(X, y).transform(X)


def _check_auto(param):
    return isinstance(param, str) and param == "auto"


class _BaseRKHSIV:

    def __init__(self, *args, **kwargs):
        return

    def _get_delta(self, n):
        delta_scale = 5 if _check_auto(self.delta_scale) else self.delta_scale
        delta_exp = 0.4 if _check_auto(self.delta_exp) else self.delta_exp
        return delta_scale / n**delta_exp

    def _get_alpha_scale(self):
        return 60 if _check_auto(self.alpha_scale) else self.alpha_scale

    def _get_alpha_scales(self):
        return (
            [c for c in np.geomspace(0.1, 100000.0, self.n_alphas)]
            if _check_auto(self.alpha_scales)
            else self.alpha_scales
        )

    def _get_alpha(self, delta, alpha_scale):
        return alpha_scale * delta**4

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)

    def _get_gamma_f(self, condition):
        if _check_auto(self.gamma_f):
            params = {"squared": True}
            K_condition_euclidean = sklearn.metrics.pairwise_distances(
                X=condition, metric="euclidean", n_jobs=-1, **params
            )
            gamma_f = 1.0 / np.median(
                K_condition_euclidean[np.tril_indices(condition.shape[0], -1)]
            )
            return gamma_f
        else:
            return self.gamma_f

    def _get_kernel_f(self, X, Y=None, gamma_f=0.1):
        params = {"gamma": gamma_f}
        return pairwise_kernels(X, Y, metric="rbf", filter_params=True, **params)

    def _get_kernel_h(self, X, Y=None, gamma_h=0.01):
        params = {"gamma": gamma_h}
        return pairwise_kernels(X, Y, metric="rbf", filter_params=True, **params)


def _make_bridge_scaler(block_sizes, block_distance_weights):
    if block_sizes is None:
        return Scaler()
    return BlockDimensionNormalizedRobustScaler(
        block_sizes, block_distance_weights=block_distance_weights
    )


def _fit_honest_approx_rkhs_cv(estimator, X, y, condition, index=None):
    (X, y) = check_X_y(X, y, accept_sparse=True)
    (condition, y) = check_X_y(condition, y, accept_sparse=True)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = X.shape[0]
    if condition.shape[0] != n:
        raise ValueError("X and condition must have the same number of rows")
    is_q = index is not None
    if is_q:
        index = np.asarray(index, dtype=bool).reshape(-1)
        if len(index) != n or np.sum(index) < 2:
            raise ValueError("q bridge index must contain at least two effective rows")
    configured_components = int(estimator.n_components)
    alpha_scales = np.asarray(estimator._get_alpha_scales(), dtype=float)
    estimator.resolved_alpha_scales_ = alpha_scales.copy()
    if is_q:
        splitter = StratifiedKFold(
            n_splits=int(estimator.cv),
            shuffle=True,
            random_state=int(estimator.nystrom_random_state),
        )
        raw_splits = splitter.split(X, index.astype(int))
    else:
        splitter = KFold(
            n_splits=int(estimator.cv),
            shuffle=True,
            random_state=int(estimator.nystrom_random_state),
        )
        raw_splits = splitter.split(X)
    fold_scores = []
    expected_gamma_count = None
    for fold_id, (train, validation) in enumerate(raw_splits):
        condition_scaler = _make_bridge_scaler(
            estimator.condition_block_sizes, estimator.condition_block_distance_weights
        )
        condition_train = condition_scaler.fit_transform(condition[train])
        condition_validation = condition_scaler.transform(condition[validation])
        gamma_f_fold = estimator._get_gamma_f(condition_train)
        fold_components = int(min(configured_components, max(2, len(train) - 1)))
        original_components = estimator.n_components
        estimator.n_components = fold_components
        try:
            condition_map = estimator._get_new_approx_instance(gamma=gamma_f_fold)
        finally:
            estimator.n_components = original_components
        root_f_train = condition_map.fit_transform(condition_train)
        root_f_validation = condition_map.transform(condition_validation)
        input_scaler = _make_bridge_scaler(
            estimator.input_block_sizes, estimator.input_block_distance_weights
        )
        input_train = input_scaler.fit_transform(X[train])
        input_validation = input_scaler.transform(X[validation])
        gamma_hs_fold = np.atleast_1d(
            np.asarray(estimator._get_gamma_hs(input_train), dtype=float)
        )
        if expected_gamma_count is None:
            expected_gamma_count = len(gamma_hs_fold)
        elif len(gamma_hs_fold) != expected_gamma_count:
            raise ValueError("gamma_h candidate count changed across bridge-CV folds")
        (n_train, n_validation) = (len(train), len(validation))
        effective_train = int(np.sum(index[train])) if is_q else n_train
        effective_validation = int(np.sum(index[validation])) if is_q else n_validation
        delta_train = estimator._get_delta(max(effective_train, 2))
        delta_validation = estimator._get_delta(max(effective_validation, 2))
        system_train = (
            root_f_train.T @ root_f_train / (2 * n_train * delta_train**2)
            + np.eye(root_f_train.shape[1]) / 2
        )
        system_validation = (
            root_f_validation.T
            @ root_f_validation
            / (2 * n_validation * delta_validation**2)
            + np.eye(root_f_validation.shape[1]) / 2
        )
        q_train = np.linalg.pinv(system_train)
        q_validation = np.linalg.pinv(system_validation)
        gamma_scores = []
        for gamma_h_fold in gamma_hs_fold:
            original_components = estimator.n_components
            estimator.n_components = fold_components
            try:
                input_map = estimator._get_new_approx_instance(
                    gamma=float(gamma_h_fold)
                )
            finally:
                estimator.n_components = original_components
            root_h_train = input_map.fit_transform(input_train)
            root_h_validation = input_map.transform(input_validation)
            if is_q:
                root_h_train = root_h_train.copy()
                root_h_validation = root_h_validation.copy()
                root_h_train[~index[train], :] = 0.0
                root_h_validation[~index[validation], :] = 0.0
            a_train = root_h_train.T @ root_f_train
            aqa_train = a_train @ q_train @ a_train.T
            b_train = a_train @ q_train @ root_f_train.T @ y[train]
            alpha_scores = []
            for alpha_scale in alpha_scales:
                alpha = estimator._get_alpha(delta_train, float(alpha_scale))
                coefficients = np.linalg.lstsq(
                    aqa_train + alpha * np.eye(root_h_train.shape[1]),
                    b_train,
                    rcond=None,
                )[0]
                residual_moment = root_f_validation.T @ (
                    y[validation] - root_h_validation @ coefficients
                )
                score = (residual_moment.T @ q_validation @ residual_moment).reshape(
                    -1
                )[0] / n_validation**2
                alpha_scores.append(float(score))
            gamma_scores.append(alpha_scores)
        fold_scores.append(gamma_scores)
    score_array = np.asarray(fold_scores, dtype=float)
    mean_scores = np.mean(score_array, axis=0)
    (best_gamma_index, best_alpha_index) = np.unravel_index(
        np.argmin(mean_scores), mean_scores.shape
    )
    estimator.best_gamma_index_ = int(best_gamma_index)
    estimator.best_alpha_index_ = int(best_alpha_index)
    estimator.best_alpha_scale = float(alpha_scales[best_alpha_index])
    estimator.transCondition = _make_bridge_scaler(
        estimator.condition_block_sizes, estimator.condition_block_distance_weights
    )
    condition_full = estimator.transCondition.fit_transform(condition)
    estimator.gamma_f = estimator._get_gamma_f(condition_full)
    estimator.featCond = estimator._get_new_approx_instance(gamma=estimator.gamma_f)
    root_f = estimator.featCond.fit_transform(condition_full)
    estimator.transX = _make_bridge_scaler(
        estimator.input_block_sizes, estimator.input_block_distance_weights
    )
    input_full = estimator.transX.fit_transform(X)
    gamma_hs_full = np.atleast_1d(
        np.asarray(estimator._get_gamma_hs(input_full), dtype=float)
    )
    estimator.resolved_gamma_hs_ = gamma_hs_full.copy()
    estimator.gamma_h = float(gamma_hs_full[best_gamma_index])
    estimator.featX = estimator._get_new_approx_instance(gamma=estimator.gamma_h)
    root_h = estimator.featX.fit_transform(input_full)
    if is_q:
        root_h = root_h.copy()
        root_h[~index, :] = 0.0
    effective_n = int(np.sum(index)) if is_q else n
    delta = estimator._get_delta(max(effective_n, 2))
    estimator.best_alpha = estimator._get_alpha(delta, estimator.best_alpha_scale)
    condition_system = (
        root_f.T @ root_f / (2 * n * delta**2) + np.eye(root_f.shape[1]) / 2
    )
    q_full = np.linalg.pinv(condition_system)
    a_full = root_h.T @ root_f
    regularized_system = a_full @ q_full @ a_full.T + estimator.best_alpha * np.eye(
        root_h.shape[1]
    )
    right_hand_side = a_full @ q_full @ root_f.T @ y
    estimator.a = np.linalg.lstsq(regularized_system, right_hand_side, rcond=None)[0]
    estimator.fitted_delta = delta
    return estimator


class ApproxRKHSIV(_BaseRKHSIV):

    def __init__(
        self,
        kernel_approx="nystrom",
        n_components=25,
        gamma_f="auto",
        gamma_h=0.1,
        delta_scale="auto",
        delta_exp="auto",
        alpha_scale="auto",
    ):
        self.kernel_approx = kernel_approx
        self.n_components = n_components
        self.gamma_f = gamma_f
        self.gamma_h = gamma_h
        self.delta_scale = delta_scale
        self.delta_exp = delta_exp
        self.alpha_scale = alpha_scale

    def _get_new_approx_instance(self, gamma):
        if self.kernel_approx == "rbfsampler":
            return RBFSampler(
                gamma=gamma, n_components=self.n_components, random_state=1
            )
        elif self.kernel_approx == "nystrom":
            return Nystroem(
                kernel="rbf",
                gamma=gamma,
                random_state=getattr(self, "nystrom_random_state", 1),
                n_components=self.n_components,
            )
        else:
            raise AttributeError("Invalid kernel approximator")

    def fit(self, X, y, condition):
        (X, y) = check_X_y(X, y, accept_sparse=True)
        (condition, y) = check_X_y(condition, y, accept_sparse=True)
        condition = Scaler().fit_transform(condition)
        gamma_f = self._get_gamma_f(condition=condition)
        self.gamma_f = gamma_f
        self.featCond = self._get_new_approx_instance(gamma=self.gamma_f)
        RootKf = self.featCond.fit_transform(condition)
        self.transX = Scaler()
        self.transX.fit(X)
        X = self.transX.transform(X)
        self.featX = self._get_new_approx_instance(gamma=self.gamma_h)
        RootKh = self.featX.fit_transform(X)
        n = X.shape[0]
        delta = self._get_delta(n)
        alpha = self._get_alpha(delta, self._get_alpha_scale())
        Q = np.linalg.pinv(
            RootKf.T @ RootKf / (2 * n * delta**2) + np.eye(self.n_components) / 2
        )
        A = RootKh.T @ RootKf
        W = A @ Q @ A.T + alpha * np.eye(self.n_components)
        B = A @ Q @ RootKf.T @ y
        self.a = np.linalg.lstsq(W, B, rcond=None)[0]
        self.fitted_delta = delta
        return self

    def predict(self, X):
        X = self.transX.transform(X)
        return self.featX.transform(X) @ self.a


class ApproxRKHSIVCV(ApproxRKHSIV):

    def __init__(
        self,
        kernel_approx="nystrom",
        n_components=25,
        gamma_f="auto",
        gamma_hs="auto",
        n_gamma_hs=10,
        delta_scale="auto",
        delta_exp="auto",
        alpha_scales="auto",
        n_alphas=30,
        cv=6,
        nystrom_random_state=1,
        input_block_sizes=None,
        condition_block_sizes=None,
        input_block_distance_weights=None,
        condition_block_distance_weights=None,
    ):
        self.kernel_approx = kernel_approx
        self.n_components = n_components
        self.gamma_f = gamma_f
        self.gamma_hs = gamma_hs
        self.n_gamma_hs = n_gamma_hs
        self.delta_scale = delta_scale
        self.delta_exp = delta_exp
        self.alpha_scales = alpha_scales
        self.n_alphas = n_alphas
        self.cv = cv
        self.nystrom_random_state = nystrom_random_state
        self.input_block_sizes = input_block_sizes
        self.condition_block_sizes = condition_block_sizes
        self.input_block_distance_weights = input_block_distance_weights
        self.condition_block_distance_weights = condition_block_distance_weights

    def _get_gamma_hs(self, X):
        if _check_auto(self.gamma_hs):
            params = {"squared": True}
            K_X_euclidean = sklearn.metrics.pairwise_distances(
                X=X, metric="euclidean", **params
            )
            return 1.0 / np.quantile(
                K_X_euclidean[np.tril_indices(X.shape[0], -1)],
                np.array(range(1, self.n_gamma_hs)) / self.n_gamma_hs,
            )
        else:
            return self.gamma_hs

    def fit(self, X, y, condition):
        if (
            self.input_block_distance_weights is not None
            and self.input_block_sizes is None
        ):
            raise ValueError("input_block_distance_weights require input_block_sizes")
        if (
            self.condition_block_distance_weights is not None
            and self.condition_block_sizes is None
        ):
            raise ValueError(
                "condition_block_distance_weights require condition_block_sizes"
            )
        return _fit_honest_approx_rkhs_cv(self, X, y, condition)


class ApproxRKHSIV_q(_BaseRKHSIV):

    def __init__(
        self,
        kernel_approx="nystrom",
        n_components=20,
        gamma_f="auto",
        gamma_h=0.1,
        delta_scale="auto",
        delta_exp="auto",
        alpha_scale="auto",
    ):
        self.kernel_approx = kernel_approx
        self.n_components = n_components
        self.gamma_f = gamma_f
        self.gamma_h = gamma_h
        self.delta_scale = delta_scale
        self.delta_exp = delta_exp
        self.alpha_scale = alpha_scale

    def _get_new_approx_instance(self, gamma):
        if self.kernel_approx == "rbfsampler":
            return RBFSampler(
                gamma=gamma, n_components=self.n_components, random_state=1
            )
        elif self.kernel_approx == "nystrom":
            return Nystroem(
                kernel="rbf",
                gamma=gamma,
                random_state=getattr(self, "nystrom_random_state", 1),
                n_components=self.n_components,
            )
        else:
            raise AttributeError("Invalid kernel approximator")

    def fit(self, X, y, condition, index):
        (X, y) = check_X_y(X, y, accept_sparse=True)
        (condition, y) = check_X_y(condition, y, accept_sparse=True)
        condition = Scaler().fit_transform(condition)
        gamma_f = self._get_gamma_f(condition=condition)
        self.gamma_f = gamma_f
        self.featCond = self._get_new_approx_instance(gamma=self.gamma_f)
        RootKf = self.featCond.fit_transform(condition)
        self.transX = Scaler()
        self.transX.fit(X)
        X = self.transX.transform(X)
        self.featX = self._get_new_approx_instance(gamma=self.gamma_h)
        RootKh = self.featX.fit_transform(X)
        RootKh[np.logical_not(index), :] = 0
        n = X.shape[0]
        delta = self._get_delta(np.sum(index))
        alpha = self._get_alpha(delta, self._get_alpha_scale())
        Q = np.linalg.pinv(
            RootKf.T @ RootKf / (2 * n * delta**2) + np.eye(self.n_components) / 2
        )
        A = RootKh.T @ RootKf
        W = A @ Q @ A.T + alpha * np.eye(self.n_components)
        B = A @ Q @ RootKf.T @ y
        self.a = np.linalg.lstsq(W, B, rcond=None)[0]
        self.fitted_delta = delta
        return self

    def predict(self, X):
        X = self.transX.transform(X)
        return self.featX.transform(X) @ self.a


class ApproxRKHSIVCV_q(ApproxRKHSIV_q):

    def __init__(
        self,
        kernel_approx="nystrom",
        n_components=25,
        gamma_f="auto",
        gamma_hs="auto",
        n_gamma_hs=10,
        delta_scale="auto",
        delta_exp="auto",
        alpha_scales="auto",
        n_alphas=30,
        cv=6,
        nystrom_random_state=1,
        input_block_sizes=None,
        condition_block_sizes=None,
        input_block_distance_weights=None,
        condition_block_distance_weights=None,
    ):
        self.kernel_approx = kernel_approx
        self.n_components = n_components
        self.gamma_f = gamma_f
        self.gamma_hs = gamma_hs
        self.n_gamma_hs = n_gamma_hs
        self.delta_scale = delta_scale
        self.delta_exp = delta_exp
        self.alpha_scales = alpha_scales
        self.n_alphas = n_alphas
        self.cv = cv
        self.nystrom_random_state = nystrom_random_state
        self.input_block_sizes = input_block_sizes
        self.condition_block_sizes = condition_block_sizes
        self.input_block_distance_weights = input_block_distance_weights
        self.condition_block_distance_weights = condition_block_distance_weights

    def _get_gamma_hs(self, X):
        if _check_auto(self.gamma_hs):
            params = {"squared": True}
            K_X_euclidean = sklearn.metrics.pairwise_distances(
                X=X, metric="euclidean", **params
            )
            return 1.0 / np.quantile(
                K_X_euclidean[np.tril_indices(X.shape[0], -1)],
                np.array(range(1, self.n_gamma_hs)) / self.n_gamma_hs,
            )
        else:
            return self.gamma_hs

    def fit(self, X, y, condition, index):
        if (
            self.input_block_distance_weights is not None
            and self.input_block_sizes is None
        ):
            raise ValueError("input_block_distance_weights require input_block_sizes")
        if (
            self.condition_block_distance_weights is not None
            and self.condition_block_sizes is None
        ):
            raise ValueError(
                "condition_block_distance_weights require condition_block_sizes"
            )
        return _fit_honest_approx_rkhs_cv(self, X, y, condition, index=index)
