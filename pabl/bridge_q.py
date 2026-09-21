import numpy as np
from pabl.kernel import ApproxRKHSIVCV_q
from pabl.splits import resolve_crossfit_splits


class MultiTreatmentProxQ:

    def __init__(self, A, X, Z, W):
        self.A = np.asarray(A).reshape(-1)
        self.X = self._ensure_2d(X)
        self.Z = self._ensure_2d(Z)
        self.W = self._ensure_2d(W)
        n = self.A.shape[0]
        if self.X.shape[0] != n or self.Z.shape[0] != n or self.W.shape[0] != n:
            raise ValueError("A, X, Z, W must have the same number of observations")
        self.classes_ = np.unique(self.A)
        self.class_to_col_ = {k: idx for (idx, k) in enumerate(self.classes_)}
        self.q_models_ = {}
        self.q_estimators_ = {}
        self.q_params_ = {}
        self.q_est = np.zeros(n, dtype=float)
        self.q_est_oof_ = None
        self.fold_models_ = []
        self.fold_indices_ = []

    @staticmethod
    def _ensure_2d(arr):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        if arr.ndim == 2:
            return arr
        raise ValueError("Input arrays must be 1D or 2D")

    def _fit_one_model(
        self,
        A,
        X,
        Z,
        W,
        gamma_f="auto",
        n_gamma_hs=20,
        n_alphas=30,
        alpha_scales="auto",
        n_components=50,
        cv=5,
        gamma_hs="auto",
        nystrom_random_state=1,
        input_block_sizes=None,
        condition_block_sizes=None,
        input_block_distance_weights=None,
        condition_block_distance_weights=None,
    ):
        model = self.__class__(A=A, X=X, Z=Z, W=W)
        model.fit_cv(
            gamma_f=gamma_f,
            n_gamma_hs=n_gamma_hs,
            n_alphas=n_alphas,
            alpha_scales=alpha_scales,
            n_components=n_components,
            cv=cv,
            gamma_hs=gamma_hs,
            nystrom_random_state=nystrom_random_state,
            input_block_sizes=input_block_sizes,
            condition_block_sizes=condition_block_sizes,
            input_block_distance_weights=input_block_distance_weights,
            condition_block_distance_weights=condition_block_distance_weights,
        )
        return model

    def fit_cv(
        self,
        gamma_f="auto",
        n_gamma_hs=20,
        n_alphas=30,
        alpha_scales="auto",
        n_components=50,
        cv=5,
        gamma_hs="auto",
        nystrom_random_state=1,
        input_block_sizes=None,
        condition_block_sizes=None,
        input_block_distance_weights=None,
        condition_block_distance_weights=None,
    ):
        input_ZX = np.concatenate((self.Z, self.X), axis=1)
        cond_XW = np.concatenate((self.X, self.W), axis=1)
        target_Y = np.ones(self.A.shape[0], dtype=float)
        self.q_estimators_ = {}
        for k in self.classes_:
            index = self.A == k
            if np.sum(index) == 0:
                raise ValueError(f"Treatment arm {k} has no observations")
            model_cv = ApproxRKHSIVCV_q(
                gamma_f=gamma_f,
                gamma_hs=gamma_hs,
                n_gamma_hs=n_gamma_hs,
                n_alphas=n_alphas,
                alpha_scales=alpha_scales,
                cv=cv,
                n_components=n_components,
                nystrom_random_state=nystrom_random_state,
                input_block_sizes=input_block_sizes,
                condition_block_sizes=condition_block_sizes,
                input_block_distance_weights=input_block_distance_weights,
                condition_block_distance_weights=condition_block_distance_weights,
            )
            model_cv.fit(X=input_ZX, y=target_Y, condition=cond_XW, index=index)
            self.q_params_[k] = {
                "gamma_f": model_cv.gamma_f,
                "gamma_h": model_cv.gamma_h,
                "alpha_scale": model_cv.best_alpha_scale,
                "delta": getattr(model_cv, "fitted_delta", None),
            }
            self.q_estimators_[k] = model_cv
            self.q_models_[k] = model_cv.predict
            self.q_est[index] = model_cv.predict(input_ZX[index, :])
        return self

    def fit_cross_fitted_observed(
        self,
        n_splits=5,
        random_state=42,
        shuffle=True,
        gamma_f="auto",
        n_gamma_hs=20,
        n_alphas=30,
        alpha_scales="auto",
        n_components=50,
        cv=5,
        groups=None,
        splitter=None,
        gamma_hs="auto",
        nystrom_random_state=1,
        input_block_sizes=None,
        condition_block_sizes=None,
        input_block_distance_weights=None,
        condition_block_distance_weights=None,
    ):
        counts = (
            np.bincount(self.A.astype(int)) if np.all(np.mod(self.A, 1) == 0) else None
        )
        if counts is not None and np.any(counts[counts > 0] < n_splits):
            raise ValueError(
                "Each treatment arm must have at least n_splits observations"
            )
        splits = resolve_crossfit_splits(
            self.X,
            self.A,
            n_splits=n_splits,
            random_state=random_state,
            shuffle=shuffle,
            groups=groups,
            splitter=splitter,
        )
        q_oof = np.zeros(self.A.shape[0], dtype=float)
        self.fold_models_ = []
        self.fold_indices_ = []
        for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
            fold_model = self._fit_one_model(
                A=self.A[train_idx],
                X=self.X[train_idx],
                Z=self.Z[train_idx],
                W=self.W[train_idx],
                gamma_f=gamma_f,
                n_gamma_hs=n_gamma_hs,
                n_alphas=n_alphas,
                alpha_scales=alpha_scales,
                n_components=n_components,
                cv=cv,
                gamma_hs=gamma_hs,
                nystrom_random_state=nystrom_random_state,
                input_block_sizes=input_block_sizes,
                condition_block_sizes=condition_block_sizes,
                input_block_distance_weights=input_block_distance_weights,
                condition_block_distance_weights=condition_block_distance_weights,
            )
            if not np.array_equal(fold_model.classes_, self.classes_):
                raise ValueError("Cross-fit q training fold has incompatible arms")
            q_oof[val_idx] = fold_model.predict_observed_for(
                X_new=self.X[val_idx], Z_new=self.Z[val_idx], A_new=self.A[val_idx]
            )
            self.fold_models_.append(fold_model)
            self.fold_indices_.append((train_idx, val_idx))
        self.q_est_oof_ = q_oof
        return q_oof

    def predict_observed(self):
        return self.q_est

    def predict_observed_oof(self):
        if self.q_est_oof_ is None:
            raise AttributeError(
                "Call fit_cross_fitted_observed(...) before predict_observed_oof()."
            )
        return self.q_est_oof_

    def predict_counterfactual(self, X_new, Z_new):
        if len(self.q_models_) == 0:
            raise AttributeError("Call fit_cv(...) before prediction.")
        X_new = self._ensure_2d(X_new)
        Z_new = self._ensure_2d(Z_new)
        if X_new.shape[0] != Z_new.shape[0]:
            raise ValueError(
                "X_new and Z_new must have the same number of observations"
            )
        input_ZX = np.concatenate((Z_new, X_new), axis=1)
        q_matrix = np.zeros((X_new.shape[0], len(self.classes_)), dtype=float)
        for idx, k in enumerate(self.classes_):
            q_matrix[:, idx] = self.q_models_[k](input_ZX)
        return q_matrix

    def predict_observed_for(self, X_new, Z_new, A_new):
        A_new = np.asarray(A_new).reshape(-1)
        q_matrix = self.predict_counterfactual(X_new=X_new, Z_new=Z_new)
        if q_matrix.shape[0] != A_new.shape[0]:
            raise ValueError("A_new must have the same length as X_new/Z_new")
        q_obs = np.zeros(A_new.shape[0], dtype=float)
        for i, a in enumerate(A_new):
            if a not in self.class_to_col_:
                raise ValueError(f"Treatment arm {a} was not seen during q fitting")
            q_obs[i] = q_matrix[i, self.class_to_col_[a]]
        return q_obs
