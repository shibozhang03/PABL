import numpy as np
from pabl.kernel import ApproxRKHSIVCV
from pabl.splits import resolve_crossfit_splits


class MultiTreatmentProxH:

    def __init__(self, A, X, Z, W, Y):
        self.A = np.asarray(A).reshape(-1)
        self.X = self._ensure_2d(X)
        self.Z = self._ensure_2d(Z)
        self.W = self._ensure_2d(W)
        self.Y = np.asarray(Y).reshape(-1)
        n = self.A.shape[0]
        if (
            self.X.shape[0] != n
            or self.Z.shape[0] != n
            or self.W.shape[0] != n
            or (self.Y.shape[0] != n)
        ):
            raise ValueError("A, X, Z, W, Y must have the same number of observations")
        self.classes_ = np.unique(self.A)
        self.class_to_col_ = {k: idx for (idx, k) in enumerate(self.classes_)}
        self.h_models_ = {}
        self.h_estimators_ = {}
        self.h_params_ = {}
        self.h_obs_ = np.zeros(n, dtype=float)
        self.h_obs_oof_ = None
        self.h_all_oof_ = None
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

    @staticmethod
    def _effective_components(n_components, n_samples):
        if n_samples <= 1:
            raise ValueError("Each treatment arm must have at least two observations")
        return int(min(n_components, max(2, n_samples - 1)))

    @staticmethod
    def _effective_cv(cv, n_samples):
        if n_samples < 2:
            raise ValueError("Each treatment arm must have at least two observations")
        return int(min(cv, n_samples))

    def _fit_one_model(
        self,
        A,
        X,
        Z,
        W,
        Y,
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
        model = self.__class__(A=A, X=X, Z=Z, W=W, Y=Y)
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
        self.h_models_ = {}
        self.h_estimators_ = {}
        self.h_params_ = {}
        self.h_obs_ = np.zeros(self.A.shape[0], dtype=float)
        for k in self.classes_:
            index = self.A == k
            n_arm = int(np.sum(index))
            if n_arm == 0:
                raise ValueError(f"Treatment arm {k} has no observations")
            bridge_X = np.concatenate((self.W[index], self.X[index]), axis=1)
            bridge_condition = np.concatenate((self.Z[index], self.X[index]), axis=1)
            y_arm = self.Y[index]
            model_cv = ApproxRKHSIVCV(
                gamma_f=gamma_f,
                gamma_hs=gamma_hs,
                n_gamma_hs=n_gamma_hs,
                n_alphas=n_alphas,
                alpha_scales=alpha_scales,
                cv=self._effective_cv(cv, n_arm),
                n_components=self._effective_components(n_components, n_arm),
                nystrom_random_state=nystrom_random_state,
                input_block_sizes=input_block_sizes,
                condition_block_sizes=condition_block_sizes,
                input_block_distance_weights=input_block_distance_weights,
                condition_block_distance_weights=condition_block_distance_weights,
            )
            model_cv.fit(X=bridge_X, y=y_arm, condition=bridge_condition)
            self.h_params_[k] = {
                "gamma_f": model_cv.gamma_f,
                "gamma_h": model_cv.gamma_h,
                "alpha_scale": model_cv.best_alpha_scale,
                "delta": getattr(model_cv, "fitted_delta", None),
            }
            self.h_estimators_[k] = model_cv
            self.h_models_[k] = model_cv.predict
            self.h_obs_[index] = model_cv.predict(bridge_X)
        return self

    def fit_cross_fitted(
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
        h_obs_oof = np.zeros(self.A.shape[0], dtype=float)
        h_all_oof = np.zeros((self.A.shape[0], len(self.classes_)), dtype=float)
        self.fold_models_ = []
        self.fold_indices_ = []
        for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
            fold_model = self._fit_one_model(
                A=self.A[train_idx],
                X=self.X[train_idx],
                Z=self.Z[train_idx],
                W=self.W[train_idx],
                Y=self.Y[train_idx],
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
                raise ValueError("Cross-fit h training fold has incompatible arms")
            h_all_oof[val_idx, :] = fold_model.predict_counterfactual(
                X_new=self.X[val_idx], W_new=self.W[val_idx]
            )
            h_obs_oof[val_idx] = fold_model.predict_observed_for(
                X_new=self.X[val_idx], W_new=self.W[val_idx], A_new=self.A[val_idx]
            )
            self.fold_models_.append(fold_model)
            self.fold_indices_.append((train_idx, val_idx))
        self.h_obs_oof_ = h_obs_oof
        self.h_all_oof_ = h_all_oof
        return (h_obs_oof, h_all_oof)

    def predict_observed(self):
        if len(self.h_models_) == 0:
            raise AttributeError("Call fit_cv(...) before prediction.")
        return self.h_obs_

    def predict_observed_oof(self):
        if self.h_obs_oof_ is None:
            raise AttributeError(
                "Call fit_cross_fitted(...) before predict_observed_oof()."
            )
        return self.h_obs_oof_

    def predict_counterfactual_oof(self):
        if self.h_all_oof_ is None:
            raise AttributeError(
                "Call fit_cross_fitted(...) before predict_counterfactual_oof()."
            )
        return self.h_all_oof_

    def predict_counterfactual(self, X_new, W_new):
        if len(self.h_models_) == 0:
            raise AttributeError("Call fit_cv(...) before prediction.")
        X_new = self._ensure_2d(X_new)
        W_new = self._ensure_2d(W_new)
        if X_new.shape[0] != W_new.shape[0]:
            raise ValueError(
                "X_new and W_new must have the same number of observations"
            )
        bridge_X = np.concatenate((W_new, X_new), axis=1)
        h_matrix = np.zeros((X_new.shape[0], len(self.classes_)), dtype=float)
        for idx, k in enumerate(self.classes_):
            h_matrix[:, idx] = self.h_models_[k](bridge_X)
        return h_matrix

    def predict_observed_for(self, X_new, W_new, A_new):
        A_new = np.asarray(A_new).reshape(-1)
        h_matrix = self.predict_counterfactual(X_new=X_new, W_new=W_new)
        if h_matrix.shape[0] != A_new.shape[0]:
            raise ValueError("A_new must have the same length as X_new/W_new")
        h_obs = np.zeros(A_new.shape[0], dtype=float)
        for i, a in enumerate(A_new):
            if a not in self.class_to_col_:
                raise ValueError(f"Treatment arm {a} was not seen during h fitting")
            h_obs[i] = h_matrix[i, self.class_to_col_[a]]
        return h_obs
