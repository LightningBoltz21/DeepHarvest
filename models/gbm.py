"""LightGBM county yield regressor.

The accuracy workhorse. County-level yield is largely driven by a trend plus a
handful of season aggregates, which is exactly the regime gradient boosting handles
well, and published deep models rarely beat a well-tuned GBM here by much.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.03,
    num_leaves=63,
    min_data_in_leaf=40,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbosity=-1,
    num_threads=0,
)


def _as_matrix(X: pd.DataFrame) -> np.ndarray:
    """DataFrame -> contiguous float64.

    LightGBM 4.7 segfaults in its C Dataset constructor when handed a pandas 3.0
    DataFrame directly, so the conversion is done here rather than relying on
    LightGBM's own pandas handling.
    """
    return np.ascontiguousarray(X.to_numpy(dtype=np.float64))


class GBMModel:
    name = "gbm"

    def __init__(self, n_rounds: int = 1500, **overrides):
        self.params = {**PARAMS, **overrides}
        self.n_rounds = n_rounds
        self.booster: lgb.Booster | None = None
        self.features: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> "GBMModel":
        self.features = list(X.columns)
        train_set = lgb.Dataset(
            _as_matrix(X),
            label=np.asarray(y, dtype=np.float64),
            feature_name=self.features,
            free_raw_data=False,
        )
        callbacks = [lgb.log_evaluation(0)]
        valid_sets = None
        if X_val is not None and len(X_val):
            valid_sets = [
                lgb.Dataset(
                    _as_matrix(X_val),
                    label=np.asarray(y_val, dtype=np.float64),
                    reference=train_set,
                )
            ]
            callbacks.append(lgb.early_stopping(100, verbose=False))
        self.booster = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.n_rounds,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.booster is not None, "fit() before predict()"
        return self.booster.predict(_as_matrix(X[self.features]))

    def importances(self, top: int = 25) -> pd.DataFrame:
        assert self.booster is not None
        return (
            pd.DataFrame(
                {
                    "feature": self.features,
                    "gain": self.booster.feature_importance("gain"),
                }
            )
            .sort_values("gain", ascending=False)
            .head(top)
            .reset_index(drop=True)
        )
