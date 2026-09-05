"""Forecast the current in-season year with an LSTM, per crop.

This generalizes notebooks/02_model.ipynb cell 19 (corn-only, ad hoc) into a
reproducible script for both corn and soybeans. Historical years already have
real NASS yields and need no model; only the current season (2026, no labels
yet, partially observed weather) gets a prediction.

Two choices carried over verbatim from the notebook, with their reasoning:

1. AS-OF CUTOFF. The forecast year's season is only partially observed
   (bins t0-9 complete as of writing, see n_days_t* in the feature table).
   Training on full 14-bin seasons and then predicting from a truncated one
   is a train/inference mismatch, so both training and inference are cut to
   the same AS_OF_BIN.

2. ANOMALY BASELINE = county's own trailing mean actual yield, not
   trend_pred. trend_pred is biased low by double digits (see features.py /
   README), which would paint every county as "up" regardless of the season.
   The plain trailing mean needs no bias correction.

Usage:
    python pipeline/predict_lstm.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model

from config import CROPS, PROCESSED

AS_OF_BIN = 10          # complete biweekly bins -> season through ~mid-August
FORECAST_YEAR = 2026
BASELINE_YEARS = 5
TRAIN_CUTOFF_YEAR = 2023   # train on years <= this
VAL_YEARS = (2024, 2025)

HISTORY_COLS = [
    "yield_lag1", "yield_lag2", "yield_lag3", "yield_prior_mean",
    "yield_prior_std", "n_prior_years", "trend_slope", "trend_pred",
]
STATIC_COLS = HISTORY_COLS + ["lat", "lon", "land_sqmi", "year"]
WEATHER_VARS = [
    "gdd", "precip", "tmax_mean", "tmax_max",
    "heat_days", "et0", "radiation", "water_balance",
]


def weather_cols(as_of_bin: int) -> list[str]:
    return [f"{v}_t{t}" for t in range(as_of_bin) for v in WEATHER_VARS]


def make_arrays(d: pd.DataFrame, w_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    W = d[w_cols].to_numpy(dtype=np.float32).reshape(-1, AS_OF_BIN, 8)
    S = d[STATIC_COLS].to_numpy(dtype=np.float32)
    return W, S


def build_model(n_static: int) -> Model:
    weather_in = Input(shape=(AS_OF_BIN, 8), name="weather")
    x = LSTM(64, return_sequences=True)(weather_in)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)

    static_in = Input(shape=(n_static,), name="static")
    s = Dense(32, activation="relu")(static_in)

    c = Concatenate()([x, s])
    c = Dense(64, activation="relu")(c)
    c = Dropout(0.2)(c)
    out = Dense(1, name="yield")(c)

    model = Model(inputs=[weather_in, static_in], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def forecast_crop(crop: str) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / f"features_{crop}.parquet")
    w_cols = weather_cols(AS_OF_BIN)

    labeled = df[df["yield"].notna()]
    train = labeled[labeled["year"] <= TRAIN_CUTOFF_YEAR]
    val = labeled[labeled["year"].isin(VAL_YEARS)]

    # Only counties that reported this crop recently get a forecast row; the
    # rest render as "no data" downstream rather than extrapolating into
    # counties that don't grow it.
    recent_fips = set(labeled[labeled["year"] >= FORECAST_YEAR - BASELINE_YEARS]["fips"])
    fc = df[(df["year"] == FORECAST_YEAR) & df["fips"].isin(recent_fips)].copy()

    if fc.empty:
        print(f"{crop}: no {FORECAST_YEAR} rows for recently-reporting counties, skipping")
        return pd.DataFrame(columns=["fips", "year", "crop", "predicted_yield", "baseline_5yr_mean", "val_mae"])

    W_train, S_train = make_arrays(train, w_cols)
    W_val, S_val = make_arrays(val, w_cols)
    W_fc, S_fc = make_arrays(fc, w_cols)
    y_train = train["yield"].to_numpy(dtype=np.float32)
    y_val = val["yield"].to_numpy(dtype=np.float32)

    assert not np.isnan(W_fc).any(), f"{crop}: {FORECAST_YEAR} weather has NaNs at bin {AS_OF_BIN} -- lower AS_OF_BIN"

    w_imputer = SimpleImputer(strategy="median")
    W_train = w_imputer.fit_transform(W_train.reshape(len(W_train), -1)).reshape(-1, AS_OF_BIN, 8)
    W_val = w_imputer.transform(W_val.reshape(len(W_val), -1)).reshape(-1, AS_OF_BIN, 8)
    W_fc = w_imputer.transform(W_fc.reshape(len(W_fc), -1)).reshape(-1, AS_OF_BIN, 8)

    s_imputer = SimpleImputer(strategy="median")
    S_train = s_imputer.fit_transform(S_train)
    S_val = s_imputer.transform(S_val)
    S_fc = s_imputer.transform(S_fc)

    w_scaler = StandardScaler()
    W_train = w_scaler.fit_transform(W_train.reshape(-1, 8)).reshape(-1, AS_OF_BIN, 8)
    W_val = w_scaler.transform(W_val.reshape(-1, 8)).reshape(-1, AS_OF_BIN, 8)
    W_fc = w_scaler.transform(W_fc.reshape(-1, 8)).reshape(-1, AS_OF_BIN, 8)

    s_scaler = StandardScaler()
    S_train = s_scaler.fit_transform(S_train)
    S_val = s_scaler.transform(S_val)
    S_fc = s_scaler.transform(S_fc)

    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

    print(f"{crop}: training at {AS_OF_BIN}-bin cutoff ({len(train):,} rows)...")
    tf.random.set_seed(0)
    np.random.seed(0)
    model = build_model(n_static=len(STATIC_COLS))
    model.fit(
        [W_train, S_train], y_train_s,
        validation_data=([W_val, S_val], y_val_s),
        epochs=100, batch_size=64, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
    )

    val_pred = y_scaler.inverse_transform(model.predict([W_val, S_val], verbose=0)).ravel()
    val_mae = float(mean_absolute_error(y_val, val_pred))

    fc["predicted_yield"] = y_scaler.inverse_transform(model.predict([W_fc, S_fc], verbose=0)).ravel()
    assert np.isfinite(fc["predicted_yield"]).all()

    baseline = (
        labeled[(labeled["year"] < FORECAST_YEAR) & (labeled["year"] >= FORECAST_YEAR - BASELINE_YEARS)]
        .groupby("fips")["yield"].mean()
    )
    fc["baseline_5yr_mean"] = fc["fips"].map(baseline)
    fc["val_mae"] = val_mae
    fc["crop"] = crop

    print(
        f"{crop}: {len(fc):,} counties forecast, val MAE {val_mae:.2f} bu/acre, "
        f"mean predicted {fc['predicted_yield'].mean():.1f} bu/acre"
    )
    return fc[["fips", "year", "crop", "predicted_yield", "baseline_5yr_mean", "val_mae"]]


def main() -> None:
    for crop in CROPS:
        out = forecast_crop(crop)
        path = PROCESSED / f"predictions_{crop}.parquet"
        out.to_parquet(path, index=False)
        print(f"  -> {path.name}")


if __name__ == "__main__":
    main()
