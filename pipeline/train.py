"""Train and evaluate county yield models with year-blocked cross-validation.

Protocol: for each evaluation year Y, train only on years < Y and test on Y. Random
k-fold would leak future harvests into the past and produce a model that scores well
but cannot forecast, so it is deliberately not offered.

Every run also scores a trend-only baseline (the per-county expanding OLS line). If a
model cannot beat that, the weather features are not earning their place, and the
comparison is printed rather than left implicit.

In-season forecasting: `--as-of-bin B` truncates every year's season to the first B
biweekly bins, so training data matches what is actually observable at prediction
time for the current season.

Usage:
    python pipeline/train.py --crop corn
    python pipeline/train.py --crop corn --as-of-bin 11     # in-season setup
"""
from __future__ import annotations

import omp_guard  # noqa: F401  # isort:skip  — must load LightGBM before torch

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CROPS, N_TIMESTEPS, PROCESSED, TIMESTEP_FEATURES  # noqa: E402
from models.gbm import GBMModel  # noqa: E402
from models.mmstvit_lite import MMSTViTLite, build_knn  # noqa: E402

STATIC_COLS = [
    "yield_lag1",
    "yield_lag2",
    "yield_lag3",
    "yield_prior_mean",
    "yield_prior_std",
    "n_prior_years",
    "trend_slope",
    "trend_pred",
    "lat",
    "lon",
    "land_sqmi",
    "year",
]
K_NEIGHBORS = 8


def weather_cols(n_bins: int) -> list[str]:
    return [f"{f}_t{b}" for f in TIMESTEP_FEATURES for b in range(n_bins)]


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    resid = y - p
    ss_res = float((resid**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt((resid**2).mean())),
        "mae": float(np.abs(resid).mean()),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "corr": float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 else float("nan"),
    }


# --------------------------------------------------------------------------- data


def load(crop: str, as_of_bin: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Feature frame plus dense weather [N,T,F] and static [N,S] arrays."""
    df = pd.read_parquet(PROCESSED / f"features_{crop}.parquet")
    df = df.sort_values(["fips", "year"]).reset_index(drop=True)

    w = np.stack(
        [
            df[[f"{f}_t{b}" for b in range(as_of_bin)]].to_numpy(dtype=np.float32)
            for f in TIMESTEP_FEATURES
        ],
        axis=-1,
    )  # [N, T, F]
    s = df[STATIC_COLS].to_numpy(dtype=np.float32)
    return df, w, s


def neighbor_index(df: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    """For each row, global row indices of its k nearest counties in the SAME year."""
    fips = df["fips"].to_numpy()
    years = df["year"].to_numpy()
    uniq = np.array(sorted(set(fips)))
    pos = {f: i for i, f in enumerate(uniq)}

    coords = (
        df.groupby("fips")[["lat", "lon"]].first().reindex(uniq).to_numpy(dtype=float)
    )
    knn = build_knn(coords, k)  # [n_counties, k] -> indices into `uniq`

    row_of = {(f, y): i for i, (f, y) in enumerate(zip(fips, years))}
    idx = np.zeros((len(df), k), dtype=np.int64)
    pad = np.ones((len(df), k), dtype=bool)  # True = padding, ignored by attention
    for i, (f, y) in enumerate(zip(fips, years)):
        for j, nb in enumerate(knn[pos[f]]):
            r = row_of.get((uniq[nb], y))
            if r is not None:
                idx[i, j] = r
                pad[i, j] = False
            else:
                idx[i, j] = i  # harmless self-reference; masked out anyway
    return idx, pad


# ------------------------------------------------------------------------- models


def train_nn(
    w: np.ndarray,
    s: np.ndarray,
    nbr_idx: np.ndarray,
    nbr_pad: np.ndarray,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    y_all: np.ndarray,
    epochs: int = 60,
    batch: int = 256,
    seed: int = 0,
) -> tuple[MMSTViTLite, dict]:
    """Fit MMST-ViT-Lite. Normalisation uses training-fold statistics only."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    w_mu = np.nanmean(w[train_rows], axis=(0, 1), keepdims=True)
    w_sd = np.nanstd(w[train_rows], axis=(0, 1), keepdims=True) + 1e-6
    s_mu = np.nanmean(s[train_rows], axis=0, keepdims=True)
    s_sd = np.nanstd(s[train_rows], axis=0, keepdims=True) + 1e-6
    y_mu = float(y_all[train_rows].mean())
    y_sd = float(y_all[train_rows].std() + 1e-6)

    # Standardise, then zero-fill: after centring, 0 is the training mean, which is
    # the right neutral value for a missing lag or an unobserved late-season bin.
    W = np.nan_to_num((w - w_mu) / w_sd).astype(np.float32)
    S = np.nan_to_num((s - s_mu) / s_sd).astype(np.float32)

    Wt = torch.from_numpy(W)
    St = torch.from_numpy(S)
    Yt = torch.from_numpy(((y_all - y_mu) / y_sd).astype(np.float32))
    NI = torch.from_numpy(nbr_idx)
    NP = torch.from_numpy(nbr_pad)

    model = MMSTViTLite(
        n_weather=W.shape[2], n_static=S.shape[1], n_timesteps=W.shape[1]
    )
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.SmoothL1Loss()

    def batch_forward(rows: torch.Tensor) -> torch.Tensor:
        ni = NI[rows]
        return model(
            Wt[rows], St[rows], Wt[ni], St[ni], nbr_valid=NP[rows]
        )

    tr = torch.from_numpy(train_rows)
    va = torch.from_numpy(val_rows) if len(val_rows) else None
    best, best_state, patience = float("inf"), None, 0

    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), batch):
            rows = perm[i : i + batch]
            opt.zero_grad()
            loss = lossf(batch_forward(rows), Yt[rows])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if va is not None:
            model.eval()
            with torch.no_grad():
                preds = torch.cat(
                    [batch_forward(va[i : i + 1024]) for i in range(0, len(va), 1024)]
                )
                vl = float(nn.functional.mse_loss(preds, Yt[va]))
            if vl < best - 1e-5:
                best, patience = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 12:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    norm = dict(w_mu=w_mu, w_sd=w_sd, s_mu=s_mu, s_sd=s_sd, y_mu=y_mu, y_sd=y_sd)
    return model, norm


def nn_predict(
    model: MMSTViTLite,
    norm: dict,
    w: np.ndarray,
    s: np.ndarray,
    nbr_idx: np.ndarray,
    nbr_pad: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    W = torch.from_numpy(
        np.nan_to_num((w - norm["w_mu"]) / norm["w_sd"]).astype(np.float32)
    )
    S = torch.from_numpy(
        np.nan_to_num((s - norm["s_mu"]) / norm["s_sd"]).astype(np.float32)
    )
    NI = torch.from_numpy(nbr_idx)
    NP = torch.from_numpy(nbr_pad)
    r = torch.from_numpy(rows)
    out = []
    with torch.no_grad():
        for i in range(0, len(r), 1024):
            b = r[i : i + 1024]
            ni = NI[b]
            out.append(model(W[b], S[b], W[ni], S[ni], nbr_valid=NP[b]))
    return (torch.cat(out).numpy() * norm["y_sd"]) + norm["y_mu"]


# ---------------------------------------------------------------------------- CV


def run(crop: str, as_of_bin: int, eval_years: list[int], skip_nn: bool) -> dict:
    df, w, s = load(crop, as_of_bin)
    wcols = weather_cols(as_of_bin)
    X_all = df[wcols + STATIC_COLS]
    y_all = df["yield"].to_numpy(dtype=np.float64)

    print("Building neighbour index ...")
    nbr_idx, nbr_pad = neighbor_index(df, K_NEIGHBORS)

    labeled = df["yield"].notna().to_numpy()
    usable = labeled & df["yield_lag1"].notna().to_numpy()

    results, preds_by_year = [], {}
    for Y in eval_years:
        tr = np.where(usable & (df["year"] < Y).to_numpy())[0]
        te = np.where(usable & (df["year"] == Y).to_numpy())[0]
        if len(te) < 50 or len(tr) < 500:
            print(f"  {Y}: skipped (train={len(tr)}, test={len(te)})")
            continue

        # Hold out the two most recent training years for early stopping.
        val_cut = Y - 2
        va = tr[df["year"].to_numpy()[tr] >= val_cut]
        tr_fit = tr[df["year"].to_numpy()[tr] < val_cut]
        if len(va) == 0 or len(tr_fit) < 500:
            tr_fit, va = tr, tr[:0]

        row = {"year": Y, "n_train": len(tr_fit), "n_test": len(te)}
        yte = y_all[te]

        row["trend"] = metrics(yte, df["trend_pred"].to_numpy()[te])

        gbm = GBMModel().fit(
            X_all.iloc[tr_fit], y_all[tr_fit],
            X_all.iloc[va] if len(va) else None,
            y_all[va] if len(va) else None,
        )
        p_gbm = gbm.predict(X_all.iloc[te])
        row["gbm"] = metrics(yte, p_gbm)

        if not skip_nn:
            t0 = time.time()
            model, norm = train_nn(w, s, nbr_idx, nbr_pad, tr_fit, va, y_all)
            p_nn = nn_predict(model, norm, w, s, nbr_idx, nbr_pad, te)
            row["mmstvit_lite"] = metrics(yte, p_nn)
            row["mmstvit_lite"]["train_sec"] = round(time.time() - t0, 1)
            p_ens = 0.5 * p_gbm + 0.5 * p_nn
            row["ensemble"] = metrics(yte, p_ens)
            preds_by_year[Y] = dict(
                fips=df["fips"].to_numpy()[te].tolist(),
                actual=yte.tolist(),
                gbm=p_gbm.tolist(),
                nn=p_nn.tolist(),
            )
        else:
            preds_by_year[Y] = dict(
                fips=df["fips"].to_numpy()[te].tolist(),
                actual=yte.tolist(),
                gbm=p_gbm.tolist(),
            )

        parts = " | ".join(
            f"{k}: R2={row[k]['r2']:.3f} RMSE={row[k]['rmse']:.1f}"
            for k in ("trend", "gbm", "mmstvit_lite", "ensemble")
            if k in row
        )
        print(f"  {Y} (n={len(te):,})  {parts}")
        results.append(row)

    summary = {}
    for key in ("trend", "gbm", "mmstvit_lite", "ensemble"):
        rows = [r[key] for r in results if key in r]
        if rows:
            n = sum(r["n"] for r in rows)
            summary[key] = {
                "r2_mean": float(np.mean([r["r2"] for r in rows])),
                "rmse_mean": float(np.mean([r["rmse"] for r in rows])),
                "mae_mean": float(np.mean([r["mae"] for r in rows])),
                "corr_mean": float(np.mean([r["corr"] for r in rows])),
                "n_total": int(n),
            }

    return {
        "crop": crop,
        "as_of_bin": as_of_bin,
        "eval_years": eval_years,
        "per_year": results,
        "summary": summary,
        "gbm_top_features": gbm.importances(15).to_dict("records"),
        "predictions": preds_by_year,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", choices=list(CROPS), required=True)
    ap.add_argument("--as-of-bin", type=int, default=N_TIMESTEPS)
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--skip-nn", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    print(f"=== {args.crop}  as_of_bin={args.as_of_bin}  eval {years[0]}-{years[-1]} ===")
    out = run(args.crop, args.as_of_bin, years, args.skip_nn)

    print("\nSummary (mean over evaluation years):")
    for k, v in out["summary"].items():
        print(
            f"  {k:14s} R2={v['r2_mean']:.3f}  RMSE={v['rmse_mean']:.2f}  "
            f"MAE={v['mae_mean']:.2f}  corr={v['corr_mean']:.3f}"
        )

    tag = args.tag or (f"asof{args.as_of_bin}" if args.as_of_bin != N_TIMESTEPS else "full")
    path = PROCESSED / f"cv_{args.crop}_{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
