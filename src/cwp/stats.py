"""Dataset metrics: how spiky, how seasonal and how hard to forecast each series is.

Everything is computed on the training part only, so the test data stays untouched.

    python -m cwp.stats            # all preprocessed datasets -> results/summary/dataset_stats.*
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config, data, metrics


def acf(x: np.ndarray, lag: int) -> float:
    """Autocorrelation at one lag."""
    if lag <= 0 or lag >= len(x):
        return float("nan")
    x = x - x.mean()
    return float(np.dot(x[:-lag], x[lag:]) / np.dot(x, x))


def seasonal_strength(x: np.ndarray, period: int) -> float:
    """1 - Var(remainder) / Var(seasonal + remainder), from a classical decomposition.

    0 means no repeating daily shape, values near 1 mean the daily shape explains almost all
    of the variation around the trend.
    """
    if len(x) < 2 * period:
        return float("nan")
    trend = pd.Series(x).rolling(period, center=True, min_periods=period // 2).mean().to_numpy()
    detr = x - trend
    phase = np.arange(len(x)) % period
    seasonal = np.array([np.nanmean(detr[phase == k]) for k in range(period)])[phase]
    rem = detr - seasonal
    denom = np.nanvar(seasonal + rem)
    return float(max(0.0, 1.0 - np.nanvar(rem) / denom)) if denom > 0 else float("nan")


def spectral_entropy(x: np.ndarray) -> float:
    """Normalized entropy of the power spectrum (0 = one clean cycle, 1 = white noise)."""
    power = np.abs(np.fft.rfft(x - x.mean())[1:]) ** 2
    p = power / power.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(power)))


def describe(dataset: str, interval: int) -> dict:
    series = data.load_series(dataset, interval)
    split = data.get_split(dataset, interval)
    x = series[: split.train_end]
    per_day = 1440 // interval
    naive = x[:-1]
    row = {
        "dataset": dataset, "interval_min": interval, "points": len(series),
        "days": round(len(series) * interval / 1440, 1), "train_points": len(x),
        "mean": x.mean(), "cv": x.std() / x.mean(),
        "p99_over_median": np.percentile(x, 99) / np.median(x) if np.median(x) > 0 else np.nan,
        "kurtosis": float(sps.kurtosis(x)),
        "acf_1_step": acf(x, 1), "acf_1_day": acf(x, per_day),
        "seasonal_strength": seasonal_strength(x, per_day),
        "spectral_entropy": spectral_entropy(x),
        "pct_near_zero": float(np.mean(x <= 0.01 * x.mean()) * 100),
        "naive_mape": metrics.mape(x[1:], naive), "naive_mae": metrics.mae(x[1:], naive),
    }
    if len(x) > per_day:
        row["seasonal_naive_mape"] = metrics.mape(x[per_day:], x[:-per_day])
    return row


def main(argv=None) -> None:
    rows = []
    for name, d in config.datasets().items():
        for iv in d["intervals"]:
            if data.processed_path(name, iv).exists():
                rows.append(describe(name, iv))
    if not rows:
        sys.exit("No preprocessed datasets yet. See `python -m cwp.data --help`.")
    df = pd.DataFrame(rows).round(3)
    out = config.path("results", "summary")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "dataset_stats.csv", index=False)
    (out / "dataset_stats.md").write_text(
        "# Dataset metrics (training part only)\n\n" + config.md_table(df) + "\n")
    print(df.T.to_string(header=False))
    print(f"\nsaved to {out / 'dataset_stats.md'}")


if __name__ == "__main__":
    sys.exit(main())
