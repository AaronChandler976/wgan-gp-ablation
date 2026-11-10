"""Forecast accuracy metrics, provisioning rates, baselines and the Diebold-Mariano test.

All metrics take values in the original units (jobs per interval), not scaled values.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def mape(y, p) -> float:
    """Mean absolute percentage error (%). Intervals with an actual value of 0 are skipped."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = y != 0
    return float(np.mean(np.abs((y[ok] - p[ok]) / y[ok])) * 100)


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def wape(y, p) -> float:
    """Weighted absolute percentage error (%): total error over total jobs."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y)) * 100)


def naive_scale(train) -> float:
    """Mean absolute error of 'repeat the last value' on the training part (MASE denominator)."""
    return float(np.mean(np.abs(np.diff(np.asarray(train, float)))))


def mase(y, p, scale: float) -> float:
    """Mean absolute scaled error. Below 1 beats the naive forecast's training-set error."""
    return mae(y, p) / scale if scale > 0 else float("nan")


def provisioning(y, p) -> dict:
    """One VM per predicted job. Rates are the share of intervals (%) with too few / too many VMs."""
    y = np.asarray(y, float)
    vms = np.ceil(np.clip(np.asarray(p, float), 0, None))
    return {"under_rate": float(np.mean(y > vms) * 100),
            "over_rate": float(np.mean(y < vms) * 100),
            "missing_vm_pct": float(np.sum(np.clip(y - vms, 0, None)) / np.sum(y) * 100),
            "idle_vm_pct": float(np.sum(np.clip(vms - y, 0, None)) / np.sum(y) * 100)}


def all_metrics(y, p, scale: float) -> dict:
    return {"mape": mape(y, p), "mae": mae(y, p), "rmse": rmse(y, p), "wape": wape(y, p),
            "mase": mase(y, p, scale), **provisioning(y, p)}


# --------------------------------------------------------------------------- baselines

def baseline_predictions(series: np.ndarray, start: int, end: int, season: int) -> dict:
    """Simple forecasts for targets t in [start, end), using only values before t."""
    idx = np.arange(start, end)
    out = {"naive": series[idx - 1]}
    if season and start >= season:
        out["seasonal_naive"] = series[idx - season]
    for k in (3, 6):
        if start >= k:
            out[f"moving_avg_{k}"] = np.mean([series[idx - j] for j in range(1, k + 1)], axis=0)
    return out


# --------------------------------------------------------------------------- significance

def diebold_mariano(y, p1, p2) -> dict:
    """Diebold-Mariano test on absolute errors for one-step forecasts (Harvey correction).

    A negative statistic means forecast 1 has the smaller error.
    """
    y, p1, p2 = (np.asarray(a, float) for a in (y, p1, p2))
    d = np.abs(y - p1) - np.abs(y - p2)
    n = len(d)
    var = np.var(d, ddof=0)
    if var == 0:
        return {"stat": 0.0, "p": 1.0, "n": n}
    stat = np.mean(d) / np.sqrt(var / n) * np.sqrt((n - 1) / n)
    return {"stat": float(stat), "p": float(2 * stats.t.sf(abs(stat), df=n - 1)), "n": n}


def holm(pvalues: list[float]) -> list[float]:
    """Holm-adjusted p-values, in the original order."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
