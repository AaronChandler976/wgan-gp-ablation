"""Datasets: download, preprocess into job-arrival-rate (JAR) series, split, scale, window.

Commands:
    python -m cwp.data download   --dataset google|azure_functions
    python -m cwp.data preprocess --dataset google|azure_functions
    python -m cwp.data synthetic
    python -m cwp.data list
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

GOOGLE_URL = "https://storage.googleapis.com/clusterdata-2011-2/job_events/part-{i:05d}-of-00500.csv.gz"
GOOGLE_PARTS = 500
GOOGLE_START_US = 600_000_000          # the trace starts 600 s after its time origin
GOOGLE_MINUTES = 29 * 24 * 60          # 29 days
AZURE_URL = ("https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/"
             "azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz")


# --------------------------------------------------------------------------- loading

def processed_path(dataset: str, interval: int) -> Path:
    return config.path("data", "processed", f"{dataset}_{interval}m.csv")


def split_path(dataset: str, interval: int) -> Path:
    return config.path("data", "splits", f"{dataset}-{interval}m.json")


def load_series(dataset: str, interval: int) -> np.ndarray:
    p = processed_path(dataset, interval)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run: python -m cwp.data preprocess --dataset {dataset} "
            f"(after `download`), or `python -m cwp.data synthetic` for the test series.")
    return pd.read_csv(p)["job_count"].to_numpy(dtype=np.float64)


def file_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


@dataclass(frozen=True)
class Split:
    n: int
    train_end: int   # train = [0, train_end)
    val_end: int     # validation = [train_end, val_end), test = [val_end, n)


def compute_split(n: int, fractions=(0.6, 0.2, 0.2)) -> Split:
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    return Split(n, int(math.floor(n * fractions[0])), int(math.floor(n * (fractions[0] + fractions[1]))))


def get_split(dataset: str, interval: int, fractions=(0.6, 0.2, 0.2)) -> Split:
    """Read the committed split file, or compute it. A stale file is an error."""
    n = len(load_series(dataset, interval))
    p = split_path(dataset, interval)
    if p.exists():
        d = json.loads(p.read_text())
        if d["n"] != n:
            raise ValueError(f"{p} was made for {d['n']} points but the series has {n}. "
                             f"Re-run `python -m cwp.data preprocess`.")
        return Split(d["n"], d["train_end"], d["val_end"])
    return compute_split(n, fractions)


def write_split(dataset: str, interval: int, fractions=(0.6, 0.2, 0.2)) -> Split:
    series = load_series(dataset, interval)
    s = compute_split(len(series), fractions)
    p = split_path(dataset, interval)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"dataset": dataset, "interval_minutes": interval, "n": s.n,
                             "train_end": s.train_end, "val_end": s.val_end,
                             "fractions": list(fractions),
                             "processed_sha256": file_sha256(processed_path(dataset, interval))},
                            indent=2) + "\n")
    return s


# --------------------------------------------------------------------------- scaling, windows

class MinMaxScaler:
    """Scale to (-1, 1) using the minimum and maximum of the training part only."""

    def __init__(self, lo: float, hi: float):
        self.lo, self.hi = float(lo), float(hi)
        self.span = (self.hi - self.lo) or 1.0

    @classmethod
    def fit(cls, train: np.ndarray) -> "MinMaxScaler":
        return cls(train.min(), train.max())

    def transform(self, x):
        return (x - self.lo) / self.span * 2.0 - 1.0

    def inverse(self, x):
        return (x + 1.0) / 2.0 * self.span + self.lo

    def state(self) -> dict:
        return {"lo": self.lo, "hi": self.hi}


def make_windows(series: np.ndarray, window: int, start: int, end: int):
    """Inputs X [N, window] and targets y [N] for target positions t in [start, end).

    The history of a target may reach back before `start` (into the previous part of the
    series). It never contains the target or anything after it, so nothing leaks, and every
    model is scored on the same target points whatever its window size.
    """
    first = max(start, window)
    if first >= end:
        raise ValueError(f"window {window} leaves no targets in [{start}, {end})")
    idx = np.arange(first, end)
    X = np.stack([series[t - window:t] for t in idx])
    return X, series[idx], idx


# --------------------------------------------------------------------------- download

def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)


def download(dataset: str) -> None:
    raw = config.path("data", "raw", dataset)
    if dataset == "google":
        for i in range(GOOGLE_PARTS):
            _fetch(GOOGLE_URL.format(i=i), raw / f"part-{i:05d}-of-00500.csv.gz")
            if i % 50 == 0:
                print(f"  google job_events {i}/{GOOGLE_PARTS}", flush=True)
    elif dataset == "azure_functions":
        archive = raw / "azurefunctions-dataset2019.tar.xz"
        print("  downloading the Azure Functions 2019 archive (a few hundred MB)", flush=True)
        _fetch(AZURE_URL, archive)
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if "invocations_per_function" in m.name]
            tar.extractall(raw, members=members, filter="data")
    else:
        raise ValueError(f"no download for dataset '{dataset}'")
    print(f"raw files are in {raw}")


# --------------------------------------------------------------------------- preprocess

def google_minutes(raw: Path) -> np.ndarray:
    """Job SUBMIT events per minute (event type 0), as the paper's code counts them."""
    files = sorted(raw.glob("part-*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"no job_events files in {raw}. Run `download` first.")
    counts = np.zeros(GOOGLE_MINUTES, dtype=np.int64)
    for f in files:
        df = pd.read_csv(f, header=None, usecols=[0, 3], names=["time", "event"])
        t = df.loc[(df["event"] == 0) & (df["time"] >= GOOGLE_START_US), "time"].to_numpy()
        minute = (t - GOOGLE_START_US) // 60_000_000
        minute = minute[minute < GOOGLE_MINUTES]
        counts += np.bincount(minute, minlength=GOOGLE_MINUTES)
    return counts


def azure_minutes(raw: Path) -> np.ndarray:
    """Total function invocations per minute, summed over all functions, for the 14 days."""
    files = sorted(raw.rglob("invocations_per_function_md.anon.d*.csv"))
    if not files:
        raise FileNotFoundError(f"no invocations_per_function files in {raw}. Run `download` first.")
    cols = [str(m) for m in range(1, 1441)]
    days = []
    for f in files:
        total = np.zeros(1440, dtype=np.int64)
        for chunk in pd.read_csv(f, usecols=cols, chunksize=20000):
            total += chunk[cols].to_numpy(dtype=np.int64).sum(axis=0)
        days.append(total)
    return np.concatenate(days)


def aggregate(minutes: np.ndarray, interval: int) -> np.ndarray:
    """Sum per-minute counts into `interval`-minute blocks, dropping an incomplete last block."""
    n = len(minutes) // interval
    return minutes[:n * interval].reshape(n, interval).sum(axis=1)


def write_processed(dataset: str, interval: int, series: np.ndarray) -> None:
    p = processed_path(dataset, interval)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"t": np.arange(len(series)), "job_count": series}).to_csv(p, index=False)
    s = write_split(dataset, interval)
    print(f"  {p.name}: {len(series)} points  train [0,{s.train_end})  "
          f"val [{s.train_end},{s.val_end})  test [{s.val_end},{s.n})")


def preprocess(dataset: str, intervals: list[int] | None = None) -> None:
    raw = config.path("data", "raw", dataset)
    minutes = {"google": google_minutes, "azure_functions": azure_minutes}[dataset](raw)
    for interval in intervals or config.datasets()[dataset]["intervals"]:
        write_processed(dataset, interval, aggregate(minutes, interval))


def synthetic() -> None:
    """Two small generated series so tests and timing work before the real data is downloaded."""
    rng = np.random.default_rng(252)
    n, per_day = 14 * 144, 144                     # 14 days of 10-minute intervals
    t = np.arange(n)
    daily = np.sin(2 * np.pi * (t % per_day) / per_day - np.pi / 2)
    weekly = 1.0 + 0.15 * np.sin(2 * np.pi * t / (7 * per_day))
    seasonal = rng.poisson(np.clip((1000 + 600 * daily) * weekly, 50, None))
    base = rng.lognormal(mean=3.6, sigma=0.35, size=n)
    bursts = (rng.random(n) < 0.04) * rng.lognormal(mean=5.0, sigma=0.6, size=n)
    spiky = np.round(base + np.convolve(bursts, [1.0, 0.6, 0.3], mode="same")).astype(int) + 1
    write_processed("synthetic_seasonal", 10, seasonal)
    write_processed("synthetic_spiky", 10, spiky)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("download", "preprocess"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True, choices=["google", "azure_functions"])
        if name == "preprocess":
            p.add_argument("--intervals", type=int, nargs="*", help="minutes (default: from datasets.yaml)")
    sub.add_parser("synthetic")
    sub.add_parser("list")
    a = ap.parse_args(argv)
    if a.cmd == "download":
        download(a.dataset)
    elif a.cmd == "preprocess":
        preprocess(a.dataset, a.intervals)
    elif a.cmd == "synthetic":
        synthetic()
    elif a.cmd == "list":
        for name, d in config.datasets().items():
            for iv in d["intervals"]:
                p = processed_path(name, iv)
                state = f"{len(pd.read_csv(p))} points" if p.exists() else "not preprocessed yet"
                print(f"{name:20s} {iv:>3d}m  {state}")


if __name__ == "__main__":
    sys.exit(main())
