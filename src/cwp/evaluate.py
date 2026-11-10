"""Evaluate finished runs and compare configurations.

    python -m cwp.evaluate baselines            # naive / seasonal-naive / moving-average errors
    python -m cwp.evaluate compare              # mean and spread over seeds, paired tests
    python -m cwp.evaluate timing               # re-time every local checkpoint on this machine
    python -m cwp.evaluate dm RUN_ID_A RUN_ID_B # Diebold-Mariano test on two runs' predictions
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from . import config, data, metrics, runs
from .config import RunSpec

FACTORS = {"arch": ("lstm", "transformer"), "critic": ("none", "wgan-gp"),
           "optimizer": ("adam", "madgrad")}          # (without the part, with the part)
METRICS = ("mape", "mae", "rmse", "mase")


# --------------------------------------------------------------------------- baselines

def baseline_table(dataset: str, interval: int) -> pd.DataFrame:
    series = data.load_series(dataset, interval)
    split = data.get_split(dataset, interval)
    scale = metrics.naive_scale(series[: split.train_end])
    rows = []
    for part, (s, e) in {"val": (split.train_end, split.val_end), "test": (split.val_end, split.n)}.items():
        for name, pred in metrics.baseline_predictions(series, s, e, 1440 // interval).items():
            m = metrics.all_metrics(series[s:e], pred, scale)
            rows.append({"part": part, "baseline": name, **{k: m[k] for k in METRICS}})
    return pd.DataFrame(rows)


def cmd_baselines() -> None:
    out = config.path("results", "baselines")
    out.mkdir(parents=True, exist_ok=True)
    for d in runs.plan()["datasets"]:
        name, iv = d["dataset"], int(d["interval"])
        if not data.processed_path(name, iv).exists():
            print(f"{name}-{iv}m: not preprocessed yet, skipped")
            continue
        df = baseline_table(name, iv).round(4)
        df.to_json(out / f"{name}-{iv}m.json", orient="records", indent=2)
        print(f"\n{name}-{iv}m\n{df.to_string(index=False)}")


# --------------------------------------------------------------------------- compare

def load_results() -> pd.DataFrame:
    rows = []
    for f in sorted(config.path("results").glob("*.json")):
        r = json.loads(f.read_text())
        spec = RunSpec.from_run_id(r["run_id"])
        rows.append({"data_id": spec.data_id, "config_id": spec.config_id, "arch": r["arch"],
                     "critic": r["critic"], "optimizer": r["optimizer"], "seed": r["seed"],
                     **{k: r["metrics"]["test"][k] for k in METRICS},
                     "val_mape": r["metrics"]["val"]["mape"], "train_seconds": r["train_seconds"],
                     "inference_ms": r["inference_ms"], "host": r["env"]["host"], "owner": r["owner"]})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    def pm(x):
        return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"
    g = df.groupby(["data_id", "config_id"], sort=False)
    out = g.agg(seeds=("seed", "count"), **{m: (m, pm) for m in METRICS},
                train_min=("train_seconds", lambda s: round(s.mean() / 60, 1))).reset_index()
    return out


def one_change_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Paired tests for every two configurations that differ in exactly one switch."""
    rows = []
    for data_id, d in df.groupby("data_id", sort=False):
        table = d.pivot_table(index="seed", columns=["arch", "critic", "optimizer"], values="mape")
        block = []
        for a, b in itertools.combinations(table.columns, 2):
            diff = [i for i in range(3) if a[i] != b[i]]
            if len(diff) != 1:
                continue
            factor = list(FACTORS)[diff[0]]
            if a[diff[0]] != FACTORS[factor][0]:      # `b` is the one with the part switched on
                a, b = b, a
            pair = table[[a, b]].dropna()
            if len(pair) < 2:
                continue
            delta = (pair[b] - pair[a]).to_numpy()
            p_t = float(stats.ttest_rel(pair[b], pair[a]).pvalue) if np.ptp(delta) > 0 else float("nan")
            p_w = float(stats.wilcoxon(delta).pvalue) if len(delta) >= 6 and np.any(delta) else float("nan")
            block.append({"data_id": data_id, "switch": f"{factor}: {a[diff[0]]} -> {b[diff[0]]}",
                          "held_fixed": "-".join(x for i, x in enumerate(a) if i != diff[0]),
                          "seeds": len(delta), "mape_change_points": delta.mean(),
                          "improved_in": f"{int((delta < 0).sum())}/{len(delta)}",
                          "p_paired_t": p_t, "p_wilcoxon": p_w})
        ps = [r["p_paired_t"] for r in block]
        if block and not np.any(np.isnan(ps)):
            for r, adj in zip(block, metrics.holm(ps)):
                r["p_holm"] = adj
        rows += block
    return pd.DataFrame(rows)


def factorial_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Main effects and interactions on log MAPE. Needs all 8 configurations for a seed."""
    rows = []
    for data_id, d in df.groupby("data_id", sort=False):
        table = d.pivot_table(index="seed", columns=["arch", "critic", "optimizer"], values="mape")
        if table.shape[1] < 8:
            continue
        logm = np.log(table.dropna())
        if len(logm) < 2:
            continue
        names = list(FACTORS)
        for k in (1, 2, 3):
            for combo in itertools.combinations(range(3), k):
                sign = np.array([np.prod([1 if col[i] == FACTORS[names[i]][1] else -1 for i in combo])
                                 for col in logm.columns])
                per_seed = (logm.to_numpy() * sign).sum(axis=1) / 4.0
                p = float(stats.ttest_1samp(per_seed, 0.0).pvalue) if np.ptp(per_seed) > 0 else float("nan")
                rows.append({"data_id": data_id, "effect": " x ".join(names[i] for i in combo),
                             "kind": "main effect" if k == 1 else "interaction", "seeds": len(per_seed),
                             "log_mape_effect": per_seed.mean(),
                             "mape_change_pct": (np.exp(per_seed.mean()) - 1) * 100 if k == 1 else np.nan,
                             "p": p})
    return pd.DataFrame(rows)


def cmd_compare() -> None:
    df = load_results()
    if df.empty:
        sys.exit("No finished runs in results/ yet.")
    out = config.path("results", "summary")
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(df)
    pairs = one_change_pairs(df).round(4)
    effects = factorial_effects(df).round(4)
    paper = config.load_yaml("configs", "paper_reference.yaml")

    md = ["# Results summary", "",
          "Test-set metrics, mean ± standard deviation over seeds. MAPE in %, MAE and RMSE in jobs "
          "per interval. Generated by `python -m cwp.evaluate compare`.", ""]
    for data_id, block in summary.groupby("data_id", sort=False):
        md += [f"## {data_id}", "", config.md_table(block.drop(columns="data_id")), ""]
        dataset, interval = data_id.rsplit("-", 1)
        if data.processed_path(dataset, int(interval[:-1])).exists():
            base = baseline_table(dataset, int(interval[:-1]))
            base = base[base["part"] == "test"].drop(columns="part").round(3)
            md += ["Baselines on the same test points:", "", config.md_table(base), ""]
        if data_id in paper:
            md += [f"Paper (Table 3) test MAPE: LoadDynamics LSTM {paper[data_id][0]}, "
                   f"WGAN-gp Transformer {paper[data_id][1]}.", ""]
    if not pairs.empty:
        md += ["## One-change comparisons", "",
               "Each row switches one part on and keeps the other two fixed, paired by seed. "
               "A negative change means the part lowered MAPE.", "", config.md_table(pairs), ""]
        pairs.to_csv(out / "one_change_comparisons.csv", index=False)
    if not effects.empty:
        md += ["## Factorial effects (all 8 configurations)", "",
               "Average effect of each part on log MAPE across the other settings, and whether "
               "parts interact.", "", config.md_table(effects), ""]
        effects.to_csv(out / "factorial_effects.csv", index=False)
    df.to_csv(out / "all_runs.csv", index=False)
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"saved to {out}")


# --------------------------------------------------------------------------- timing, DM test

def cmd_timing(device_name: str) -> None:
    """Inference speed is only comparable on one machine, so re-time all checkpoints here."""
    import torch
    from .models import build_forecaster
    from .train import pick_device, prepare_data, time_inference
    device = pick_device(device_name)
    rows = []
    for f in sorted(config.path("checkpoints").glob("*.pt")):
        ck = torch.load(f, map_location=device, weights_only=False)
        spec = RunSpec(**ck["spec"])
        model = build_forecaster(spec.arch, ck["config"]).to(device)
        model.load_state_dict(ck["model"])
        X = prepare_data(spec, ck["config"], device)["parts"]["test"][0]
        rows.append({"run": f.stem, "config_id": spec.config_id, "data_id": spec.data_id,
                     "inference_ms": time_inference(model, X, ck["config"]["eval"]["timing_reps"])})
    if not rows:
        sys.exit("No checkpoints in checkpoints/. Timing needs the .pt files on this machine.")
    df = pd.DataFrame(rows)
    out = config.path("results", "summary", "inference_timing.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"device: {device}")
    print(df.groupby(["data_id", "config_id"])["inference_ms"].agg(["count", "mean", "std"]).round(3))
    print(f"saved to {out}")


def cmd_dm(a: str, b: str) -> None:
    pa = pd.read_csv(config.path("results", "predictions", f"{a}.csv"))
    pb = pd.read_csv(config.path("results", "predictions", f"{b}.csv"))
    if not pa["t"].equals(pb["t"]):
        sys.exit("The two runs were scored on different test points.")
    r = metrics.diebold_mariano(pa["y_true"], pa["y_pred"], pb["y_pred"])
    better = a if r["stat"] < 0 else b
    print(f"Diebold-Mariano on {r['n']} test points: statistic {r['stat']:.3f}, p = {r['p']:.4f}")
    print(f"smaller absolute error: {better}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baselines")
    sub.add_parser("compare")
    t = sub.add_parser("timing")
    t.add_argument("--device", default="auto")
    dm = sub.add_parser("dm")
    dm.add_argument("run_a")
    dm.add_argument("run_b")
    a = ap.parse_args(argv)
    if a.cmd == "baselines":
        cmd_baselines()
    elif a.cmd == "compare":
        cmd_compare()
    elif a.cmd == "timing":
        cmd_timing(a.device)
    elif a.cmd == "dm":
        cmd_dm(a.run_a, a.run_b)


if __name__ == "__main__":
    sys.exit(main())
