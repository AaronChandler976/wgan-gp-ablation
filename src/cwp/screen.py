"""Screen variants of a configuration on validation data only.

    python -m cwp.screen run configs/screens/<name>.yaml [--shard 0/4]
    python -m cwp.screen summarize configs/screens/<name>.yaml

A screen file lists cells (dataset + configuration), variants (named sets of overrides) and
tuning seeds. Every cell x variant x seed is one trial: the cell's tuned settings, plus the
variant's overrides, trained with early stopping. Only the best validation MAPE is kept; the
test part is never evaluated here. Each trial is one small file under
results/v2/<experiment>/trials/, so a screen can be stopped, resumed and split over several
processes (--shard). Seeds must be tuning seeds, never the evaluation seeds in plan.yaml.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import config, runs
from .config import RunSpec
from .train import fit, pick_device


def load_screen(path: str) -> dict:
    s = yaml.safe_load(Path(path).read_text())
    missing = {"experiment", "seeds", "cells", "variants"} - set(s)
    if missing:
        raise SystemExit(f"{path}: missing {sorted(missing)}")
    clash = set(s["seeds"]) & set(runs.plan()["seeds"])
    if clash:
        raise SystemExit(f"{path}: seeds {sorted(clash)} are evaluation seeds. Use tuning seeds.")
    s.setdefault("fixed", {})
    s.setdefault("reference", next(iter(s["variants"])))
    return s


def cell_id(c: dict) -> str:
    return RunSpec(c["dataset"], int(c["interval"]), c["arch"], c["critic"], c["optimizer"], 0).run_id[:-4]


def trials(s: dict) -> list[dict]:
    return [{"cell": c, "variant": v, "seed": int(seed)}
            for c in s["cells"] for v in s["variants"] for seed in s["seeds"]]


def trial_path(s: dict, t: dict) -> Path:
    return config.path("results", "v2", s["experiment"], "trials",
                       f"{cell_id(t['cell'])}__{t['variant']}__s{t['seed']}.json")


def run_trial(s: dict, t: dict, device) -> dict:
    c = t["cell"]
    spec = RunSpec(c["dataset"], int(c["interval"]), c["arch"], c["critic"], c["optimizer"], t["seed"])
    overrides = {**s["fixed"], **(c.get("overrides") or {}), **(s["variants"][t["variant"]] or {})}
    lr = overrides.pop("lr", None)                       # `lr` sets the run's learning rate
    cfg, tuned = config.resolve_config(spec, overrides, lr=lr)
    start = time.time()
    out = fit(spec, cfg, device, log=lambda *_: None)
    val = [h["val_mape"] for h in out["history"]]
    return {"experiment": s["experiment"], "cell": cell_id(c), "variant": t["variant"], "seed": t["seed"],
            "overrides": s["variants"][t["variant"]] or {}, "tuned_base": tuned,
            "val_mape": float(min(val)), "best_epoch": out["states"].get("best_val", (0,))[0],
            "epochs_run": out["epochs_run"], "seconds": round(time.time() - start, 1),
            "git": config.git_info()}


def cmd_run(path: str, shard: str | None, device_name: str) -> None:
    s = load_screen(path)
    todo = trials(s)
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        todo = todo[i::n]
    device = pick_device(device_name)
    for k, t in enumerate(todo, 1):
        p = trial_path(s, t)
        if p.exists():
            continue
        r = run_trial(s, t, device)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(r, indent=2) + "\n")
        print(f"[{k}/{len(todo)}] {r['cell']} {r['variant']} s{r['seed']}: val MAPE {r['val_mape']:.3f}% "
              f"(epoch {r['best_epoch']}, {r['seconds']:.0f}s)", flush=True)


def load_trials(s: dict) -> pd.DataFrame:
    rows = [json.loads(trial_path(s, t).read_text()) for t in trials(s) if trial_path(s, t).exists()]
    return pd.DataFrame(rows)


def summarize(s: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per cell and variant: mean validation MAPE over the tuning seeds. Overall: each variant's
    average change against the reference variant (mean of log ratios over cells), best first.
    A trial that diverged counts as it is (a very large value), so unstable variants rank last."""
    df = load_trials(s)
    df["val_mape"] = df["val_mape"].replace([np.inf], 1e9)
    cell = df.groupby(["cell", "variant"], sort=False).agg(
        seeds=("seed", "count"), val_mape=("val_mape", "mean"),
        seed_gap=("val_mape", lambda x: x.max() - x.min()), seconds=("seconds", "mean")).reset_index()
    ref = cell[cell["variant"] == s["reference"]].set_index("cell")["val_mape"]
    cell["vs_reference_pct"] = (cell["val_mape"] / cell["cell"].map(ref) - 1) * 100
    cell["log_ratio"] = np.log(cell["val_mape"] / cell["cell"].map(ref))
    overall = cell.groupby("variant", sort=False).agg(
        cells=("cell", "count"), mean_log_ratio=("log_ratio", "mean"),
        worst_cell_pct=("vs_reference_pct", "max"), best_cell_pct=("vs_reference_pct", "min")).reset_index()
    overall["mean_change_pct"] = (np.exp(overall["mean_log_ratio"]) - 1) * 100
    overall = overall.sort_values("mean_log_ratio").drop(columns="mean_log_ratio")
    return cell.drop(columns="log_ratio"), overall


def cmd_summarize(path: str) -> None:
    s = load_screen(path)
    n_all = len(trials(s))
    cell, overall = summarize(s)
    done = int(cell["seeds"].sum())
    out = config.path("results", "v2", s["experiment"])
    md = [f"# Screen: {s['experiment']}", "",
          f"{done} of {n_all} trials finished. Validation MAPE (%) only; the test part is not used. "
          f"Tuning seeds {s['seeds']}, settings while screening: {s['fixed']}. "
          f"Changes are against the variant `{s['reference']}` in the same cell.", ""]
    if s.get("description"):
        md += [s["description"].strip(), ""]
    md += ["## Variants, best first (average over cells)", "", config.md_table(overall.round(3)), "",
           "## Every cell", "", config.md_table(cell.round(3)), ""]
    (out / "screen.md").write_text("\n".join(md))
    cell.to_csv(out / "screen_cells.csv", index=False)
    overall.to_csv(out / "screen_variants.csv", index=False)
    print("\n".join(md))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("screen")
    r.add_argument("--shard", help="i/n: run every n-th trial starting at i (to share the GPU)")
    r.add_argument("--device", default="auto")
    sm = sub.add_parser("summarize")
    sm.add_argument("screen")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        cmd_run(a.screen, a.shard, a.device)
    else:
        cmd_summarize(a.screen)


if __name__ == "__main__":
    sys.exit(main())
