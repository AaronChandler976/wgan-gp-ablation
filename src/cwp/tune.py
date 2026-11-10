"""Hyperparameter search with the same grid and budget for every configuration.

    python -m cwp.tune --dataset google --interval 10 --arch lstm --critic none --optimizer adam
    python -m cwp.tune --plan        # every dataset x configuration in the plan that is not tuned yet

The grid is in configs/tuning.yaml. The setting with the lowest validation MAPE is written to
configs/tuned/<dataset>-<interval>m__<config>.yaml, and `python -m cwp.train` then uses it.
Tuning uses its own seed (plan.yaml: tuning_seed), never an evaluation seed, and never the test set.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from datetime import date

import yaml

from . import config, runs
from .config import ARCHS, CRITICS, OPTIMIZERS, RunSpec
from .train import fit, pick_device


def tune(spec: RunSpec, device, epochs: int | None = None) -> dict:
    t = config.load_yaml("configs", "tuning.yaml")
    epochs = epochs or t["epochs"]
    keys = list(t["grid"])
    trials = []
    for values in itertools.product(*(t["grid"][k] for k in keys)):
        setting = dict(zip(keys, values))
        overrides = {k: v for k, v in setting.items() if k != "lr"}
        cfg, _ = config.resolve_config(spec, {**overrides, "train.epochs": epochs},
                                       lr=setting.get("lr"), use_tuned=False)
        out = fit(spec, cfg, device, log=lambda *_: None)
        val = min(h["val_mape"] for h in out["history"])
        trials.append({"setting": setting, "val_mape": val})
        print(f"  {setting}  ->  val MAPE {val:.3f}%", flush=True)
    best = min(trials, key=lambda r: r["val_mape"])
    tuned = {"lr": best["setting"].get("lr"),
             "overrides": {k: v for k, v in best["setting"].items() if k != "lr"},
             "val_mape": round(best["val_mape"], 4), "epochs": epochs, "trials": trials,
             "tuned_by": config.owner(), "date": date.today().isoformat()}
    p = config.tuned_path(spec)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(tuned, sort_keys=False))
    print(f"best: {best['setting']} (val MAPE {best['val_mape']:.3f}%) -> {p.name}")
    return tuned


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="tune everything in the plan that has no tuned file")
    ap.add_argument("--dataset")
    ap.add_argument("--interval", type=int)
    ap.add_argument("--arch", choices=ARCHS)
    ap.add_argument("--critic", choices=CRITICS)
    ap.add_argument("--optimizer", choices=OPTIMIZERS)
    ap.add_argument("--epochs", type=int, help="override the epochs in configs/tuning.yaml")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    seed = runs.plan()["tuning_seed"]
    device = pick_device(a.device)

    if a.plan:
        seen = {}
        for s in runs.plan_specs():
            seen.setdefault((s.data_id, s.config_id),
                            RunSpec(s.dataset, s.interval, s.arch, s.critic, s.optimizer, seed))
        specs = [s for s in seen.values() if not config.tuned_path(s).exists()]
    else:
        missing = [k for k in ("dataset", "interval", "arch", "critic", "optimizer") if getattr(a, k) is None]
        if missing:
            sys.exit(f"missing: {', '.join('--' + m for m in missing)} (or use --plan)")
        specs = [RunSpec(a.dataset, a.interval, a.arch, a.critic, a.optimizer, seed)]
    for s in specs:
        print(f"tuning {s.data_id} {s.config_id} on {device}")
        tune(s, device, a.epochs)
    if not specs:
        print("everything in the plan is already tuned")


if __name__ == "__main__":
    sys.exit(main())
