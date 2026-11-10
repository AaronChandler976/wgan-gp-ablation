"""Estimate how long training takes on this machine.

    python -m cwp.bench                      # times the 4 kinds of run for a few epochs
    python -m cwp.bench --epochs 5 --device cpu

It trains each kind of run (LSTM or Transformer, with or without the critic) for a few
epochs, then scales the time per epoch to a full run and to the runs still left in the plan.
The optimizer hardly changes the speed, so it is not timed separately.
"""
from __future__ import annotations

import argparse
import platform
import statistics
import sys

from . import config, data, runs
from .config import RunSpec
from .train import fit, pick_device

KINDS = [("lstm", "none"), ("lstm", "wgan-gp"), ("transformer", "none"), ("transformer", "wgan-gp")]


def bench_dataset() -> tuple[str, int]:
    """The first plan dataset that is preprocessed, else the synthetic series."""
    for d in runs.plan()["datasets"]:
        if data.processed_path(d["dataset"], int(d["interval"])).exists():
            return d["dataset"], int(d["interval"])
    return "synthetic_seasonal", 10


def train_windows(dataset: str, interval: int, window: int) -> int | None:
    if not data.processed_path(dataset, interval).exists():
        return None
    return data.get_split(dataset, interval).train_end - window


def measure(dataset: str, interval: int, device, epochs: int = 3) -> dict:
    """Seconds per epoch for each kind of run (the first epoch is warm-up and is dropped)."""
    out = {}
    for arch, critic in KINDS:
        spec = RunSpec(dataset, interval, arch, critic, "madgrad", 0)
        cfg, _ = config.resolve_config(spec, {"train.eval_every": 10 ** 9}, use_tuned=False)
        rec = fit(spec, cfg, device, log=lambda *_: None, max_epochs=epochs + 1)
        out[(arch, critic)] = {"sec_per_epoch": statistics.median(rec["epoch_seconds"][1:]),
                               "epochs": cfg["train"]["epochs"], "window": cfg["data"]["window"]}
    return out


def estimate_plan(timing: dict, dataset: str, interval: int, specs=None) -> dict:
    """Total seconds for the unfinished, unclaimed runs in the plan (or for `specs`)."""
    specs = runs.todo() if specs is None else specs
    total, assumed = 0.0, set()
    for s in specs:
        t = timing[(s.arch, s.critic)]
        base = train_windows(dataset, interval, t["window"])
        mine = train_windows(s.dataset, s.interval, t["window"])
        if mine is None:               # not preprocessed yet: assume the benchmark dataset's size
            mine = base
            assumed.add(s.data_id)
        total += t["sec_per_epoch"] * t["epochs"] * mine / base
    return {"runs": len(specs), "seconds": total, "assumed_size_for": sorted(assumed)}


def fmt(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def report(device, epochs: int) -> dict:
    import torch
    dataset, interval = bench_dataset()
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Timing on {platform.node()} ({name}) with {dataset}-{interval}m ...", flush=True)
    timing = measure(dataset, interval, device, epochs)
    print(f"\n{'kind of run':28s}{'s / epoch':>10s}   full run")
    for (arch, critic), t in timing.items():
        label = arch + (" + critic" if critic == "wgan-gp" else "")
        print(f"{label:28s}{t['sec_per_epoch']:>10.3f}   {fmt(t['sec_per_epoch'] * t['epochs'])} "
              f"({t['epochs']} epochs)")
    plan = estimate_plan(timing, dataset, interval)
    print(f"\nRuns left in the plan: {plan['runs']}  ->  about {fmt(plan['seconds'])} one after another")
    if plan["assumed_size_for"]:
        print(f"({', '.join(plan['assumed_size_for'])} not preprocessed yet: assumed the same size "
              f"as {dataset}-{interval}m)")
    print("Tuning runs are extra. Several runs can share one GPU, which shortens the wall time.")
    return {"timing": timing, "plan": plan, "dataset": (dataset, interval)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=3, help="timed epochs per kind (default 3)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    report(pick_device(a.device), a.epochs)


if __name__ == "__main__":
    sys.exit(main())
