"""Train one run. The three switches pick one of the 8 configurations:

    python -m cwp.train --dataset google --interval 10 \\
        --arch lstm|transformer --critic none|wgan-gp --optimizer adam|madgrad --seed 1

Planned runs use the default config (plus configs/tuned/ if present) so everyone trains the
same thing. To try other settings, add `key=value` overrides and a `--tag`; tagged runs are
saved under results/scratch/ and do not count toward the plan.

    python -m cwp.train ... --tag quick train.epochs=20 model.transformer.d_model=32
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import config, data, metrics, runs
from .config import ARCHS, CRITICS, OPTIMIZERS, RunSpec
from .models import build_critic, build_forecaster, build_optimizer, count_params


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # TF32 changes float results on newer GPUs; off so machines agree as closely as possible.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def prepare_data(spec: RunSpec, cfg: dict, device: torch.device) -> dict:
    series = data.load_series(spec.dataset, spec.interval)
    split = data.get_split(spec.dataset, spec.interval, tuple(cfg["data"]["split"]))
    scaler = data.MinMaxScaler.fit(series[: split.train_end])   # training part only
    scaled = scaler.transform(series)
    bounds = {"train": (0, split.train_end), "val": (split.train_end, split.val_end),
              "test": (split.val_end, split.n)}
    parts = {}
    for name, (s, e) in bounds.items():
        X, y, idx = data.make_windows(scaled, cfg["data"]["window"], s, e)
        parts[name] = (torch.tensor(X, dtype=torch.float32, device=device),
                       torch.tensor(y, dtype=torch.float32, device=device), idx)
    return {"series": series, "split": split, "scaler": scaler, "parts": parts,
            "mase_scale": metrics.naive_scale(series[: split.train_end])}


@torch.no_grad()
def predict(model, X: torch.Tensor, batch: int = 1024) -> np.ndarray:
    model.eval()
    return torch.cat([model(X[i:i + batch]) for i in range(0, X.size(0), batch)]).cpu().numpy()


def evaluate(model, d: dict, part: str):
    """Metrics in original units for one part ('val' or 'test'). Returns (metrics, y, pred, idx)."""
    X, _, idx = d["parts"][part]
    pred = d["scaler"].inverse(predict(model, X).astype(np.float64))
    y = d["series"][idx]
    return metrics.all_metrics(y, pred, d["mase_scale"]), y, pred, idx


def gradient_penalty(critic, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """WGAN-gp penalty: push the critic's gradient norm toward 1 on points between real and fake."""
    alpha = torch.rand(real.size(0), 1, device=real.device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    grad, = torch.autograd.grad(critic(interp).sum(), interp, create_graph=True)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


@torch.no_grad()
def time_inference(model, X: torch.Tensor, reps: int) -> float:
    """Median milliseconds for one prediction (batch of 1). Depends on the machine."""
    model.eval()
    sync = torch.cuda.synchronize if X.is_cuda else (lambda: None)
    times = []
    for i in range(reps + 20):
        x = X[i % X.size(0)].unsqueeze(0)
        sync()
        t0 = time.perf_counter()
        model(x)
        sync()
        if i >= 20:                                   # the first 20 are warm-up
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def fit(spec: RunSpec, cfg: dict, device: torch.device, log=print, max_epochs: int | None = None) -> dict:
    """Train one configuration and return the model, data and training record."""
    set_seed(spec.seed)
    d = prepare_data(spec, cfg, device)
    Xtr, ytr, _ = d["parts"]["train"]
    tc = cfg["train"]
    with_critic = spec.critic == "wgan-gp"

    model = build_forecaster(spec.arch, cfg).to(device)
    opt_g = build_optimizer(spec.optimizer, model.parameters(), cfg, with_critic)
    critic = opt_d = None
    if with_critic:
        critic = build_critic(cfg).to(device)
        opt_d = build_optimizer(spec.optimizer, critic.parameters(), cfg, with_critic)
    loss_fn = {"l1": F.l1_loss, "mse": F.mse_loss}[tc["loss"]]

    order = torch.Generator().manual_seed(spec.seed)       # batch order, same on every machine
    n, bs = Xtr.size(0), tc["batch_size"]
    epochs = max_epochs or tc["epochs"]
    best = {"val_mape": float("inf"), "epoch": 0, "state": None}
    history, epoch_seconds, stale = [], [], 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        model.train()
        perm = torch.randperm(n, generator=order).to(device)
        total = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            xb, yb = Xtr[b], ytr[b]
            if with_critic:
                for p in critic.parameters():
                    p.requires_grad_(True)
                for _ in range(tc["n_critic"]):
                    j = torch.randint(0, n, (bs,), generator=order).to(device)
                    xr, yr = Xtr[j], ytr[j]
                    with torch.no_grad():
                        fake_y = model(xr)
                    real = torch.cat([xr, yr.unsqueeze(1)], dim=1)
                    fake = torch.cat([xr, fake_y.unsqueeze(1)], dim=1)
                    d_loss = (critic(fake).mean() - critic(real).mean()
                              + tc["gp_lambda"] * gradient_penalty(critic, real, fake))
                    opt_d.zero_grad(set_to_none=True)
                    d_loss.backward()
                    opt_d.step()
                for p in critic.parameters():
                    p.requires_grad_(False)
            pred = model(xb)
            pred_loss = loss_fn(pred, yb)
            loss = pred_loss
            if with_critic:      # the generator also wants the critic to score its sequence highly
                loss = loss - tc["adv_weight"] * critic(torch.cat([xb, pred.unsqueeze(1)], dim=1)).mean()
            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            if tc["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt_g.step()
            total += pred_loss.item() * xb.size(0)
        epoch_seconds.append(time.time() - t_epoch)

        if epoch % tc["eval_every"] == 0 or epoch == epochs:
            val = evaluate(model, d, "val")[0]["mape"]
            val = val if np.isfinite(val) else float("inf")
            history.append({"epoch": epoch, "train_loss": total / n, "val_mape": val})
            log(f"epoch {epoch:5d}  train {tc['loss']} {total / n:.5f}  val MAPE {val:.3f}%")
            if val < best["val_mape"] or best["state"] is None:
                best = {"val_mape": val, "epoch": epoch,
                        "state": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})}
                stale = 0
            else:
                stale += tc["eval_every"]
            if tc["patience"] and stale >= tc["patience"]:
                log(f"stopped early at epoch {epoch} (no improvement for {stale} epochs)")
                break

    if tc["select"] == "best_val" and best["state"] is not None:
        model.load_state_dict(best["state"])
        selected = best["epoch"]
    else:
        selected = epoch
    return {"model": model, "data": d, "history": history, "selected_epoch": selected,
            "epochs_run": epoch, "train_seconds": time.time() - start, "epoch_seconds": epoch_seconds}


def environment(device: torch.device) -> dict:
    return {"host": platform.node(), "platform": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "git": config.git_info()}


def run(spec: RunSpec, cfg: dict, device: torch.device, tag: str | None = None,
        tuned: bool = False, log=print) -> dict:
    """Train, evaluate, and save the checkpoint, test predictions and result file."""
    out = fit(spec, cfg, device, log)
    model, d = out["model"], out["data"]
    val_m = evaluate(model, d, "val")[0]
    test_m, y, pred, idx = evaluate(model, d, "test")
    name = spec.run_id + (f"__{tag}" if tag else "")
    base = config.path("results", "scratch") if tag else config.path("results")
    (base / "predictions").mkdir(parents=True, exist_ok=True)
    config.path("checkpoints").mkdir(exist_ok=True)

    result = {
        "run_id": spec.run_id, "tag": tag, "dataset": spec.dataset, "interval": spec.interval,
        "arch": spec.arch, "critic": spec.critic, "optimizer": spec.optimizer, "seed": spec.seed,
        "metrics": {"val": val_m, "test": test_m},
        "selected_epoch": out["selected_epoch"], "epochs_run": out["epochs_run"],
        "train_seconds": round(out["train_seconds"], 1),
        "inference_ms": time_inference(model, d["parts"]["test"][0], cfg["eval"]["timing_reps"]),
        "n_params": count_params(model),
        "config": cfg, "tuned": tuned, "scaler": d["scaler"].state(),
        "split": {"n": d["split"].n, "train_end": d["split"].train_end, "val_end": d["split"].val_end},
        "history": out["history"], "owner": config.owner(), "env": environment(device),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    torch.save({"model": model.state_dict(), "spec": spec.__dict__, "config": cfg,
                "scaler": d["scaler"].state()}, config.path("checkpoints", f"{name}.pt"))
    pd.DataFrame({"t": idx, "y_true": y, "y_pred": pred}).to_csv(
        base / "predictions" / f"{name}.csv", index=False)
    (base / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--interval", type=int, required=True, help="minutes")
    ap.add_argument("--arch", required=True, choices=ARCHS)
    ap.add_argument("--critic", required=True, choices=CRITICS)
    ap.add_argument("--optimizer", required=True, choices=OPTIMIZERS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, help="learning rate for the chosen optimizer (needs --tag)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--tag", help="save as an exploratory run under results/scratch/")
    ap.add_argument("--no-tuned", action="store_true", help="ignore configs/tuned/ (needs --tag)")
    ap.add_argument("--force", action="store_true", help="retrain a finished or claimed run")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides (need --tag)")
    a = ap.parse_args(argv)

    spec = RunSpec(a.dataset, a.interval, a.arch, a.critic, a.optimizer, a.seed)
    changed = bool(a.overrides or a.lr is not None or a.no_tuned)
    if changed and not a.tag:
        sys.exit("Overrides change the configuration, so this would not be the planned run.\n"
                 "Add --tag NAME to save it as an exploratory run (results/scratch/).")
    cfg, tuned = config.resolve_config(spec, a.overrides, a.lr, use_tuned=not a.no_tuned)

    if not data.processed_path(spec.dataset, spec.interval).exists():
        sys.exit(f"No processed data for {spec.data_id}. Run `python -m cwp.data preprocess "
                 f"--dataset {spec.dataset}` after `download`, or pull it from the repo.")

    new_claim = False
    if not a.tag:
        st, info = runs.state(spec)
        if st == "done" and not a.force:
            sys.exit(f"{spec.run_id} is already finished ({info['owner']}, "
                     f"test MAPE {info['metrics']['test']['mape']:.2f}%). Use --force to redo it.")
        if st == "claimed" and info["owner"] != config.owner() and not a.force:
            sys.exit(f"{spec.run_id} is claimed by {info['owner']}. Pick another run "
                     f"(`python -m cwp.runs next`) or use --force.")
        if st == "todo":
            runs.write_claim(spec)
            new_claim = True

    device = pick_device(a.device)
    print(f"{spec.run_id}{' [' + a.tag + ']' if a.tag else ''} on {device}"
          f"{' (tuned settings)' if tuned else ''}")
    try:
        result = run(spec, cfg, device, tag=a.tag, tuned=tuned)
    except BaseException:
        if new_claim:                                  # do not leave a claim for a run that failed
            runs.claim_path(spec.run_id).unlink(missing_ok=True)
        raise
    t = result["metrics"]["test"]
    print(f"done in {result['train_seconds']:.0f}s, best epoch {result['selected_epoch']}: "
          f"test MAPE {t['mape']:.2f}%  MAE {t['mae']:.2f}  RMSE {t['rmse']:.2f}  MASE {t['mase']:.3f}  "
          f"inference {result['inference_ms']:.2f} ms")
    if not a.tag:
        print("Share it with the team: python -m cwp.runs sync")


if __name__ == "__main__":
    sys.exit(main())
