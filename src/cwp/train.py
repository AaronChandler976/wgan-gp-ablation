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


def warm_up(device: torch.device) -> None:
    """Run one tiny forward and backward pass so the GPU libraries are initialized.

    The first GPU computation in a new process rounds slightly differently from later ones.
    Critic training amplifies that, so a critic run started on its own used to differ from the
    same run started after another one in the same process (as `cwp.launch` does, and as the
    reported runs were made). Warming up first makes both give the later-run result. It draws
    no random numbers.
    """
    if device.type == "cuda":
        weight = torch.ones(4, 4, device=device, requires_grad=True)
        torch.nn.functional.linear(torch.zeros(2, 4, device=device), weight).sum().backward()


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def prepare_data(spec: RunSpec, cfg: dict, device: torch.device) -> dict:
    dc = cfg["data"]
    series = data.load_series(spec.dataset, spec.interval)
    if dc["drop_empty"]:                                        # released code: empty intervals vanish
        series = series[series != 0]
        split = data.compute_split(len(series), tuple(dc["split"]))
    else:
        split = data.get_split(spec.dataset, spec.interval, tuple(dc["split"]))
    bounds = {"train": (0, split.train_end), "val": (split.train_end, split.val_end),
              "test": (split.val_end, split.n)}
    if dc["scaler"] == "train":
        scaler = data.MinMaxScaler.fit(series[: split.train_end])   # training part only
        scalers = {name: scaler for name in bounds}
    elif dc["scaler"] == "per_split":                           # released code; leaks test statistics
        scalers = {name: data.MinMaxScaler.fit(series[s:e]) for name, (s, e) in bounds.items()}
    else:
        raise ValueError("data.scaler must be 'train' or 'per_split'")
    if dc["windows"] not in ("continuous", "inside_split"):
        raise ValueError("data.windows must be 'continuous' or 'inside_split'")
    parts = {}
    for name, (s, e) in bounds.items():
        X, y, idx = data.make_windows(scalers[name].transform(series), dc["window"], s, e,
                                      inside=dc["windows"] == "inside_split")
        parts[name] = (torch.tensor(X, dtype=torch.float32, device=device),
                       torch.tensor(y, dtype=torch.float32, device=device), idx)
    # With the encoding indexed by batch position, a prediction depends on how the data is
    # batched, so evaluation uses the training batch size, in time order, as the released code does.
    by_batch = spec.arch == "transformer" and cfg["model"]["transformer"]["pos_encoding"] == "batch"
    return {"series": series, "split": split, "scaler": scalers["train"], "scalers": scalers,
            "parts": parts, "mase_scale": metrics.naive_scale(series[: split.train_end]),
            "predict_batch": cfg["train"]["batch_size"] if by_batch else 1024}


@torch.no_grad()
def predict(model, X: torch.Tensor, batch: int = 1024) -> np.ndarray:
    model.eval()
    return torch.cat([model(X[i:i + batch]) for i in range(0, X.size(0), batch)]).cpu().numpy()


def evaluate(model, d: dict, part: str):
    """Metrics in original units for one part ('val' or 'test'). Returns (metrics, y, pred, idx)."""
    X, _, idx = d["parts"][part]
    pred = d["scalers"][part].inverse(predict(model, X, d["predict_batch"]).astype(np.float64))
    y = d["series"][idx]
    return metrics.all_metrics(y, pred, d["mase_scale"]), y, pred, idx


def released_units_mape(d: dict, part: str, y: np.ndarray, pred: np.ndarray) -> float:
    """MAPE the way the released code computes it under per-split scaling.

    It refits one scaler on train, test, then validation, and un-scales every part with that
    last (validation) fit. True values and predictions of `part` therefore pass through the
    same wrong affine map, and the result is not in jobs per interval."""
    own, val = d["scalers"][part], d["scalers"]["val"]
    return metrics.mape(val.inverse(own.transform(y)), val.inverse(own.transform(pred)))


def score(model, d: dict) -> dict:
    """Validation and test metrics of the model as it is now. For reporting, never for selection."""
    out = {}
    for part in ("val", "test"):
        m, y, pred, _ = evaluate(model, d, part)
        if d["scalers"]["val"] is not d["scalers"]["train"]:    # per-split scaling
            m["mape_released_units"] = released_units_mape(d, part, y, pred)
        out[part] = m
    return out


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


class Diagnostics:
    """Per-epoch averages of what the critic and the two generator loss terms are doing.

    It only reads values the training step has already computed (plus two cheap gradients with
    respect to the predictions), draws no random numbers and changes no update, so a run gives
    the same model with or without it.
    """

    def __init__(self):
        self.rows, self.sums, self.counts = [], {}, {}

    def add(self, **values) -> None:
        for k, v in values.items():
            self.sums[k] = self.sums.get(k, 0.0) + v.detach()
            self.counts[k] = self.counts.get(k, 0) + 1

    def generator_step(self, pred, pred_loss, adv, adv_weight: float) -> None:
        """Compare the two terms of the generator loss by their gradients on the predictions.

        `adv_to_l1_grad` is the size of the critic term's gradient over the L1 term's. `grad_cosine`
        is +1 when the critic pushes every prediction the same way the L1 loss does, 0 when the
        two are unrelated and -1 when they pull in opposite directions. `critic_slope` is the
        mean absolute slope of the critic score along the predicted value (the L1 loss has slope 1).
        """
        g_l1, = torch.autograd.grad(pred_loss, pred, retain_graph=True)
        g_adv, = torch.autograd.grad(-adv_weight * adv, pred, retain_graph=True)
        n_l1, n_adv = g_l1.norm(), g_adv.norm()
        self.add(l1=pred_loss, adv_term=-adv_weight * adv, l1_grad=n_l1, adv_grad=n_adv,
                 adv_to_l1_grad=n_adv / n_l1.clamp_min(1e-12),
                 grad_cosine=(g_l1 * g_adv).sum() / (n_l1 * n_adv).clamp_min(1e-12),
                 critic_slope=g_adv.abs().mean() * pred.numel() / max(abs(adv_weight), 1e-12))

    def end_epoch(self, epoch: int) -> None:
        self.rows.append({"epoch": epoch, **{k: float(v / self.counts[k]) for k, v in self.sums.items()}})
        self.sums, self.counts = {}, {}


def fit(spec: RunSpec, cfg: dict, device: torch.device, log=print, max_epochs: int | None = None,
        diagnostics: bool = False) -> dict:
    """Train one configuration and return the model, data and training record."""
    warm_up(device)
    set_seed(spec.seed)
    d = prepare_data(spec, cfg, device)
    Xtr, ytr, _ = d["parts"]["train"]
    tc = cfg["train"]
    with_critic = has_critic = spec.critic == "wgan-gp"

    model = build_forecaster(spec.arch, cfg).to(device)
    opt_g = build_optimizer(spec.optimizer, model.parameters(), cfg, with_critic)
    critic = opt_d = None
    if with_critic:
        critic = build_critic(cfg).to(device)
        opt_d = build_optimizer(spec.optimizer, critic.parameters(), cfg, with_critic,
                                lr_scale=tc["critic_lr_scale"])
    loss_fn = {"l1": F.l1_loss, "mse": F.mse_loss}[tc["loss"]]

    if tc["adv_sign"] not in ("minus", "plus") or tc["critic_batch"] not in ("random", "same"):
        raise ValueError("train.adv_sign must be minus|plus and train.critic_batch random|same")
    if tc["critic_rng"] not in ("shared", "own"):
        raise ValueError("train.critic_rng must be shared|own")
    order = torch.Generator().manual_seed(spec.seed)       # batch order, same on every machine
    # "own": critic batches no longer disturb the batch order, so it matches a no-critic run's.
    critic_order = order if tc["critic_rng"] == "shared" else torch.Generator().manual_seed(spec.seed + 1_000_003)
    n, bs = Xtr.size(0), tc["batch_size"]
    epochs = max_epochs or tc["epochs"]
    best = {"val_mape": float("inf"), "epoch": 0, "state": None}
    history, epoch_seconds, stale = [], [], 0
    diag = Diagnostics() if diagnostics else None
    start = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        model.train()
        perm = (torch.randperm(n, generator=order) if tc["shuffle"] else torch.arange(n)).to(device)
        total = 0.0
        # The critic can be switched on late (adv_warmup) and its weight raised gradually (adv_ramp).
        with_critic = has_critic and epoch > tc["adv_warmup"]
        adv_weight = tc["adv_weight"]
        if with_critic and tc["adv_ramp"]:
            adv_weight = adv_weight * min(1.0, (epoch - tc["adv_warmup"]) / tc["adv_ramp"])
        same_batch = with_critic and tc["critic_batch"] == "same"
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            xb, yb = Xtr[b], ytr[b]
            if with_critic:
                if same_batch:       # released code: one forward pass serves the critic and the generator
                    pred = model(xb)
                for p in critic.parameters():
                    p.requires_grad_(True)
                for _ in range(tc["n_critic"]):
                    if same_batch:
                        xr, yr, fake_y = xb, yb, pred.detach()
                    else:
                        j = torch.randint(0, n, (bs,), generator=critic_order).to(device)
                        xr, yr = Xtr[j], ytr[j]
                        with torch.no_grad():
                            fake_y = model(xr)
                    real = torch.cat([xr, yr.unsqueeze(1)], dim=1)
                    fake = torch.cat([xr, fake_y.unsqueeze(1)], dim=1)
                    score_real, score_fake = critic(real).mean(), critic(fake).mean()
                    gp = gradient_penalty(critic, real, fake)
                    d_loss = score_fake - score_real + tc["gp_lambda"] * gp
                    if diag:
                        diag.add(critic_loss=d_loss, wasserstein=score_real - score_fake, gp=gp,
                                 score_real=score_real, score_fake=score_fake)
                    opt_d.zero_grad(set_to_none=True)
                    d_loss.backward()
                    opt_d.step()
                for p in critic.parameters():
                    p.requires_grad_(False)
            if not same_batch:
                pred = model(xb)
            pred_loss = loss_fn(pred, yb)
            loss = pred_loss
            if with_critic:      # the generator also wants the critic to score its sequence highly
                adv = critic(torch.cat([xb, pred.unsqueeze(1)], dim=1)).mean()
                if tc["adv_sign"] == "minus":                   # paper Eq. 4
                    loss = loss - adv_weight * adv
                else:                                           # released code: L1 + critic score
                    loss = loss + adv_weight * adv
                if diag:
                    diag.generator_step(pred, pred_loss, adv,
                                        adv_weight * (1 if tc["adv_sign"] == "minus" else -1))
            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            if diag:
                diag.add(gen_grad_norm=torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                                                      if p.grad is not None)))
            if tc["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt_g.step()
            total += pred_loss.item() * xb.size(0)
        epoch_seconds.append(time.time() - t_epoch)
        if diag:
            diag.end_epoch(epoch)

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

    # Both candidate models are kept so a run can report "best validation epoch" and "last epoch".
    states = {"last": (epoch, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}))}
    if best["state"] is not None:
        states["best_val"] = (best["epoch"], best["state"])
    if tc["select"] == "best_val" and best["state"] is not None:
        model.load_state_dict(best["state"])
        selected = best["epoch"]
    else:
        selected = epoch
    return {"model": model, "data": d, "history": history, "selected_epoch": selected, "states": states,
            "diagnostics": diag.rows if diag else None, "epochs_run": epoch, "train_seconds": time.time() - start, "epoch_seconds": epoch_seconds}


def environment(device: torch.device) -> dict:
    return {"host": platform.node(), "platform": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "git": config.git_info()}


def result_dir(tag: str | None = None, exp: str | None = None):
    """Planned runs: results/. Tagged try-outs: results/scratch/. v2 experiments: results/v2/<exp>/."""
    if exp:
        return config.path("results", "v2", exp)
    return config.path("results", "scratch") if tag else config.path("results")


def run(spec: RunSpec, cfg: dict, device: torch.device, tag: str | None = None,
        tuned: bool = False, log=print, diagnostics: bool = False, exp: str | None = None,
        profile: str | None = None) -> dict:
    """Train, evaluate, and save the checkpoint, test predictions and result file."""
    out = fit(spec, cfg, device, log, diagnostics=diagnostics)
    model, d = out["model"], out["data"]
    by_selection = {}                # reporting only: the same run under both ways of picking an epoch
    chosen = copy.deepcopy(model.state_dict())
    for how, (ep, state) in out["states"].items():
        model.load_state_dict(state)
        by_selection[how] = {"epoch": ep, **score(model, d)}
    model.load_state_dict(chosen)
    scored = score(model, d)
    val_m, test_m = scored["val"], scored["test"]
    _, y, pred, idx = evaluate(model, d, "test")
    name = spec.run_id + (f"__{tag}" if tag else "")
    base = result_dir(tag, exp)
    (base / "predictions").mkdir(parents=True, exist_ok=True)
    config.path("checkpoints").mkdir(exist_ok=True)

    result = {
        "run_id": spec.run_id, "tag": tag, "dataset": spec.dataset, "interval": spec.interval,
        "arch": spec.arch, "critic": spec.critic, "optimizer": spec.optimizer, "seed": spec.seed,
        "experiment": exp, "profile": profile,
        "metrics": {"val": val_m, "test": test_m}, "metrics_by_selection": by_selection,
        "selected_epoch": out["selected_epoch"], "epochs_run": out["epochs_run"],
        "train_seconds": round(out["train_seconds"], 1),
        "inference_ms": time_inference(model, d["parts"]["test"][0], cfg["eval"]["timing_reps"]),
        "n_params": count_params(model),
        "config": cfg, "tuned": tuned, "scaler": d["scaler"].state(),
        "scalers": {k: v.state() for k, v in d["scalers"].items()},
        "split": {"n": d["split"].n, "train_end": d["split"].train_end, "val_end": d["split"].val_end},
        "history": out["history"], **({"diagnostics": out["diagnostics"]} if diagnostics else {}),
        "owner": config.owner(), "env": environment(device),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    torch.save({"model": model.state_dict(), "spec": spec.__dict__, "config": cfg,
                "scaler": d["scaler"].state()},
               config.path("checkpoints", (f"v2-{exp}__" if exp else "") + f"{name}.pt"))
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
    ap.add_argument("--diagnostics", action="store_true",
                    help="also save per-epoch critic and gradient diagnostics (needs --tag or --exp)")
    ap.add_argument("--profile", help="apply configs/profiles/<name>.yaml (needs --tag or --exp)")
    ap.add_argument("--exp", help="save as a run of the named v2 experiment under results/v2/<name>/")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides (need --tag)")
    a = ap.parse_args(argv)

    spec = RunSpec(a.dataset, a.interval, a.arch, a.critic, a.optimizer, a.seed)
    changed = bool(a.overrides or a.lr is not None or a.no_tuned or a.profile)
    if changed and not (a.tag or a.exp):
        sys.exit("Overrides change the configuration, so this would not be the planned run.\n"
                 "Add --tag NAME to save it as an exploratory run (results/scratch/), "
                 "or --exp NAME for a v2 experiment (results/v2/NAME/).")
    if a.diagnostics and not (a.tag or a.exp):
        sys.exit("--diagnostics is for exploratory runs. Add --tag NAME (results/scratch/).")
    if a.exp and not a.exp.replace("_", "").replace("-", "").isalnum():
        sys.exit("--exp must be a simple name (letters, digits, - and _).")
    cfg, tuned = config.resolve_config(spec, a.overrides, a.lr, use_tuned=not a.no_tuned,
                                       profile=a.profile)

    if not data.processed_path(spec.dataset, spec.interval).exists():
        sys.exit(f"No processed data for {spec.data_id}. Run `python -m cwp.data preprocess "
                 f"--dataset {spec.dataset}` after `download`, or pull it from the repo.")

    new_claim = False
    if a.exp:                                   # v2 experiments: additive, no claims, never the plan
        done = result_dir(a.tag, a.exp) / f"{spec.run_id}{'__' + a.tag if a.tag else ''}.json"
        if done.exists() and not a.force:
            sys.exit(f"{done.name} already exists in experiment '{a.exp}'. Use --force to redo it.")
    elif not a.tag:
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
    print(f"{spec.run_id}{' [' + a.tag + ']' if a.tag else ''}{' {' + a.exp + '}' if a.exp else ''} on {device}"
          f"{' (tuned settings)' if tuned else ''}")
    try:
        result = run(spec, cfg, device, tag=a.tag, tuned=tuned, diagnostics=a.diagnostics,
                     exp=a.exp, profile=a.profile)
    except BaseException:
        if new_claim:                                  # do not leave a claim for a run that failed
            runs.claim_path(spec.run_id).unlink(missing_ok=True)
        raise
    t = result["metrics"]["test"]
    print(f"done in {result['train_seconds']:.0f}s, selected epoch {result['selected_epoch']}: "
          f"test MAPE {t['mape']:.2f}%  MAE {t['mae']:.2f}  RMSE {t['rmse']:.2f}  MASE {t['mase']:.3f}  "
          f"inference {result['inference_ms']:.2f} ms")
    if not (a.tag or a.exp):
        print("Share it with the team: python -m cwp.runs sync")


if __name__ == "__main__":
    sys.exit(main())
