"""Phase 2 options: critic variants, warm-up, critic learning rate, and the screening tool."""
import json

import pytest
import torch
import yaml

from cwp import config, screen, train
from cwp.config import RunSpec
from cwp.models import ConditionalCritic, Critic, build_critic, build_forecaster, build_optimizer

from conftest import TINY

SPEC = RunSpec("synthetic_spiky", 10, "lstm", "wgan-gp", "madgrad", 3)
CPU = torch.device("cpu")


def tiny(*extra):
    return config.apply_overrides(config.default_config(), [*TINY, *extra])


def weights(out):
    return list(out["model"].state_dict().values())


@pytest.mark.parametrize("kind", ["mlp", "conditional"])
@pytest.mark.parametrize("norm", ["none", "spectral", "layer"])
@pytest.mark.parametrize("inp", ["levels", "diffs"])
def test_critic_variants_give_one_score_per_sequence_and_train(repo, kind, norm, inp):
    cfg = tiny(f"model.critic.kind={kind}", f"model.critic.norm={norm}", f"model.critic.input={inp}")
    critic = build_critic(cfg)
    seq = torch.randn(6, cfg["data"]["window"] + 1)
    assert critic(seq).shape == (6,)
    gp = train.gradient_penalty(critic, seq, torch.randn_like(seq))
    gp.backward()
    assert torch.isfinite(gp)
    out = train.fit(SPEC, cfg, CPU, log=lambda *_: None)
    assert all(torch.isfinite(w).all() for w in weights(out))


def test_default_critic_keeps_its_initial_weights(repo):
    cfg = tiny()
    torch.manual_seed(5)
    new = build_critic(cfg)
    torch.manual_seed(5)
    h = cfg["model"]["critic"]["hidden"]
    old = torch.nn.Sequential(torch.nn.Linear(13, h), torch.nn.LeakyReLU(0.2),
                              torch.nn.Linear(h, 2 * h), torch.nn.LeakyReLU(0.2), torch.nn.Linear(2 * h, 1))
    assert isinstance(new, Critic)
    for a, b in zip(new.net.state_dict().values(), old.state_dict().values()):
        assert torch.equal(a, b)


def test_diff_critic_ignores_a_constant_shift_and_conditional_sees_the_forecast(repo):
    seq = torch.randn(4, 13)
    diff = build_critic(tiny("model.critic.input=diffs"))
    assert torch.allclose(diff(seq), diff(seq + 3.0), atol=1e-5)
    cond = build_critic(tiny("model.critic.kind=conditional"))
    assert isinstance(cond, ConditionalCritic)
    moved = seq.clone()
    moved[:, -1] += 1.0
    assert not torch.allclose(cond(seq), cond(moved))


def test_spectral_norm_bounds_the_critic_slope(repo):
    critic = build_critic(tiny("model.critic.norm=spectral"))
    for _ in range(30):                                   # power iteration settles in training mode
        critic(torch.randn(8, 13))
    critic.eval()
    a, b = torch.randn(64, 13), torch.randn(64, 13)
    slope = (critic(a) - critic(b)).abs() / (a - b).norm(dim=1)
    assert slope.max() <= 1.05


def test_warmup_trains_like_no_critic_until_the_critic_starts(repo):
    plain = RunSpec(SPEC.dataset, 10, SPEC.arch, "none", SPEC.optimizer, SPEC.seed)
    no_critic = train.fit(plain, tiny(), CPU, log=lambda *_: None)
    # own critic batches + a warm-up longer than the run = exactly the no-critic run
    warm = train.fit(SPEC, tiny("train.adv_warmup=5", "train.critic_rng=own"), CPU, log=lambda *_: None)
    assert all(torch.equal(a, b) for a, b in zip(weights(no_critic), weights(warm)))
    started = train.fit(SPEC, tiny("train.adv_warmup=1", "train.critic_rng=own"), CPU,
                        log=lambda *_: None, diagnostics=True)
    assert not all(torch.equal(a, b) for a, b in zip(weights(no_critic), weights(started)))
    assert "adv_term" not in started["diagnostics"][0] and "adv_term" in started["diagnostics"][1]


def test_ramp_raises_the_critic_weight_gradually(repo):
    out = train.fit(SPEC, tiny("train.epochs=4", "train.adv_ramp=4", "train.adv_weight=0.5"), CPU,
                    log=lambda *_: None, diagnostics=True)
    ratios = [abs(r["adv_grad"]) / max(r["critic_slope"], 1e-12) for r in out["diagnostics"]]
    # adv_grad / critic_slope is proportional to the weight in force: 1/4, 2/4, 3/4, 4/4 of 0.5
    assert ratios[0] < ratios[1] < ratios[2] < ratios[3]
    assert ratios[3] / ratios[0] == pytest.approx(4.0, rel=0.35)


def test_critic_lr_scale_and_weight_decay_reach_the_optimizers(repo):
    cfg = tiny("train.critic_lr_scale=4", "optim.adam.weight_decay=0.01")
    model = build_forecaster("lstm", cfg)
    assert build_optimizer("madgrad", model.parameters(), cfg, True, lr_scale=4).defaults["lr"] == pytest.approx(0.004)
    adam = build_optimizer("adam", model.parameters(), cfg, False)
    assert adam.defaults["weight_decay"] == 0.01 and adam.defaults["betas"] == (0.9, 0.999)
    assert build_optimizer("adam", model.parameters(), cfg, True).defaults["betas"] == (0.0, 0.9)
    a = train.fit(SPEC, tiny(), CPU, log=lambda *_: None)
    b = train.fit(SPEC, tiny("train.critic_lr_scale=4"), CPU, log=lambda *_: None)
    assert not all(torch.equal(p, q) for p, q in zip(weights(a), weights(b)))


def test_lstm_output_dropout_is_off_by_default(repo):
    assert build_forecaster("lstm", tiny()).drop is None
    model = build_forecaster("lstm", tiny("model.lstm.out_dropout=0.5"))
    x = torch.randn(4, 12)
    model.train()
    assert not torch.equal(model(x), model(x))
    model.eval()
    assert torch.equal(model(x), model(x))


def screen_file(repo, **extra):
    spec = {"experiment": "demo_screen", "seeds": [0, 100],
            "fixed": {k: yaml.safe_load(v) for k, v in (o.split("=") for o in TINY)},
            "cells": [{"dataset": "synthetic_spiky", "interval": 10, "arch": "lstm",
                       "critic": "wgan-gp", "optimizer": "madgrad"}],
            "variants": {"base": {}, "small_weight": {"train.adv_weight": 0.01},
                         "fast": {"lr": 0.01}}, **extra}
    p = repo / "screen.yaml"
    p.write_text(yaml.safe_dump(spec, sort_keys=False))
    return p


def test_screen_runs_resumes_and_summarizes_on_validation_only(repo, capsys):
    p = screen_file(repo)
    screen.cmd_run(str(p), "0/2", "cpu")
    folder = repo / "results" / "v2" / "demo_screen" / "trials"
    assert len(list(folder.glob("*.json"))) == 3                     # half of 3 variants x 2 seeds
    screen.cmd_run(str(p), None, "cpu")
    files = sorted(folder.glob("*.json"))
    assert len(files) == 6
    first = json.loads(files[0].read_text())
    assert "val_mape" in first and "test" not in json.dumps(first)   # no test metric is ever stored
    capsys.readouterr()
    screen.cmd_run(str(p), None, "cpu")                              # nothing left to do
    assert capsys.readouterr().out == ""
    screen.cmd_summarize(str(p))
    cell, overall = screen.summarize(screen.load_screen(str(p)))
    assert set(cell["seeds"]) == {2} and len(overall) == 3
    assert overall.set_index("variant").loc["base", "mean_change_pct"] == pytest.approx(0.0)
    assert (repo / "results" / "v2" / "demo_screen" / "screen.md").exists()
    assert not list((repo / "results").glob("*.json"))


def test_screen_refuses_evaluation_seeds(repo):
    p = screen_file(repo, seeds=[0, 1])
    with pytest.raises(SystemExit) as stop:
        screen.load_screen(str(p))
    assert "evaluation seeds" in str(stop.value)
