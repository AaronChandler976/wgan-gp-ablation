"""The released-code switches (configs/profiles/released_code.yaml) and v2 experiment folders."""
import json

import numpy as np
import pytest
import torch

from cwp import config, data, train
from cwp.config import RunSpec
from cwp.models import PointwiseTanhCritic, PositionalEncoding, build_critic, build_forecaster

from conftest import TINY

SPEC = RunSpec("synthetic_spiky", 10, "transformer", "wgan-gp", "madgrad", 1)
CPU = torch.device("cpu")


def tiny(*extra):
    return config.apply_overrides(config.default_config(), [*TINY, *extra])


def test_windows_inside_a_split_skip_its_first_points():
    series = np.arange(100.0)
    _, _, across = data.make_windows(series, 10, 60, 80)
    X, y, inside = data.make_windows(series, 10, 60, 80, inside=True)
    assert across[0] == 60 and inside[0] == 70 and inside[-1] == 79
    assert X[0][0] == 60 and y[0] == 70                    # history starts inside the part


def test_per_split_scaler_fits_each_part_on_its_own(repo):
    d = train.prepare_data(SPEC, tiny("data.scaler=per_split", "data.windows=inside_split"), CPU)
    for part in ("train", "val", "test"):                  # every part spans exactly (-1, 1)
        X, y, _ = d["parts"][part]
        assert min(X.min(), y.min()).item() == pytest.approx(-1.0)
        assert max(X.max(), y.max()).item() == pytest.approx(1.0)
    assert d["scalers"]["test"].state() != d["scalers"]["train"].state()
    one = train.prepare_data(SPEC, tiny(), CPU)
    assert one["scalers"]["test"] is one["scalers"]["train"]


def test_released_units_mape_uses_the_validation_scaler(repo):
    d = train.prepare_data(SPEC, tiny("data.scaler=per_split"), CPU)
    y = d["series"][d["parts"]["test"][2]]
    assert train.released_units_mape(d, "val", y, y * 1.1) > 0
    own, val = d["scalers"]["test"], d["scalers"]["val"]
    expected = np.mean(np.abs(0.1 * y * val.span / own.span /
                              ((y - own.lo) * val.span / own.span + val.lo))) * 100
    assert train.released_units_mape(d, "test", y, y * 1.1) == pytest.approx(expected)


def test_drop_empty_removes_zero_intervals(repo):
    p = data.processed_path("synthetic_spiky", 10)
    series = data.load_series("synthetic_spiky", 10)
    series[5:9] = 0
    data.write_processed("synthetic_spiky", 10, series.astype(int))
    kept = train.prepare_data(SPEC, tiny(), CPU)
    dropped = train.prepare_data(SPEC, tiny("data.drop_empty=true"), CPU)
    assert len(kept["series"]) == len(series) and len(dropped["series"]) == len(series) - 4
    assert p.exists() and (dropped["series"] != 0).all()


def test_pointwise_critic_scores_every_value_on_its_own(repo):
    critic = build_critic(tiny("model.critic.kind=pointwise_tanh"))
    assert isinstance(critic, PointwiseTanhCritic)
    seq = torch.randn(5, 13)
    out = critic(seq)
    assert out.shape == (5, 13) and out.abs().max() <= 1          # one bounded score per value
    changed = seq.clone()
    changed[:, -1] += 1.0
    assert torch.equal(critic(changed)[:, :-1], out[:, :-1])      # other values' scores do not move


def test_positional_encoding_by_batch_ignores_time_order():
    x = torch.zeros(3, 4, 8)
    by_time, by_batch = PositionalEncoding(8)(x), PositionalEncoding(8, by="batch")(x)
    assert not torch.equal(by_time[0, 0], by_time[0, 1]) and torch.equal(by_time[0], by_time[1])
    assert torch.equal(by_batch[0, 0], by_batch[0, 1]) and not torch.equal(by_batch[0], by_batch[1])


def test_adv_sign_plus_flips_the_critic_term(repo):
    rows = {}
    for sign in ("minus", "plus"):
        out = train.fit(SPEC, tiny(f"train.adv_sign={sign}", "train.epochs=1"), CPU,
                        log=lambda *_: None, diagnostics=True)
        rows[sign] = out["diagnostics"][0]
    # epoch averages differ after the first step, but the two terms must have opposite signs early on
    assert np.sign(rows["minus"]["adv_term"]) == -np.sign(rows["plus"]["adv_term"])


def test_no_shuffle_and_same_batch_draw_no_batch_order(repo):
    cfg = tiny("train.shuffle=false", "train.critic_batch=same", "train.n_critic=1")
    a = train.fit(SPEC, cfg, CPU, log=lambda *_: None)
    b = train.fit(RunSpec(SPEC.dataset, 10, SPEC.arch, SPEC.critic, SPEC.optimizer, 1), cfg, CPU,
                  log=lambda *_: None)
    for p, q in zip(a["model"].state_dict().values(), b["model"].state_dict().values()):
        assert torch.equal(p, q)
    assert set(a["states"]) == {"last", "best_val"}


@pytest.mark.parametrize("profile", ["released_code", "released_data"])
@pytest.mark.parametrize("arch,critic,optimizer", [("transformer", "wgan-gp", "madgrad"),
                                                   ("lstm", "none", "adam")])
def test_profiles_train_into_their_own_experiment_folder(repo, profile, arch, critic, optimizer):
    args = ["--dataset", "synthetic_seasonal", "--interval", "10", "--arch", arch, "--critic", critic,
            "--optimizer", optimizer, "--seed", "1", "--device", "cpu", "--profile", profile]
    with pytest.raises(SystemExit) as stop:                # a profile is never the planned run
        train.main(args)
    assert "--exp" in str(stop.value)
    train.main([*args, "--exp", "demo", *TINY])
    spec = RunSpec("synthetic_seasonal", 10, arch, critic, optimizer, 1)
    r = json.loads((repo / "results" / "v2" / "demo" / f"{spec.run_id}.json").read_text())
    assert r["experiment"] == "demo" and r["profile"] == profile
    assert r["config"]["data"]["scaler"] == "per_split" and r["config"]["train"]["select"] == "last"
    assert r["selected_epoch"] == r["epochs_run"] == 2
    assert set(r["metrics_by_selection"]) == {"last", "best_val"}
    assert "mape_released_units" in r["metrics"]["test"]
    assert not list((repo / "results").glob("*.json"))     # nothing lands among the planned runs
    with pytest.raises(SystemExit) as again:
        train.main([*args, "--exp", "demo", *TINY])
    assert "already exists" in str(again.value)
    if profile == "released_code":
        assert r["config"]["optim"][optimizer]["lr"] == 0.01
        assert r["config"]["train"]["adv_sign"] == "plus" and r["config"]["train"]["n_critic"] == 1


def test_unknown_profile_and_unknown_profile_entries_are_errors(repo):
    with pytest.raises(SystemExit):
        config.resolve_config(SPEC, profile="no_such_profile")
    config.profile_path("bad").write_text("overrides: {}\nlearning_rate: 1\n")
    with pytest.raises(KeyError):
        config.resolve_config(SPEC, profile="bad")


def test_old_saved_configs_get_defaults_for_new_keys(repo):
    old = config.default_config()
    del old["data"]["scaler"], old["train"]["shuffle"]
    old["data"]["window"] = 7
    full = config.with_defaults(old)
    assert full["data"]["scaler"] == "train" and full["train"]["shuffle"] is True
    assert full["data"]["window"] == 7


def test_v2_summary_keeps_setups_apart(repo, capsys):
    from cwp import evaluate
    base = ["--dataset", "synthetic_seasonal", "--interval", "10", "--arch", "lstm", "--critic", "none",
            "--optimizer", "adam", "--device", "cpu"]
    for seed in ("1", "2"):
        train.main([*base, "--seed", seed, "--tag", "t", *TINY])              # stays out of everything
        train.run(RunSpec("synthetic_seasonal", 10, "lstm", "none", "adam", int(seed)), tiny(), CPU,
                  log=lambda *_: None)                                        # a "reported" run
        train.main([*base, "--seed", seed, "--profile", "released_data", "--exp", "exp_a", *TINY])
    evaluate.cmd_v2(["exp_a"], "demo")
    text = (repo / "results" / "v2" / "summary" / "demo.md").read_text()
    assert "paper settings (reported runs)" in text and "exp_a" in text and "NO (test statistics" in text
    runs_csv = (repo / "results" / "v2" / "summary" / "demo_runs.csv").read_text()
    assert runs_csv.count("exp_a") == 2
    with pytest.raises(SystemExit):
        evaluate.cmd_v2(["missing_exp"], "demo")
