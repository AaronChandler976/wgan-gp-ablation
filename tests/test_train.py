import itertools
import json

import pytest
import torch

from cwp import config, evaluate, runs, train
from cwp.config import ARCHS, CRITICS, OPTIMIZERS, RunSpec
from cwp.models import build_critic, build_forecaster

from conftest import TINY


@pytest.mark.parametrize("arch", ARCHS)
def test_forecasters_map_a_window_to_one_value(repo, arch):
    cfg = config.apply_overrides(config.default_config(), TINY)
    model = build_forecaster(arch, cfg)
    assert model(torch.zeros(7, cfg["data"]["window"])).shape == (7,)
    assert build_critic(cfg)(torch.zeros(7, cfg["data"]["window"] + 1)).shape == (7,)


@pytest.mark.parametrize("arch,critic,optimizer", list(itertools.product(ARCHS, CRITICS, OPTIMIZERS)))
def test_all_eight_configurations_train_and_save(repo, arch, critic, optimizer):
    code = train.main(["--dataset", "synthetic_seasonal", "--interval", "10", "--arch", arch,
                       "--critic", critic, "--optimizer", optimizer, "--seed", "1",
                       "--device", "cpu", "--tag", "test", *TINY])
    assert code is None
    spec = RunSpec("synthetic_seasonal", 10, arch, critic, optimizer, 1)
    result = json.loads((repo / "results" / "scratch" / f"{spec.run_id}__test.json").read_text())
    assert result["seed"] == 1 and result["epochs_run"] == 2
    assert 0 < result["metrics"]["test"]["mape"] < 1000
    assert (repo / "checkpoints" / f"{spec.run_id}__test.pt").exists()
    assert runs.state(spec)[0] == "todo"               # tagged runs never count toward the plan


def test_same_seed_gives_the_same_result_and_planned_runs_are_not_repeated(repo):
    spec = RunSpec("synthetic_spiky", 10, "lstm", "wgan-gp", "madgrad", 2)
    cfg = config.apply_overrides(config.default_config(), TINY)
    dev = torch.device("cpu")
    a = train.evaluate(train.fit(spec, cfg, dev, log=lambda *_: None)["model"],
                       train.prepare_data(spec, cfg, dev), "test")[0]["mape"]
    b = train.run(spec, cfg, dev, log=lambda *_: None)["metrics"]["test"]["mape"]
    assert a == pytest.approx(b, rel=1e-6)
    assert runs.state(spec)[0] == "done"
    with pytest.raises(SystemExit) as stop:
        train.main(["--dataset", "synthetic_spiky", "--interval", "10", "--arch", "lstm",
                    "--critic", "wgan-gp", "--optimizer", "madgrad", "--seed", "2"])
    assert "already finished" in str(stop.value)


def test_overrides_need_a_tag(repo):
    with pytest.raises(SystemExit) as stop:
        train.main(["--dataset", "synthetic_spiky", "--interval", "10", "--arch", "lstm",
                    "--critic", "none", "--optimizer", "adam", "--seed", "1", "train.epochs=1"])
    assert "--tag" in str(stop.value)


def test_compare_runs_paired_tests_and_factorial_effects(repo, capsys):
    cfg = config.apply_overrides(config.default_config(), TINY)
    for arch, critic, opt in itertools.product(ARCHS, CRITICS, OPTIMIZERS):
        for seed in (1, 2, 3):
            train.run(RunSpec("synthetic_seasonal", 10, arch, critic, opt, seed), cfg,
                      torch.device("cpu"), log=lambda *_: None)
    df = evaluate.load_results()
    assert len(df) == 24
    pairs = evaluate.one_change_pairs(df)
    assert len(pairs) == 12 and set(pairs["seeds"]) == {3}          # 12 edges of the 2x2x2 cube
    effects = evaluate.factorial_effects(df)
    assert list(effects["kind"]).count("main effect") == 3 and len(effects) == 7
    evaluate.cmd_compare()
    assert (repo / "results" / "summary" / "summary.md").exists()
    assert "One-change comparisons" in capsys.readouterr().out
