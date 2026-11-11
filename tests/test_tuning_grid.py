import pytest
import yaml

from cwp import config
from cwp.config import RunSpec

GRID = {"epochs": 2, "grid": {"lr": [0.001, 0.01], "data.window": [8, 12]},
        "arch_grid": {"lstm": {"model.lstm.hidden": [8, 16]},
                      "transformer": {"model.transformer.d_model": [8, 16]}}}


def test_each_architecture_gets_shared_plus_its_own_settings():
    lstm = config.tuning_settings("lstm", GRID)
    tr = config.tuning_settings("transformer", GRID)
    assert len(lstm) == len(tr) == 8
    assert all("model.lstm.hidden" in s and "model.transformer.d_model" not in s for s in lstm)
    assert all("model.transformer.d_model" in s and "model.lstm.hidden" not in s for s in tr)
    assert config.check_equal_tuning_budget(GRID) == 8


def test_unequal_budget_is_refused():
    bad = {**GRID, "arch_grid": {"lstm": {"model.lstm.hidden": [8, 16, 32]},
                                 "transformer": {"model.transformer.d_model": [8, 16]}}}
    with pytest.raises(ValueError):
        config.check_equal_tuning_budget(bad)


def test_shipped_grid_is_equal_and_valid(repo):
    n = config.check_equal_tuning_budget()
    assert n == 32
    assert 'train.patience' in config.tuning_fixed()
    for arch in config.ARCHS:                        # every key must be a real config key
        for setting in config.tuning_settings(arch):
            config.apply_overrides(config.default_config(),
                                   {k: v for k, v in setting.items() if k != "lr"})


def test_grid_id_changes_only_for_the_architecture_whose_grid_changed():
    other = {**GRID, "arch_grid": {**GRID["arch_grid"], "lstm": {"model.lstm.hidden": [8, 32]}}}
    assert config.tuning_grid_id("lstm", GRID) != config.tuning_grid_id("lstm", other)
    assert config.tuning_grid_id("transformer", GRID) == config.tuning_grid_id("transformer", other)
    assert config.tuning_grid_id("lstm", GRID) != config.tuning_grid_id("lstm", {**GRID, "epochs": 3})


def test_stale_tuned_file_is_not_current_and_train_refuses_it(repo):
    spec = RunSpec("synthetic_seasonal", 10, "lstm", "none", "adam", 1)
    p = config.tuned_path(spec)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump({"lr": 0.01, "overrides": {"data.window": 24}}))   # no grid_id
    assert not config.tuned_is_current(spec)
    with pytest.raises(SystemExit):
        config.resolve_config(spec)
    p.write_text(yaml.safe_dump({"lr": 0.01, "overrides": {"data.window": 24},
                                 "grid_id": config.tuning_grid_id("lstm")}))
    assert config.tuned_is_current(spec)
    cfg, used = config.resolve_config(spec)
    assert used and cfg["data"]["window"] == 24


def test_fixed_settings_change_the_grid_id():
    with_fixed = {**GRID, "fixed": {"train.patience": 50}}
    assert config.tuning_fixed(with_fixed) == {"train.patience": 50}
    assert config.tuning_grid_id("lstm", GRID) != config.tuning_grid_id("lstm", with_fixed)
