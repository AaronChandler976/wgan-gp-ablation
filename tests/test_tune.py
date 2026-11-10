import torch
import yaml

from cwp import config, tune
from cwp.config import RunSpec

from conftest import TINY


def test_tune_writes_the_best_setting_and_train_picks_it_up(repo):
    (repo / "configs" / "tuning.yaml").write_text(yaml.safe_dump(
        {"epochs": 2, "grid": {"lr": [0.001, 0.01], "data.window": [8, 12]}}))
    d = repo / "configs" / "default.yaml"
    d.write_text(yaml.safe_dump(config.apply_overrides(config.default_config(), TINY)))

    spec = RunSpec("synthetic_seasonal", 10, "lstm", "none", "madgrad", 0)
    tuned = tune.tune(spec, torch.device("cpu"))
    assert len(tuned["trials"]) == 4
    assert tuned["val_mape"] == round(min(t["val_mape"] for t in tuned["trials"]), 4)

    planned = RunSpec("synthetic_seasonal", 10, "lstm", "none", "madgrad", 1)
    cfg, used = config.resolve_config(planned)
    assert used and cfg["optim"]["madgrad"]["lr"] == tuned["lr"]
    assert cfg["data"]["window"] == tuned["overrides"]["data.window"]
    other = RunSpec("synthetic_seasonal", 10, "lstm", "none", "adam", 1)
    assert not config.resolve_config(other)[1]            # tuned settings are per configuration
