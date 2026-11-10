import json

from cwp import config, runs
from cwp.config import RunSpec


def test_run_id_round_trip():
    s = RunSpec("azure_functions", 5, "transformer", "wgan-gp", "madgrad", 3)
    assert s.run_id == "azure_functions-5m__transformer-critic-madgrad__s3"
    assert RunSpec.from_run_id(s.run_id) == s


def test_plan_counts_and_status(repo, capsys):
    p = runs.plan()
    specs = runs.plan_specs()
    assert len(specs) == len(p["datasets"]) * len(p["designs"][p["design"]]) * len(p["seeds"])
    assert len({s.run_id for s in specs}) == len(specs)
    assert len(p["designs"]["full"]) == 8

    first, second = specs[0], specs[1]
    assert runs.state(first)[0] == "todo"
    runs.write_claim(first, "angie")
    assert runs.state(first) [0] == "claimed" and runs.state(first)[1]["owner"] == "angie"
    runs.result_path(second.run_id).parent.mkdir(parents=True, exist_ok=True)
    runs.result_path(second.run_id).write_text(json.dumps({"run_id": second.run_id}))
    assert runs.state(second)[0] == "done"
    assert first not in runs.todo() and second not in runs.todo()
    assert len(runs.todo()) == len(specs) - 2

    runs.main(["status"])
    out = capsys.readouterr().out
    assert "claimed:angie" in out and "done 1" in out


def test_overrides_reject_unknown_keys(repo):
    cfg = config.default_config()
    assert config.apply_overrides(cfg, ["train.epochs=5"])["train"]["epochs"] == 5
    try:
        config.apply_overrides(cfg, ["train.epoch=5"])
    except KeyError:
        pass
    else:
        raise AssertionError("a misspelled key must be an error")
