import pathlib

import torch
import yaml

from cwp import bench, launch, runs


def shrink(repo):
    """Point the plan at the synthetic series and make every run tiny."""
    p = repo / "configs" / "plan.yaml"
    plan = yaml.safe_load(p.read_text())
    plan["datasets"] = [{"dataset": "synthetic_spiky", "interval": 10}]
    plan["seeds"] = [1]
    p.write_text(yaml.safe_dump(plan))
    d = repo / "configs" / "default.yaml"
    cfg = yaml.safe_load(d.read_text())
    cfg["train"].update(epochs=2, eval_every=1)
    cfg["data"]["window"] = 12
    cfg["model"]["transformer"].update(d_model=8, nhead=2, dim_ff=16)
    cfg["eval"]["timing_reps"] = 5
    d.write_text(yaml.safe_dump(cfg))


def test_bench_times_every_kind_and_estimates_the_plan(repo):
    shrink(repo)
    timing = bench.measure("synthetic_spiky", 10, torch.device("cpu"), epochs=1)
    assert set(timing) == set(bench.KINDS)
    assert all(t["sec_per_epoch"] > 0 for t in timing.values())
    est = bench.estimate_plan(timing, "synthetic_spiky", 10)
    assert est["runs"] == len(runs.plan_specs()) and est["seconds"] > 0


def test_launch_runs_the_next_planned_runs_locally(repo):
    shrink(repo)
    before = len(runs.todo())
    launch.main(["-n", "2", "--where", "local", "--yes", "--device", "cpu"])
    assert len(runs.todo()) == before - 2
    assert len(list(pathlib.Path(repo, "results").glob("*.json"))) == 2
