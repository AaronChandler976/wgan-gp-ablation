"""Run tracking: which runs the plan needs, which are finished, and who has claimed what.

A run is finished when results/<run_id>.json exists and claimed when runs/claims/<run_id>.json
exists. One small file per run means several people can push at the same time without conflicts.

Commands:
    python -m cwp.runs status
    python -m cwp.runs next   [--seeds 1 2] [--dataset google] [-n 5]
    python -m cwp.runs claim  <run_id> [<run_id> ...]
    python -m cwp.runs release <run_id> [...]
    python -m cwp.runs sync   # pull, commit results and claims, push
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

from . import config
from .config import RunSpec


def plan() -> dict:
    return config.load_yaml("configs", "plan.yaml")


def plan_specs() -> list[RunSpec]:
    p = plan()
    design = p["designs"][p["design"]]
    return [RunSpec(d["dataset"], int(d["interval"]), arch, critic, opt, int(seed))
            for d in p["datasets"] for arch, critic, opt in design for seed in p["seeds"]]


def result_path(run_id: str):
    return config.path("results", f"{run_id}.json")


def claim_path(run_id: str):
    return config.path("runs", "claims", f"{run_id}.json")


def load_json(p):
    return json.loads(p.read_text()) if p.exists() else None


def state(spec: RunSpec) -> tuple[str, dict | None]:
    """('done', result) | ('claimed', claim) | ('todo', None)"""
    result = load_json(result_path(spec.run_id))
    if result is not None:
        return "done", result
    claim = load_json(claim_path(spec.run_id))
    return ("claimed", claim) if claim is not None else ("todo", None)


def write_claim(spec: RunSpec, who: str | None = None) -> dict:
    claim = {"run_id": spec.run_id, "owner": who or config.owner(), "machine": platform.node(),
             "claimed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    p = claim_path(spec.run_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(claim, indent=2) + "\n")
    return claim


def todo(seeds=None, dataset=None) -> list[RunSpec]:
    out = []
    for s in plan_specs():
        if seeds and s.seed not in seeds:
            continue
        if dataset and s.dataset != dataset:
            continue
        if state(s)[0] == "todo":
            out.append(s)
    return out


def train_command(spec: RunSpec) -> str:
    return (f"python -m cwp.train --dataset {spec.dataset} --interval {spec.interval} "
            f"--arch {spec.arch} --critic {spec.critic} --optimizer {spec.optimizer} --seed {spec.seed}")


def print_status() -> None:
    p = plan()
    specs = plan_specs()
    counts = {"done": 0, "claimed": 0, "todo": 0}
    seeds = p["seeds"]
    by_data: dict[str, dict[str, dict[int, str]]] = {}
    for s in specs:
        st, info = state(s)
        counts[st] += 1
        cell = {"done": "done", "todo": "-"}.get(st) or f"claimed:{info['owner']}"
        by_data.setdefault(s.data_id, {}).setdefault(s.config_id, {})[s.seed] = cell
    print(f"Plan: design '{p['design']}', {len(p['datasets'])} dataset(s), seeds {seeds} "
          f"-> {len(specs)} runs")
    print(f"done {counts['done']}   claimed {counts['claimed']}   to do {counts['todo']}\n")
    width = max(len(c) for d in by_data.values() for row in d.values() for c in row.values())
    width = max(width, 6)
    for data_id, rows in by_data.items():
        print(data_id)
        print("  " + "configuration".ljust(30) + "".join(f"s{s}".ljust(width + 2) for s in seeds))
        for config_id, cells in rows.items():
            print("  " + config_id.ljust(30) + "".join(cells[s].ljust(width + 2) for s in seeds))
        print()
    planned = {s.run_id for s in specs}
    extra = sorted(f.stem for f in config.path("results").glob("*.json") if f.stem not in planned)
    if extra:
        print(f"{len(extra)} finished run(s) outside the current plan, e.g. {extra[0]}")


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=config.root(), text=True, capture_output=True)


def sync(message: str | None = None) -> None:
    """Pull, commit new results / claims / tuned settings, and push."""
    pull = git("pull", "--rebase", "--autostash")
    if pull.returncode != 0:
        print("git pull failed:\n" + pull.stderr.strip())
        return
    git("add", "results", "runs/claims", "configs/tuned")
    if git("diff", "--cached", "--quiet").returncode == 0:
        print("nothing new to commit")
    else:
        msg = message or f"Results and claims from {config.owner()} on {platform.node()}"
        commit = git("commit", "-m", msg)
        print(commit.stdout.strip().splitlines()[0] if commit.returncode == 0 else commit.stderr.strip())
    push = git("push")
    print("pushed" if push.returncode == 0 else "git push failed:\n" + push.stderr.strip())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    nx = sub.add_parser("next", help="list unclaimed, unfinished runs as train commands")
    nx.add_argument("--seeds", type=int, nargs="*")
    nx.add_argument("--dataset")
    nx.add_argument("-n", type=int, default=0, help="show at most n runs (0 = all)")
    nx.add_argument("--ids", action="store_true", help="print run ids instead of commands")
    for name in ("claim", "release"):
        c = sub.add_parser(name)
        c.add_argument("run_ids", nargs="+")
        c.add_argument("--owner")
    sy = sub.add_parser("sync")
    sy.add_argument("-m", "--message")
    a = ap.parse_args(argv)

    if a.cmd == "status":
        print_status()
    elif a.cmd == "next":
        specs = todo(a.seeds, a.dataset)
        for s in specs[: a.n or None]:
            print(s.run_id if a.ids else train_command(s))
        if not specs:
            print("nothing left to do for this filter", file=sys.stderr)
    elif a.cmd == "claim":
        for rid in a.run_ids:
            spec = RunSpec.from_run_id(rid)
            st, info = state(spec)
            if st == "done":
                print(f"{rid}: already finished, not claimed")
            elif st == "claimed" and info["owner"] != (a.owner or config.owner()):
                print(f"{rid}: already claimed by {info['owner']}, not changed")
            else:
                print(f"{rid}: claimed by {write_claim(spec, a.owner)['owner']}")
        print("Run `python -m cwp.runs sync` so your teammates see the claims.")
    elif a.cmd == "release":
        for rid in a.run_ids:
            p = claim_path(rid)
            if p.exists():
                p.unlink()
                print(f"{rid}: claim removed")
            else:
                print(f"{rid}: no claim")
    elif a.cmd == "sync":
        sync(a.message)


if __name__ == "__main__":
    sys.exit(main())
