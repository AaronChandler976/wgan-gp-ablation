"""Pick the next runs and start them, on this machine or on the HPC.

    python -m cwp.launch                       # asks how many runs and where
    python -m cwp.launch -n 3 --seeds 1 --where local --yes
    python -m cwp.launch -n 8 --where hpc --yes

"local" trains the runs one after another here, after a short timing estimate.
"hpc" hands them to scripts/hpc/submit.sh, which claims them and submits a Slurm job array.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from . import bench, config, runs, train


def ask(prompt: str, default: str) -> str:
    try:
        answer = input(f"{prompt} [{default}] ").strip()
    except EOFError:
        answer = ""
    return answer or default


def run_local(specs, device_name: str, yes: bool, no_bench: bool) -> None:
    device = train.pick_device(device_name)
    if not no_bench:
        dataset, interval = bench.bench_dataset()
        print(f"Estimating time on {device} (a few epochs of each kind of run) ...", flush=True)
        timing = bench.measure(dataset, interval, device, epochs=2)
        est = bench.estimate_plan(timing, dataset, interval, specs)
        print(f"These {len(specs)} run(s) should take about {bench.fmt(est['seconds'])} here.")
    if not yes and ask("Start now? (y/n)", "y").lower() != "y":
        print("Nothing started.")
        return
    for s in specs:                       # claim everything first so teammates see it after a sync
        if runs.state(s)[0] == "todo":
            runs.write_claim(s)
    for i, s in enumerate(specs, 1):
        print(f"\n[{i}/{len(specs)}] {s.run_id}", flush=True)
        train.main(["--dataset", s.dataset, "--interval", str(s.interval), "--arch", s.arch,
                    "--critic", s.critic, "--optimizer", s.optimizer, "--seed", str(s.seed),
                    "--device", device_name])
    print("\nAll done. Share the results: python -m cwp.runs sync")


def run_hpc(n: int, seeds, dataset) -> None:
    if not shutil.which("sbatch"):
        print("`sbatch` is not available here, so this is not the HPC login node.\n"
              "Log in (ssh, with the SJSU VPN on), `cd` to this repo there, and run the same command.\n"
              "First time on the HPC: bash scripts/hpc/setup_env.sh")
        return
    cmd = ["bash", str(config.path("scripts", "hpc", "submit.sh")), "-n", str(n)]
    if seeds:
        cmd += ["--seeds", " ".join(map(str, seeds))]
    if dataset:
        cmd += ["--dataset", dataset]
    subprocess.run(cmd, cwd=config.root(), check=False)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, help="how many runs to start")
    ap.add_argument("--seeds", type=int, nargs="*", help="only these seeds (split the work by seed)")
    ap.add_argument("--dataset")
    ap.add_argument("--where", choices=["local", "hpc"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--no-bench", action="store_true", help="skip the timing estimate")
    a = ap.parse_args(argv)

    todo = runs.todo(a.seeds, a.dataset)
    if not todo:
        print("Nothing left to do for this filter. See `python -m cwp.runs status`.")
        return
    print(f"{len(todo)} run(s) are unclaimed and unfinished for this filter.")
    n = a.n or int(ask("How many do you want to start?", str(min(len(todo), 4))))
    specs = todo[:n]
    for s in specs:
        print("  " + s.run_id)
    where = a.where or {"1": "local", "2": "hpc"}.get(
        ask("Where?  1 = this machine   2 = HPC (Slurm)", "1"), "local")
    if where == "local":
        run_local(specs, a.device, a.yes, a.no_bench)
    else:
        run_hpc(len(specs), a.seeds, a.dataset)


if __name__ == "__main__":
    sys.exit(main())
