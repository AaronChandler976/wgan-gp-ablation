# HPC scripts (Slurm)

For the SJSU College of Engineering HPC. **These scripts were tested against a fake Slurm
only, never on the cluster.** Run `check_slots.sh` first and fix the settings if needed.

| Script | Where | What it does |
| --- | --- | --- |
| `setup_env.sh` | HPC login node, once | Installs Miniconda if needed, creates the `cwp` environment, installs PyTorch and this package |
| `check_slots.sh` | HPC | Free GPU nodes, your jobs, and when a new GPU job would start (`--gpu` also shows the GPU model) |
| `submit.sh -n N` | HPC | Claims the next N unclaimed runs and submits them as one job array |
| `sync_results.sh` | HPC | Commits and pushes finished results and claims |
| `fetch_checkpoints.sh` | Your machine | Copies the trained weights from the HPC (they are not in git) |

## First time

```bash
ssh <sjsu-id>@coe-hpc1.sjsu.edu            # SJSU VPN on
git clone https://github.com/AaronChandler976/cmpe252-wgan-ablation.git
cd cmpe252-wgan-ablation && git checkout tools/hpc-launcher
bash scripts/hpc/setup_env.sh
bash scripts/hpc/check_slots.sh --gpu
```

## Every time

```bash
git pull
bash scripts/hpc/submit.sh -n 6 --seeds "1"     # claims 6 runs and queues them
squeue -u $USER                                  # watch
bash scripts/hpc/sync_results.sh                 # when they are done
```

Or use the launcher, which asks how many runs and where: `python -m cwp.launch`.

## Settings

Set these as environment variables, or edit the defaults in `_env.sh`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CWP_PARTITION` | `gpu` | Slurm partition with GPU nodes |
| `CWP_GRES` | `gpu` | GPU request (`--gres`) |
| `CWP_TIME` | `04:00:00` | Time limit per run |
| `CWP_CONCURRENT` | `4` | Runs of one batch allowed at the same time |
| `CWP_ENV` | `cwp` | conda environment name |
| `CWP_TORCH`, `CWP_TORCH_INDEX` | `torch==2.5.1`, cu121 | PyTorch build installed by `setup_env.sh` |

## What the cluster's documentation says

From http://coe-hpc-web.sjsu.edu/ (the page looks dated and only describes the P100 nodes):

- Scheduler: Slurm. Never compute on the login node.
- Partitions: `compute` (no GPU, 24 h limit), `gpu` (P100, 48 h limit), `interactive` (8 h),
  `condo` (jobs can be preempted). The default time limit is 4 hours.
- A GPU is only given when requested: `srun -p gpu --gres=gpu --pty /bin/bash`.
- Software: `module avail`; conda has to be installed by the user.

The newer A100 / H100 nodes may use a different partition or cluster. Check with `sinfo` and
set `CWP_PARTITION` and `CWP_GRES` to match.

## Pushing from the HPC

`sync_results.sh` needs GitHub access from the HPC: add an SSH key there
(`ssh-keygen`, then add the public key in GitHub settings) and clone with the SSH URL, or use
a personal access token. Without that, copy `results/` and `runs/claims/` back with `rsync`
and run `python -m cwp.runs sync` on your own machine.
