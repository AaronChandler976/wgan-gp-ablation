# Which part of the WGAN-gp Transformer helps?

An ablation study of the WGAN-gp Transformer for cloud workload prediction
([Arbat et al., AAAI 2022](https://doi.org/10.1609/aaai.v36i11.21509)). The paper changes three
things at once compared with its LSTM baseline (LoadDynamics). This repo trains every
combination of those three switches so we can measure each one.

| Switch | Off | On |
| --- | --- | --- |
| `--arch` | `lstm` (LoadDynamics) | `transformer` |
| `--critic` | `none` | `wgan-gp` (adversarial training) |
| `--optimizer` | `adam` | `madgrad` |

Team: Tung Nguyen, Angie Do, Aaron Chandler.

## Install

```bash
git clone https://github.com/AaronChandler976/wgan-gp-ablation.git
cd wgan-gp-ablation
conda env create -f environment.yml
conda activate cwp
pip install torch            # or the line for your GPU from the table below
pip install -e .
pytest                       # about 30 seconds
```

| Machine | PyTorch install |
| --- | --- |
| RTX 50-series (needs a recent build) | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| HPC with older GPUs (P100) | `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121` |
| No GPU | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

## Quick start (no downloads needed)

Two small synthetic series are included so you can try everything right away.

```bash
python -m cwp.train --dataset synthetic_seasonal --interval 10 \
    --arch transformer --critic wgan-gp --optimizer madgrad --seed 1 \
    --tag demo train.epochs=30
```

`--tag` marks an exploratory run: it is saved under `results/scratch/` and does not count
toward the plan. `key=value` arguments override anything in `configs/default.yaml`.

## The 8 configurations

One command, three switches:

```bash
python -m cwp.train --dataset google --interval 10 \
    --arch <lstm|transformer> --critic <none|wgan-gp> --optimizer <adam|madgrad> --seed 1
```

| `--arch` | `--critic` | `--optimizer` | What it is |
| --- | --- | --- | --- |
| lstm | none | adam | LoadDynamics baseline (in the paper) |
| lstm | none | madgrad | Does the optimizer alone help the LSTM? |
| lstm | wgan-gp | adam | LSTM with the critic |
| lstm | wgan-gp | madgrad | LSTM with the critic and MADGRAD |
| transformer | none | adam | Plain Transformer |
| transformer | none | madgrad | Plain Transformer with MADGRAD |
| transformer | wgan-gp | adam | The paper's Adam comparison (its Fig. 3) |
| transformer | wgan-gp | madgrad | The paper's final model |

Which of these we run, on which datasets and with which seeds is set in
[`configs/plan.yaml`](configs/plan.yaml). It currently holds the proposal's plan:
2 datasets x 5 configurations x 3 seeds = 30 runs.

## Data

One person downloads and preprocesses; everyone else just pulls. Details in
[`data/README.md`](data/README.md).

```bash
python -m cwp.data download   --dataset google
python -m cwp.data preprocess --dataset google
python -m cwp.data download   --dataset azure_functions
python -m cwp.data preprocess --dataset azure_functions
python -m cwp.stats           # how spiky / seasonal each dataset is (training part only)
git add data/processed data/splits results/summary && git commit -m "Add processed datasets" && git push
```

## Team workflow

A run is **done** when `results/<run_id>.json` exists and **claimed** when
`runs/claims/<run_id>.json` exists. Each run has its own file, so we never get merge conflicts.

```bash
python -m cwp.runs status                 # what is done, claimed and left
python -m cwp.runs next --seeds 1         # train commands for unclaimed runs (your seeds)
python -m cwp.runs claim <run_id> ...     # reserve runs before a long session
python -m cwp.runs sync                   # pull, commit your results and claims, push
```

Then paste a command from `next`. Training claims the run, refuses to repeat a finished one,
and saves the result, the test predictions and a checkpoint.

**Split the work by seed, not by configuration.** If one person ran all the MADGRAD runs and
another all the Adam runs, any difference between their machines would look like an optimizer
effect. With a split by seed, every configuration is trained on every machine.

Set your name once so claims and results show who ran what:
`export CWP_OWNER=aaron` (otherwise your git user name is used).

## Evaluate and compare

```bash
python -m cwp.evaluate baselines          # naive, seasonal naive, moving average
python -m cwp.evaluate compare            # -> results/summary/summary.md
python -m cwp.evaluate dm RUN_A RUN_B     # Diebold-Mariano test on two runs' test predictions
python -m cwp.evaluate timing             # re-time all local checkpoints on this machine
```

`compare` reports mean ± standard deviation over seeds for MAPE, MAE, RMSE and MASE, the
baselines on the same test points, and the paper's Table 3 values. It then tests every pair of
configurations that differ in exactly one switch (paired by seed, Holm-corrected). When all 8
configurations are finished it also reports each part's average effect and the interactions.

Inference time depends on the machine. Use `timing` on **one** machine for the speed comparison.

## Tuning

```bash
python -m cwp.tune --plan                 # same budget for every configuration (32 trials each)
python -m cwp.tune --plan --only-dataset google --only-arch lstm   # one slice; run several at once
```

The grid is in [`configs/tuning.yaml`](configs/tuning.yaml): history length for every
configuration, plus a learning rate range, cell size and layers for the LSTM (what LoadDynamics
tunes) and a learning rate range and width for the Transformer (what the WGAN-gp paper tunes).
Each configuration gets the same 32 trials. Each trial trains for up to the full 1,000 epochs
and stops after 150 epochs without validation improvement; that early stopping applies to
tuning only, not to the planned runs. The tool refuses a grid that would tune one architecture
harder than the other. `tuning.yaml` explains why the ranges are where they are (round 1 picked
the edges of a smaller grid).

The models are small, so several tunes can share one GPU: run `--plan` with `--only-dataset`
and `--only-arch` in separate terminals, one slice each.

The best setting per dataset and configuration is saved in `configs/tuned/` and used by
`train` automatically. Tuning uses its own seed and the validation part only. Each tuned file
records the grid that made it: if `tuning.yaml` changes, `tune --plan` re-tunes that
configuration and `train` refuses the old file. Commit `configs/tuned/` **before** anyone
starts the planned runs, or runs will not be comparable.

## Checkpoints

`checkpoints/<run_id>.pt` holds the weights, the config and the scaler. They are not in git.
Put them in a shared Drive folder if someone else needs them (for example for `timing`).

## Layout

```
configs/      default.yaml (hyperparameters), plan.yaml (runs + seeds), datasets.yaml, tuning.yaml
data/         raw/ (not in git), processed/, splits/
src/cwp/      data, models, train, runs, evaluate, stats, tune, metrics, config
results/      <run_id>.json, predictions/, baselines/, summary/
runs/claims/  who is working on what
docs/         REPLICATION_NOTES.md (paper vs released code vs this repo), BUILD_PROMPT.md
tests/        pytest
```

## Decisions for the group

1. **Plan size.** `configs/plan.yaml`: `design` (`one_chain` 4, `proposal` 5, `two_chains` 6,
   `full` 8), `seeds`, and `datasets`. Adding more later never invalidates finished runs.
2. **Loss.** Every configuration uses L1 (the WGAN-gp paper's loss) so the loss is not a fourth
   difference. LoadDynamics originally used MSE (`train.loss`).
3. **Early stopping.** Off by default (1,000 epochs, as in the paper). `train.patience: 100`
   would save a lot of compute.
4. **Tuning.** Tune every configuration with the shared grid, or use the defaults for all.
5. **Adam with the critic.** The paper used WGAN-style Adam settings (beta1 = 0, lr = 0.0001) for
   its Adam run. `optim.adam_with_critic.enabled: false` uses ordinary Adam everywhere.

Change these in the YAML files, commit, and only then start planned runs.

## Not verified yet

- **Download and preprocessing of the real datasets.** Written against the published file
  formats and tested on small fake files in the same format, but never run on the real
  downloads. Check the first rows and the point counts (Google 10-minute: 4,176 points; Azure
  Functions 5-minute: 4,032 points).
- **GPU training.** Everything was tested on CPU only.
- The PyTorch install line for the HPC's P100 GPUs.

Paper-versus-code differences and our choices are in
[`docs/REPLICATION_NOTES.md`](docs/REPLICATION_NOTES.md).

## Optional tooling (this branch: `tools/hpc-launcher`)

```bash
python -m cwp.bench        # how long a run and the remaining plan take on this machine
python -m cwp.launch       # pick the next runs, then train here or submit to the HPC
```

Slurm scripts for the SJSU HPC are in [`scripts/hpc/`](scripts/hpc/README.md): set up the
environment, check free GPU nodes, submit runs as a job array, and push results.
They were tested against a fake Slurm only, not on the cluster.
