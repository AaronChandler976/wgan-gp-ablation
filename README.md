# Which part of the WGAN-gp Transformer helps?

An ablation study of the WGAN-gp Transformer for cloud workload prediction
([Arbat et al., AAAI 2022](https://doi.org/10.1609/aaai.v36i11.21509)). The paper changes three
things at once compared with its LSTM baseline (LoadDynamics): the architecture, an adversarial
critic, and the optimizer. **Research question: which of the three is responsible for the
reported improvement?** This repo trains every combination of the three switches (2 x 2 x 2 = 8
configurations) on two datasets with 5 seeds each, 80 runs in total, so each part can be
measured on its own.

| Switch | Off | On |
| --- | --- | --- |
| `--arch` | `lstm` (LoadDynamics) | `transformer` |
| `--critic` | `none` | `wgan-gp` (adversarial training) |
| `--optimizer` | `adam` | `madgrad` |

Team: Tung Nguyen, Angie Do, Aaron Chandler.

## Results in short

All 80 runs are finished. Every number below comes from
[`results/summary/summary.md`](results/summary/summary.md); the per-run values are in
[`results/summary/all_runs.csv`](results/summary/all_runs.csv).

Average effect of switching each part on, across the other two (factorial main effect on test
MAPE, 5 seeds; negative = lower error):

| Part | Google Cluster 2011, 10 min | Azure Functions 2019, 5 min |
| --- | --- | --- |
| Transformer instead of LSTM | -4.8% (p = 0.002) | +0.1% (p = 0.85) |
| WGAN-gp critic | +4.6% (p = 0.024) | +6.1% (p = 0.002) |
| MADGRAD instead of Adam | -0.3% (p = 0.65) | -2.6% (p = 0.078) |

- **The Transformer helps on Google only.** On Azure Functions it makes no measurable difference.
- **The critic hurts on both datasets.**
- **MADGRAD has no reliable effect** on either dataset.

Test MAPE (%), mean ± standard deviation over 5 seeds, for four of the eight configurations:

| Configuration | Google 10 min | Azure Functions 5 min |
| --- | --- | --- |
| LSTM, no critic, Adam (LoadDynamics baseline) | 22.69 ± 0.34 | 2.014 ± 0.054 |
| Transformer, no critic, MADGRAD (lowest on both) | 20.66 ± 0.21 | 2.012 ± 0.021 |
| Transformer, critic, MADGRAD (the paper's final model) | 22.62 ± 1.11 | 2.133 ± 0.057 |
| Naive forecast (repeat the last value) | 27.60 | 3.452 |

Notes for reading these:

- The p-values above are for the main effects. Most single one-change comparisons are not
  significant after the Holm correction; see the "One-change comparisons" table in `summary.md`.
- Our absolute MAPE values do not match the paper's Table 3 (Google 10 min: 11.49 LSTM, 10.58
  WGAN-gp Transformer; Azure Functions 5 min: 1.63 and 3.05). No variant of the public Google
  trace we tried is smooth enough to give the paper's values
  ([`google_variants.md`](results/summary/google_variants.md)), and the paper does not say how
  it aggregated Azure Functions. Compare directions, not exact numbers.
- Inference speed ([`timing_sweep_cuda.md`](results/summary/timing_sweep_cuda.md),
  [`timing_sweep_cpu.md`](results/summary/timing_sweep_cpu.md)): at a history of 672 steps on the
  GPU, a fused (cuDNN) LSTM takes 0.18 to 0.58 ms per prediction and the Transformer 0.29 to
  0.35 ms, while the same LSTM run one time step at a time takes 9.8 to 18.8 ms. The LSTM is
  only much slower than the Transformer when it is not run as one fused kernel.

### Follow-up experiments, in short

These came after the 80 reported runs and never change them. Each line names its own setup
and source; numbers from different setups are not pooled. Details and every variant tried,
including the ones that failed: [`docs/EXPERIMENTS_V2.md`](docs/EXPERIMENTS_V2.md). Suggested
text and tables for the report: [`docs/REPORT_NOTES.md`](docs/REPORT_NOTES.md).

- **Ten seeds** (reported runs plus `seeds_6_10`,
  [`effects_reported+seeds_6_10.md`](results/v2/summary/effects_reported+seeds_6_10.md)): the
  three conclusions above hold. The critic raises MAPE in 10 of 10 seeds on both datasets
  (Google +4.4%, Azure Functions +6.4%, Holm p at most 0.0004). The Transformer's gain on
  Google is -5.2% (10 of 10 seeds). MADGRAD gives -3.4% on Azure Functions (Holm p 0.006) and
  nothing on Google.
- **Four more series** (`more_data`: Google 30 min, Azure Functions 10 min, Azure VM 2017
  10 min, Wikipedia page views 60 min; 5 seeds, tuned with the same budget). Pooled over all
  six series ([`pooled_reported+more_data.md`](results/v2/summary/pooled_reported+more_data.md)):
  critic **+5.9%** (95% interval +4.4 to +7.5, worse on every series), Transformer **-3.3%**
  (-5.8 to -0.6), MADGRAD +0.6% (-1.3 to +2.6, differs by series). The Transformer's gain does
  not follow seasonality or burstiness
  ([`workload_reported+more_data.md`](results/v2/summary/workload_reported+more_data.md)).
- **Can the critic be made to help?** A validation-only screen of 34 critic variants (weight,
  warm-up, update ratio, penalty, normalization, conditional and first-difference critics),
  with 34 variants for the no-critic side as well, found that the smaller the critic's weight
  the better. With the screen's winner (`critic_v2`, weight 0.001) the critic cells improve by
  1 to 6% against the reported runs, but the critic's effect on test MAPE is still on the
  wrong side: Google +3.2% (Holm p 0.037), Azure Functions +1.7% (not significant)
  ([`effects_critic_v2.md`](results/v2/summary/effects_critic_v2.md)). On the four added
  series the same weight (`critic_v2_more`) leaves +0.1% to +2.3%, none significant, pooled
  +1.5% (0.2 to 2.8): no gain on any series
  ([`effects_critic_v2_more+nocritic_of_more_data.md`](results/v2/summary/effects_critic_v2_more+nocritic_of_more_data.md)).
- **Standard Adam with the critic** (`adam_control`,
  [`adam_control.md`](results/v2/summary/adam_control.md)): the critic cells are still no
  better than the no-critic cells, so the harm does not come from Adam's beta1 = 0.
- **The released code** (`released_full`, `released_data`, `released_item_*`): its data
  handling does not explain the paper's Google numbers (24.9 in the released code's own units
  against the paper's 10.6 to 11.5), and no single released choice does. Three choices break
  the paper's final model on their own: the sign of the critic term, learning rate 0.01, and
  positional encoding by batch position. The sign only breaks training with the paper's
  unbounded critic; with the released Tanh critic it does not
  ([`released_code.md`](results/v2/summary/released_code.md),
  [`versus_reported_released_items.md`](results/v2/summary/versus_reported_released_items.md)).
  These rows include settings that leak test statistics and are labeled as not fair.
- **Provisioning** ([`autoscale_reported.md`](results/v2/summary/autoscale_reported.md),
  [`autoscale_more_data.md`](results/v2/summary/autoscale_more_data.md)): the spare capacity
  needed to keep short intervals at or under 5% is about 3% on Wikipedia, 5 to 11% on Azure
  Functions, 54 to 67% on Google and over 150% on Azure VM. The workload decides this far more
  than the model does.
- **Training stability:** the LSTM with critic and MADGRAD diverges late in training on five
  of the six series (3 to 5 of 5 seeds, at weight 1 and at weight 0.001 alike; on Google
  10 min 0 and 1 of 5). No other configuration does. Choosing the epoch on validation hides
  this in the MAPE tables (counts: "5 result" in `docs/EXPERIMENTS_V2.md`).

## Setup

```bash
git clone https://github.com/AaronChandler976/wgan-gp-ablation.git
cd wgan-gp-ablation
conda env create -f environment.yml     # Python 3.11, PyTorch 2.11.0 (CUDA 12.8 build), the rest
conda activate cwp
pip install -e .
pytest                                  # 93 tests, about a minute; no data download, no GPU
```

[`environment.yml`](environment.yml) pins the versions the reported runs were made with. The
PyTorch wheel brings its own CUDA libraries and also runs on a machine without a GPU. If you
need a different build, install it over the pinned one after creating the environment:

| Machine | PyTorch install |
| --- | --- |
| Older GPUs (for example the HPC's P100 nodes; not tested) | `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121` |
| No GPU, smaller download | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

The tests use only the bundled synthetic series. If `pytest` fails with import errors from
another Python installation (for example on a machine with ROS, which sets `PYTHONPATH`), run
`unset PYTHONPATH` first. If you already have a conda environment called `cwp`, pick another
name with `conda env create -n <name> -f environment.yml`.

## Data

The processed series and the splits are in git, so nothing has to be downloaded to run the
tests, retrain, or rebuild the summaries. To rebuild them from the published traces
(Google: 500 small `job_events` files; Azure Functions: one archive of a few hundred MB):

```bash
python -m cwp.data download   --dataset google
python -m cwp.data preprocess --dataset google            # -> data/processed/google_10m.csv, google_30m.csv
python -m cwp.data download   --dataset azure_functions
python -m cwp.data preprocess --dataset azure_functions   # -> data/processed/azure_functions_5m.csv, _10m.csv
python -m cwp.stats                                       # -> results/summary/dataset_stats.md
```

Downloads go to `data/raw/` (not in git). The split is chronological, 60% / 20% / 20%. Sources,
licenses and file formats are in [`data/README.md`](data/README.md).

## Tests

```bash
pytest          # 93 tests; `unset PYTHONPATH` first if another Python installation interferes
```

The same tests run on every push (`.github/workflows/tests.yml`).

## Reproduce the experiments

Everything is driven by three flags; `key=value` arguments override
[`configs/default.yaml`](configs/default.yaml) for exploratory runs.

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

The plan ([`configs/plan.yaml`](configs/plan.yaml)) is 2 datasets x 8 configurations x 5 seeds
(1 to 5) = 80 runs. A run is **done** when `results/<run_id>.json` exists, and all 80 are, so
`train` refuses to repeat them unless you pass `--force`.

```bash
python -m cwp.runs status                 # what is done, claimed and left
python -m cwp.runs next                   # train commands for the runs that are left
python -m cwp.train ... --force           # retrain a finished run (overwrites its result)
python -m cwp.train --dataset synthetic_seasonal --interval 10 --arch transformer \
    --critic wgan-gp --optimizer madgrad --seed 1 --tag demo train.epochs=30   # quick try-out
```

`--tag` marks an exploratory run: it is saved under `results/scratch/` and does not count toward
the plan. Each planned run uses the tuned settings in `configs/tuned/` (in git) and saves its
result, its test predictions (`results/predictions/`) and a checkpoint (`checkpoints/`, not in
git). The reported runs took 0.6 to 20 minutes each on one RTX 5070 Ti (`train_min` in
`summary.md`).

**Tuning.** `configs/tuned/` was produced by `python -m cwp.tune --plan` from the grid in
[`configs/tuning.yaml`](configs/tuning.yaml): 32 trials for every configuration, validation
part only, with its own seed. You only need to run it again if you change the grid; `train`
refuses tuned files that were made from a different grid.

## Reproduce the summaries

```bash
python -m cwp.evaluate baselines          # naive, seasonal naive, moving average
python -m cwp.evaluate compare            # -> results/summary/summary.md and the CSV files
python -m cwp.evaluate dm RUN_A RUN_B     # Diebold-Mariano test on two runs' test predictions
python -m cwp.evaluate timing             # re-time all local checkpoints on this machine
python -m cwp.timing_sweep                # inference time vs history length -> timing_sweep_<device>.md
python -m cwp.timing_sweep --device cpu
```

`baselines`, `compare` and `dm` need only the files in git. `compare` reports mean ± standard
deviation over seeds for MAPE, MAE, RMSE and MASE, the baselines on the same test points, and
the paper's Table 3 values. It tests every pair of configurations that differ in exactly one
switch (paired by seed, Holm-corrected) and reports each part's average effect and the
interactions.

`timing` needs the checkpoints, which are not in git. `timing_sweep` needs none: it times
untrained models (speed does not depend on the weights) at history lengths 24 to 672: a fused
LSTM (one cuDNN kernel on a GPU), the same LSTM with cuDNN off, the same LSTM run one time step
at a time, and the Transformer generator. Inference time depends on the machine, so compare
speeds measured on one machine only.

## Follow-up experiments (v2)

Everything run after the 80 reported runs is additive: each experiment has its own name, its
runs are saved in `results/v2/<name>/`, and its tables in `results/v2/summary/`. The reported
runs and `results/summary/` are never rewritten. The plan of each experiment (what is run and
how a winner is picked, written before the runs) and its result are in
[`docs/EXPERIMENTS_V2.md`](docs/EXPERIMENTS_V2.md).

```bash
python -m cwp.train ... --exp NAME [key=value ...]     # one run of a v2 experiment -> results/v2/NAME/
python -m cwp.train ... --exp NAME --profile released_code   # with the released code's choices
python -m cwp.screen run configs/screens/critic_s1_weight.yaml      # a screen: short tuning runs,
python -m cwp.screen summarize configs/screens/critic_s1_weight.yaml  # scored on validation only
python -m cwp.evaluate v2 NAME --name NAME             # NAME's test results next to the reported runs
python -m cwp.analysis effects --exp NAME              # factorial effects, intervals, Holm correction
python -m cwp.analysis pooled --exp reported+more_data # effects across series, and whether they differ
python -m cwp.analysis pairs --exp NAME                # paired tests, one switch apart
python -m cwp.analysis autoscale --exp NAME            # provisioning cost of the forecasts
python -m cwp.analysis workload --exp reported+more_data   # effects next to workload properties
python -m cwp.analysis versus NAME --name NAME         # NAME against the reported run of the same seed
```

Without `--exp` the `cwp.analysis` commands read the reported runs. Sources joined with `+`
must share one setup and are labeled by source in the table.

## Report figures

```bash
python scripts/make_figures.py              # all -> figures/*.png and figures/*.pdf
python scripts/make_figures.py --only mape  # or any other name from --help
```

| File in [`figures/`](figures/) | Shows | Read from |
| --- | --- | --- |
| `mape_by_configuration` | Test MAPE of the 8 configurations on each dataset: one point per seed, mean ± standard deviation | `results/summary/all_runs.csv` |
| `factorial_main_effects` | Average effect of the Transformer, the critic and MADGRAD on MAPE, per dataset, with p-values | `results/summary/factorial_effects.csv` |
| `timing_sweep` | Inference time against history length on the GPU and the CPU, log scales | `results/summary/timing_sweep_cuda.csv`, `timing_sweep_cpu.csv` |
| `effects_all_series` | Effect of each part on all six series, with the pooled value | `results/v2/summary/effects_reported+more_data.csv`, `pooled_reported+more_data.csv` |
| `released_code_comparison` | The paper's Table 3, the reported runs and the released code's settings | `results/v2/summary/released_code_runs.csv` |
| `critic_weight_screen` | Validation MAPE against the critic's weight | `results/v2/critic_s1_weight/`, `nocritic_reference/` |
| `critic_diagnostics` | What the critic's gradient does during training | `results/v2/critic_diagnostics/`, `critic_diagnostics_w0p01/` |
| `critic_designs` | The critic's effect under the paper's setting and the two designs the screen chose | `results/v2/summary/effects_reported.csv`, `effects_critic_v2.csv`, `effects_critic_v2_diffs+nocritic_of_critic_v2.csv` |
| `autoscale_tradeoff` | Short intervals against idle machines as spare capacity grows | `results/v2/summary/autoscale_reported_curve.csv` |
| `forecast_best_worst` | Forecasts of the best and the worst configuration on part of the test set | `results/predictions/` |

The script reads the CSV files and hard-codes no values, so run it again after `compare` or
`timing_sweep`. It needs matplotlib, which `environment.yml` installs; in an environment made
before that was added, run `conda install -c conda-forge matplotlib-base`. The PDFs are sized
for a full text width (6.5 in) with 9 pt serif text; use them with `\includegraphics` in LaTeX.

## Repo map

```
configs/      default.yaml (hyperparameters), plan.yaml (runs + seeds), datasets.yaml,
              tuning.yaml (search grid), tuned/ (chosen settings), paper_reference.yaml,
              profiles/ (the released code's choices), screens/ (validation-only screens)
data/         raw/ (downloads, not in git), processed/ (one CSV per series), splits/
src/cwp/      data, models, train, runs, tune, evaluate, metrics, stats, timing_sweep, config,
              screen + analysis (v2 experiments), bench + launch (optional tooling)
results/      <run_id>.json (one per reported run), predictions/, baselines/, summary/ (tables
              and CSVs), v2/<experiment>/ (follow-up runs), v2/summary/ (their tables)
runs/claims/  who ran what (one file per run)
checkpoints/  trained weights (not in git)
scripts/      make_figures.py (report figures), hpc/ (Slurm scripts for the SJSU HPC)
figures/      report figures (PNG and PDF), made by scripts/make_figures.py
docs/         REPLICATION_NOTES.md, EXPERIMENTS_V2.md, REPORT_NOTES.md, CODE_REVIEW.md,
              BUILD_PROMPT.md
tests/        pytest
```

## Replication notes

[`docs/REPLICATION_NOTES.md`](docs/REPLICATION_NOTES.md) lists, item by item, what the paper
says, what its released code does, and what this repo does. Where the two disagree, this repo
follows the paper. The choices that matter most for the results:

- The scaler is fitted on the training part only (the released code fits it on each part).
- Every model is scored on exactly the same test points, whatever its history length.
- Every configuration uses the L1 loss, 1,000 epochs with no early stopping, and the epoch with
  the best validation MAPE.
- With the critic on, Adam uses the paper's WGAN-style settings (beta1 = 0; `optim.adam_with_critic`).
- Every configuration gets the same tuning budget.

## Software and data credits

- **WGAN-gp Transformer, released code** (Arbat et al., MIT license):
  <https://github.com/shivaniarbat/wgan-gp-transformer>. We read it at commit `d9519dc` to
  write the replication notes. The models here are our own implementation and follow the
  [paper](https://doi.org/10.1609/aaai.v36i11.21509) where the paper and that code differ.
- **LoadDynamics** (Jayakumar et al., IPDPS 2020): the LSTM baseline follows the paper's
  description; none of its code is used.
- **MADGRAD** optimizer package (`madgrad` 1.3): <https://github.com/facebookresearch/madgrad>,
  described in [arXiv:2101.11075](https://arxiv.org/abs/2101.11075).
- **PyTorch** (2.11.0): <https://pytorch.org/>.
- **Google Cluster 2011 trace** (`job_events` tables), CC BY 4.0:
  <https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md>.
- **Azure Functions 2019 trace**, CC BY:
  <https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md>.
  The trace's authors ask users to cite Shahrad et al., "Serverless in the Wild: Characterizing
  and Optimizing the Serverless Workload at a Large Cloud Provider", USENIX ATC 2020.

The files in `data/processed/` are derived from the two traces (job submissions or function
invocations counted per interval), so the traces' attribution terms apply to them too.

## How the runs were shared

Each run has its own result file and claim file (`runs/claims/<run_id>.json`), so several
people can train and push at once without merge conflicts: `python -m cwp.runs claim <run_id>`
reserves a run and `python -m cwp.runs sync` pulls, commits results and claims, and pushes.
Set `CWP_OWNER` to label your runs (otherwise your git user name is used).

## Optional tooling

```bash
python -m cwp.bench        # how long a run and the remaining plan take on this machine
python -m cwp.launch       # pick the next runs, then train here or submit to the HPC
```

### HPC scripts (Slurm)

> **Untested on the cluster.** The scripts in `scripts/hpc/` were only ever run against a fake
> Slurm, never on the SJSU HPC. All 80 reported runs were trained on one local machine, not
> with these scripts. Expect to fix partition names, GPU requests and the PyTorch build (the
> P100 install line is also untested) before they work.

```bash
bash scripts/hpc/setup_env.sh               # once, on the login node: conda env, PyTorch, this package
bash scripts/hpc/check_slots.sh --gpu       # free GPU nodes and the GPU model
bash scripts/hpc/submit.sh -n 6 --seeds "1" # claim the next 6 runs and queue them as a job array
bash scripts/hpc/sync_results.sh            # commit and push finished results
```

Settings and details are in [`scripts/hpc/README.md`](scripts/hpc/README.md).
