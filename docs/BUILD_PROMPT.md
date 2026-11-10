# Build prompt

This is the prompt that produced this repository. It works in Claude Code (run it from an
empty folder) or in a Claude chat that has a shell. Keep it up to date if the design changes,
so the repo can be rebuilt or extended from one place.

---

You are building the code repository for a three-person graduate course project (CMPE 252).
The project is an **ablation study** of the WGAN-gp Transformer for cloud workload prediction
(Arbat et al., AAAI 2022, https://github.com/shivaniarbat/wgan-gp-transformer), compared with
the LoadDynamics LSTM (Jayakumar et al., IPDPS 2020). The research question is which of three
parts causes the paper's improvement: the Transformer architecture, the adversarial critic, or
the MADGRAD optimizer.

## Before writing code

1. Clone and read the WGAN-gp repo (`main.py`, `quaesita/`, `utils/`). Read the LoadDynamics
   paper (https://wwang.github.io/papers/LoadDynamics.pdf) for the LSTM setup.
2. Write down every place the released code differs from the paper (learning rate, critic
   activation, critic update ratio, scaling, positional encoding). Follow the **paper** where
   they differ, and record each difference in `docs/REPLICATION_NOTES.md`.

## What the repo must do

1. **Three independent switches**, each a command-line flag, so all 2 x 2 x 2 = 8 combinations
   train through one code path:
   - `--arch lstm | transformer` (LoadDynamics-style LSTM; 1-encoder/1-decoder Transformer)
   - `--critic none | wgan-gp` (adversarial training with a 3-layer MLP critic and gradient
     penalty; must work with either architecture)
   - `--optimizer adam | madgrad`
   Hyperparameters live in YAML and can be overridden with dotted `key=value` arguments. A
   misspelled key must be an error.
2. **Seeds.** The list of evaluation seeds is stored in `configs/plan.yaml` and every result
   records its seed. A separate seed is reserved for hyperparameter tuning.
3. **Run tracking without merge conflicts.** A run has a deterministic id
   (`<dataset>-<interval>m__<arch>-<critic>-<optimizer>__s<seed>`). "Done" means
   `results/<run_id>.json` exists; "claimed" means `runs/claims/<run_id>.json` exists. One file
   per run, so three people can push at once. Provide `status`, `next`, `claim`, `release` and
   `sync` commands. Training refuses to repeat a finished run.
4. **Training.** One trainer for supervised and WGAN-gp training. Paper settings by default:
   1,000 epochs, 5 critic updates per generator update, gradient penalty weight 10, L1 loss.
   Select the best epoch on validation MAPE. Save a checkpoint (weights, config, scaler), the
   test predictions and a result JSON with metrics, timing, hardware and git commit. Turn off
   TF32 and seed everything so runs from different machines are comparable.
5. **Data.** Download and preprocess Google Cluster 2011 (job SUBMIT events) and Azure Functions
   2019 (total invocations) into job-arrival-rate series at the paper's intervals. Raw data is
   git-ignored; processed series and the 60/20/20 chronological split boundaries are committed.
   Scale with min-max to (-1, 1) fitted on the training part only. Validation and test windows
   may use earlier points as history, so every configuration is scored on exactly the same
   target points. Include two small synthetic series for tests and timing.
6. **Evaluation.** MAPE, MAE, RMSE, MASE, WAPE, under- and over-provisioning rates, and
   single-prediction inference time. Baselines: naive, seasonal naive, moving average.
   A `compare` command aggregates mean and standard deviation over seeds, runs paired tests on
   every pair of configurations that differ in exactly one switch (paired t-test, Wilcoxon when
   there are at least 6 seeds, Holm correction), computes factorial main effects and
   interactions on log MAPE when all 8 are present, and offers a Diebold-Mariano test. Show the
   paper's Table 3 values next to ours.
7. **Dataset metrics**, computed on the training part only: coefficient of variation, 99th
   percentile over median, kurtosis, autocorrelation at one day, seasonal strength, spectral
   entropy, share of near-zero intervals, and naive / seasonal-naive error.
8. **Hyperparameter tuning** with the same small grid and budget for every configuration.

## Optional tooling, on a separate branch (`tools/hpc-launcher`)

- `bench`: train each kind of run for a few epochs and estimate the time for a full run and
  for the remaining plan on this machine.
- HPC scripts for a Slurm cluster (SJSU CoE HPC): set up the conda environment, show free GPU
  nodes and the estimated start time, submit runs as a job array (claiming them first), and
  commit and push results when they finish. Partition and GPU flags must be configurable.
- `launch`: an interactive command that picks the next unclaimed runs, offers "this machine"
  (after a time estimate) or "HPC", and starts them.

## Engineering rules

- Python package under `src/cwp`, commands as `python -m cwp.<module>`, conda
  `environment.yml`, small dependencies (PyTorch, NumPy, pandas, SciPy, PyYAML, madgrad).
- Tests with pytest: data splitting and leakage, metrics, run tracking, and a two-epoch
  training run of all 8 configurations on synthetic data.
- README written for the team: quick start, the 8 configurations as copy-paste commands, the
  team workflow for splitting runs by seed, and the decisions the group still has to make.
- Say plainly in the README what was not verified (anything that needs the real datasets or
  the cluster).
- Commit the core on `main` and the optional tooling on `tools/hpc-launcher`.
