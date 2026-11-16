# Experiments v2: plan and log

Everything here is **additive**. The 80 reported runs (`results/*.json`), their tuned settings
(`configs/tuned/`) and `results/summary/` stay as they are. Each new experiment has its own
name and its own results folder, and results from different setups are never put in one table
without a label.

**Status: proposed at checkpoint 1, not approved yet. No v2 experiment has been run.**
The only training done so far is for the code review (`docs/CODE_REVIEW.md`, findings 1 and
14): reruns of 8 reported runs at seed 1, saved under `results/scratch/` (not in git) with the
tags `diag`, `diagcold` and `repro`. They reproduce reported runs and add no new result.

## Rules for every experiment below

1. **Selection uses validation MAPE only.** Hyperparameters, critic variants and epochs are all
   chosen on the validation part. Test metrics are computed once, for reporting.
2. **Written down first.** Before a batch of runs starts, its section here lists what is run
   and how the winner is picked. Results are added to the same section afterwards, including
   the variants that failed.
3. **Matched tuning budgets.** If the critic configurations get N extra tuning trials, the
   matching no-critic configurations get N extra trials over their own knobs.
4. **Tuning noise** (code review, finding 3). New tuning keeps one seed per trial, then reruns
   the best 4 settings with 2 more tuning seeds and picks by the 3-seed mean. Tuning seeds are
   0, 100 and 200; they are never evaluation seeds.
5. **Names.** `results/v2/<experiment>/` holds each experiment's result files, predictions and
   summary. Tuned settings go to `configs/tuned_v2/<experiment>/`.

## Compute basis

Times come from `train_min` in `results/summary/summary.md` and from the epochs recorded in
`configs/tuned/*.yaml`. They were measured with about 5 runs sharing the one RTX 5070 Ti, so
"GPU-hours" below means the sum of those per-run times. The 80 runs sum to 9.9 GPU-hours and
took about 3 hours of wall-clock time, so wall-clock time is roughly one third of the
GPU-hours when 5 runs share the card.

| Kind of run | One full run (1,000 epochs) | One 32-trial tuning (early stopping) |
| --- | --- | --- |
| LSTM, no critic | 0.6-0.9 min | 0.1-0.2 h |
| LSTM, critic | 6.2-6.8 min | 0.8-1.1 h |
| Transformer, no critic | 4.5-5.9 min | 1.0-1.4 h |
| Transformer, critic | 15.3-20.1 min | 3.0-4.0 h |

Tuning all 8 configurations on one dataset took about 10 GPU-hours (Google) and 12 (Azure).

## Proposed runs and cost

| Step | What | Runs | GPU-hours |
| --- | --- | --- | --- |
| 1a | Released-code profile in full: paper's final model and the LSTM baseline, 2 datasets, 5 seeds | 20 | 1.5 |
| 1b | Data-handling bundle only (per-split scaling, windows inside each split, last epoch, released un-scaling) on the paper settings: same 2 configurations, 2 datasets, 5 seeds | 20 | 3 |
| 1c | One released-code item at a time, from the paper settings, **3 seeds**: 9 items for the paper's final model, 4 for the LSTM baseline, 2 datasets | 78 | 16 |
| 2a | Critic screening on validation (2 tuning seeds, early stopping): adversarial weight (7 values), warm-up (4), critic knobs (16), normalization (2), conditional and first-difference critics (2); Transformer and LSTM with MADGRAD, 2 datasets | about 250 short trials | 13 |
| 2b | Tune the 1-2 best critic variants and, with the same number of trials, the no-critic side over dropout, weight decay and `dim_ff` (16 extra trials per configuration, plus the 3-seed recheck) | 16 configurations | 14 |
| 2c | Full design with "critic v2" and the re-tuned no-critic side: 8 configurations, 2 datasets, 5 seeds | 80 | 10 |
| 3a | Google 30 min and Azure Functions 10 min through the original pipeline (tune, then 8 configurations x 5 seeds) | 80 + tuning | 14 |
| 3b | Up to 2 new datasets, after approval (tune, then 8 x 5 seeds each) | 80 + tuning | about 16 each |
| 4a | Seeds 6-10 for the original 8 configurations on both datasets | 80 | 10 |
| 4b | Optional control for code review finding 2: Adam + critic with standard Adam, tuned learning rate reused, 2 configurations, 2 datasets, 5 seeds | 20 | 4 |
| | **Total without 3b and 4b** | | **about 82** |
| | **Total with both** | | **about 118** |

At one third wall-clock, that is about 27 to 40 hours of the local GPU, spread over three
weeks. **The plan does not rely on the HPC.** `scripts/hpc/` has never run on the cluster; if
the local GPU turns out to be too slow, the first HPC job would be a smoke test (one
`synthetic_seasonal` run with `train.epochs=30` under a tag) before anything real is submitted.

Notes on the counts:

- 1c items for the paper's final model: learning rate 0.01; Tanh critic applied per value;
  `n_critic` 1 with gradient-penalty weight 0.1; generator loss L1 *plus* the critic score;
  positional encoding by batch position; broadcast input embedding; scaler fitted per split;
  windows built inside each split; no shuffling. For the LSTM baseline only learning rate,
  scaler, windows and shuffling apply.
- "Last-epoch selection" needs no runs of its own: every new run will also record its test
  metrics at the last epoch (for reporting only).
- "Dropping empty Google intervals" needs no runs: no interval is empty in any processed
  series (code review, finding 13).
- 1c is the most expensive step for what it tells us. If time is short, run it on Google only
  (about 8 GPU-hours), where the gap to the paper is largest.

## Proposed order

1. **1a and 1b** (4.5 GPU-hours). Cheapest, and they answer the Phase 1 hypothesis: does data
   handling explain the paper's lower MAPE?
2. **2a** (13). Find out whether any critic variant is worth carrying forward.
3. **3a** (14), overnight. Two more comparison points for Table 3, no code changes.
4. **2b and 2c** (24). The fair critic-v2 factorial.
5. **4a** (10), overnight. More seeds for the borderline effects.
6. **1c** (16, or 8 for Google only). Which single discrepancy matters.
7. **3b** after the dataset table is approved; **4b** if there is time.

Dataset research for 3b (availability, license, size, how to derive the job arrival rate)
needs no GPU and can happen alongside steps 1-3.

## Log

Nothing yet. Each experiment gets a section here before it starts.
