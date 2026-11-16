# Experiments v2: plan and log

Everything here is **additive**. The 80 reported runs (`results/*.json`), their tuned settings
(`configs/tuned/`) and `results/summary/` stay as they are. Each new experiment has its own
name and its own results folder, and results from different setups are never put in one table
without a label.

**Status: approved on 2026-11-16** with these decisions: the proposed order stands (the
one-item-at-a-time ablation stays at step 6); new tuning rechecks the best 4 settings with 2 more
tuning seeds (rule 4); the Adam control (4b) is included.
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

Each experiment gets a section here before it starts; results are added to the same section.

### 1a. `released_full`: the released code's choices, all together

*Written before the runs started (2026-11-16).*

**Question.** What does our pipeline give when it makes every choice the released code makes?

**What is run.** `--profile released_code --exp released_full` for the paper's final model
(`transformer`, `wgan-gp`, `madgrad`) and the LSTM baseline (`lstm`, `none`, `adam`), on
`google-10m` and `azure_functions-5m`, seeds 1-5: 20 runs, 1,000 epochs each. The profile
(`configs/profiles/released_code.yaml`) sets: learning rate 0.01; Tanh critic that scores each
value on its own; 1 critic update per generator update on the generator's own batch;
gradient-penalty weight 0.1; generator loss L1 *plus* the critic score; positional encoding by
batch position; broadcast input embedding; scaler fitted separately on each part; windows
inside each part; no shuffling; last epoch reported; empty intervals dropped (there are none).
For the LSTM baseline only the learning rate, the data handling, the batch order and the epoch
choice apply.

**What is not reproduced.** The released code takes history length, `d_model`, heads, dropout
and batch size from the command line and the paper does not list the values used, so these
stay at this repo's tuned values (batch size 64, dropout 0.1). Its critic is as wide as
`d_model`; ours stays at 64. Its gradient penalty draws one mixing weight per time position;
ours draws one per sample. The released repository has no LSTM, so "LSTM under the released
profile" is our construction.

**No selection.** Nothing is tuned or chosen in this experiment; the settings are fixed in
advance. Test metrics are reported as they come, including runs that diverge.

**Reported.** `results/v2/summary/released_code.md`: test MAPE in true units and in the
released code's units, at the last epoch and at the best validation epoch, next to the
reported paper-settings runs and the paper's Table 3. These settings are labeled **not a fair
evaluation** (per-split scaling uses the test part's minimum and maximum).

### 1b. `released_data`: data handling only

*Written before the runs started (2026-11-16).*

**Hypothesis.** The paper's lower Google MAPE (10.58 / 11.49 against our 21-23) is explained by
data handling: per-split scaling, windows inside each part, last-epoch reporting, and
un-scaling with the validation part's range.

**What is run.** `--profile released_data --exp released_data`: the paper settings and this
repo's tuned hyperparameters, with only `data.scaler=per_split`, `data.windows=inside_split`,
`data.drop_empty=true` and `train.select=last` changed. Same 2 configurations, 2 datasets,
seeds 1-5: 20 runs.

**Decision rule, fixed in advance.** The hypothesis counts as **supported** if, on Google
10 min, mean test MAPE in the released code's units falls at least halfway from our reported
value to the paper's for both configurations (below 16.6 for the final model, below 17.1 for
the LSTM; reported 22.62 and 22.69). If it moves less than a quarter of the way (stays above
19.6 and 19.9) it is **rejected**. In between is "partly". The zero-training check in the code
review (finding 13) predicts rejection: the same un-scaling applied to the reported runs'
predictions gives 20.1-22.0.

**No selection** here either. Both epoch choices are recorded for every run.

### 2a. Critic screen (validation only)

*Written before the screen started (2026-11-16).*

**Question.** Is there any way of using the critic that gives a lower validation MAPE than the
reported critic setting, and does any of them reach the no-critic level?

**Protocol.** `python -m cwp.screen run configs/screens/<stage>.yaml`. A trial is one cell x
variant x tuning seed: the cell's tuned settings from `configs/tuned/`, plus the variant's
overrides, trained with `train.patience: 150` (as in tuning). Only the best validation MAPE is
stored; the screening tool never evaluates the test part. Tuning seeds 0 and 100. Cells: the
Transformer and the LSTM with the critic and MADGRAD, on `google-10m` and `azure_functions-5m`
(MADGRAD because the Adam cells mix the critic with a change of Adam's settings, code review
finding 2). A variant's score is its validation MAPE relative to the reference variant in the
same cell, averaged over the 4 cells on the log scale. Diverged trials count as they are.

**Noise.** The seed-to-seed spread of validation MAPE at one setting is 1-4% of its value
(code review, finding 3), so with 2 seeds per cell a difference of 1-2% in one cell means
little. A variant is
called better than another only if it is better in at least 3 of the 4 cells *and* its average
is more than 1% lower.

**Stages.** Each stage's file is committed before it runs, and its rule for what goes forward
is fixed here.

1. `critic_s1_weight`: `train.adv_weight` in {1 (reported), 0.3, 0.1, 0.03, 0.01, 0.003, 0.001},
   plus 0 (the critic is trained but cannot influence the forecaster). 64 trials.
   `nocritic_reference`: the 4 matching no-critic cells at their own tuned settings, 8 trials.
   *Forward:* `w_best`, the weight with the lowest average; and `w_knob`, the **largest**
   weight whose average is within 1% of `w_best`'s. Later stages use `w_knob`, because the
   critic's own settings can only matter where the critic term still has some weight.
2. `critic_s2_warmup`: at `w_knob` and at weight 1: L1 only for 50 epochs then the critic at
   full weight; and 50 epochs L1 only then a linear ramp over 100 epochs. 4 variants.
3. `critic_s3_knobs`: at `w_knob`: 16 of the 24 combinations of `train.n_critic` {1, 5},
   `train.gp_lambda` {1, 10}, `model.critic.hidden` {32, 128}, `train.critic_lr_scale`
   {0.25, 1, 4}, drawn with a fixed random seed. *Forward:* the best combination, if it is
   better than the stage-1 setting by the noise rule; otherwise the stage-1 setting.
4. `critic_s4_norm`: spectral normalization and layer normalization on the setting carried
   forward. 2 variants.
5. `critic_s5_input`: the conditional critic and the first-difference critic, each at `w_knob`
   and at weight 1 (an informative critic may deserve more weight). 4 variants.

**What is carried into the full design (2b, 2c).** The single best variant over all stages by
average score, and additionally the best stage-5 variant if it is a different one. If nothing
beats the reported setting by the noise rule, "critic v2" is the reported critic at `w_best`,
and the report says that no variant helped.

**Matched budget (rule 3).** The screen is tuning for the critic side: 8 + 4 + 16 + 2 + 4 = 34
variants per critic cell (the reference included). The matching no-critic cells get a screen of
34 variants over their own knobs under the same protocol (`nocritic_s_reg`: dropout, weight
decay, feed-forward width or cell size, gradient clipping), written down before it runs.

### 3b. Candidate new datasets (waiting for approval; nothing downloaded)

*Researched 2026-11-16 from the paper (arXiv 2203.06501, Tables 1 and 3) and each trace's own
page. Sizes marked "to confirm" were not stated on the page and will be checked (file listing
only) before any download.*

| Dataset | Public now? | License | Download needed | How the job arrival rate would be derived | Paper's intervals and Table 3 MAPE (LSTM / WGAN-gp) | Character |
| --- | --- | --- | --- | --- | --- | --- |
| Azure VM 2017 (`AzurePublicDatasetV1`) | Yes, direct links | CC BY 4.0 (repo `LICENSE`) | Only the VM table (one file; size to confirm, the full trace is 78.5 GB compressed and is not needed) | Count VMs by "timestamp VM created" per interval; 30 days | 10 / 30 / 60 min: 42.63 / 41.32, 29.35 / 27.48, 16.11 / 12.77 | Bursty VM workload |
| Azure VM 2019 (`AzurePublicDatasetV2`) | Yes, direct links | CC BY 4.0 | Only the VM table (size to confirm; full trace 156 GB compressed, not needed) | Same as 2017; 30 days | 30 / 60 min: 19.74 / 15.19, 13.5 / 10.82 | VM workload |
| Alibaba 2018 (`cluster-trace-v2018`) | Yes, after a short survey (the page says it can be skipped) | No license file; "research or study purpose" only | `batch_task.tar.gz`, 125 MB | Count batch tasks by start time per interval; 8 days | 5 / 10 min: 17.95 / 15.76, 16.90 / 14.67 | Data-center batch jobs, short trace |
| Wikipedia, as in the paper (WikiBench 2007 traces) | **No**: `wikibench.eu` no longer resolves | Unknown | - | Requests per interval | 10 / 30 min: 1.17 / 1.34, 1.75 / 3.43 | Strongly seasonal |
| Wikipedia, substitute (Wikimedia hourly pageview dumps) | Yes | CC0 | Hourly project-level files (small; size to confirm) | Sum of page views per hour for one project (for example `en.wikipedia`) | Not in the paper: hourly only, so the paper's 10 and 30 min cannot be matched | Strongly seasonal |
| Facebook (Hadoop traces, Chen et al. 2011) | Only 1-day samples in the SWIM repository | None stated | Small | Jobs by submit time | 5 / 10 min: 47.20 / 42.11, 43.68 / 39.31 | Bursty, but one day is too short to split 60/20/20 |

**Recommendation (at most two).**

1. **Azure VM 2017**: openly licensed, in the paper at three intervals, and a different kind of
   workload (VM creations, bursty) from the two we have.
2. **Wikipedia hourly page views** (the substitute): CC0 and strongly seasonal, which is the
   property we want for tying "where does the Transformer help" to workload character. It is
   **not** the paper's Wikipedia trace and has no Table 3 counterpart; the report must say so.

Not recommended: Alibaba 2018 has no open license (the assignment asks for open-source data),
Facebook is too short, and the paper's own Wikipedia trace is no longer online. One thing to
know about Table 3: the paper's Table 1 lists Azure Functions at 30 and 60 minutes, while its
Table 3 reports 5 and 10 minutes, so the paper is not consistent about that dataset's interval.
