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

#### Results of 1a and 1b (2026-11-16)

All 40 runs finished; none diverged. Full table: `results/v2/summary/released_code.md`
(per-run values in `released_code_runs.csv`). Test MAPE in %, mean ± SD over 5 seeds.

| Google 10 min | LSTM baseline | Paper's final model | Fair? |
| --- | --- | --- | --- |
| Paper's Table 3 | 11.49 | 10.58 | - |
| Paper settings, reported runs (best validation epoch) | 22.69 ± 0.34 | 22.62 ± 1.11 | yes |
| 1b data handling only, last epoch, true units | 25.62 ± 0.59 | 25.69 ± 1.30 | no |
| 1b data handling only, last epoch, **released code's units** | 24.86 ± 0.59 | 24.94 ± 1.25 | no |
| 1b, same runs at their best validation epoch, true units | 22.07 ± 0.47 | 23.48 ± 0.97 | no |
| 1a full released profile, last epoch, true units | 27.69 ± 2.83 | 30.06 ± 1.30 | no |
| 1a full released profile, last epoch, released code's units | 26.84 ± 2.70 | 29.22 ± 1.26 | no |

| Azure Functions 5 min | LSTM baseline | Paper's final model | Fair? |
| --- | --- | --- | --- |
| Paper's Table 3 | 1.63 | 3.05 | - |
| Paper settings, reported runs (best validation epoch) | 2.01 ± 0.05 | 2.13 ± 0.06 | yes |
| 1b data handling only, last epoch, true units | 2.51 ± 0.37 | 2.28 ± 0.10 | no |
| 1b data handling only, last epoch, released code's units | 5.19 ± 0.74 | 4.70 ± 0.19 | no |
| 1a full released profile, last epoch, true units | 2.44 ± 0.31 | 3.82 ± 0.03 | no |
| 1a full released profile, last epoch, released code's units | 5.03 ± 0.62 | 7.92 ± 0.08 | no |

The rows marked "no" use per-split scaling (the test part's own minimum and maximum) and are
scored on 48 fewer test points (windows inside the split). They explain or fail to explain the
gap; they are not improvements and must not be compared with the fair rows as if they were.

**Verdict on the hypothesis: rejected**, by the rule fixed in advance. On Google 10 min the
data-handling bundle in the released code's units gives 24.9 for both configurations, above
the rejection thresholds (19.6 and 19.9) and in fact *higher* than our reported values. Data
handling does not explain the paper's 10.6-11.5.

What the two experiments do show:

- **Reporting the last epoch costs 2-5 MAPE points on Google** (25.6 at the last epoch against
  22.1 at the best validation epoch for the same LSTM runs), because the models overfit after
  the first few hundred epochs (code review, finding 6).
- **Per-split scaling with windows inside the split changes little** when the epoch is chosen
  the same way (LSTM 22.07 against 22.69; final model 23.48 against 22.62).
- **The released code's un-scaling** hardly moves Google (25.6 to 24.9) and doubles Azure
  (2.5 to 5.2), as the zero-training check predicted.
- **The full released profile is worse than the paper settings on both datasets**, most of all
  for the final model (Google 30.1, Azure 3.82). Which of its items does the damage is what
  step 1c (one item at a time) will tell.
- **One direction does match the paper:** on Azure Functions 5 min the paper has its own model
  behind the LSTM (3.05 against 1.63), and so does the full released profile (3.82 against 2.44).

So the gap to Table 3 on Google remains unexplained by anything in the released code that we
can run. The remaining candidates are a different input series (no variant of the public trace
we tried is smooth enough, `results/summary/google_variants.md`) or settings that are in
neither the paper nor the repository.

Provenance: these runs started while later commits were being prepared, so their result files
record commit `5f0cdce` or a later one, some with `dirty: true`. The code path they exercise
did not change in those commits (checked by the default-path reproduction after each).

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

### 2a (matched side). `nocritic_reg_transformer` and `nocritic_reg_lstm`

*Written as the screens started (2026-11-16); the screen files were committed before.*

The no-critic Transformer and LSTM with MADGRAD, on both datasets, under the same protocol as
the critic screen (tuning seeds 0 and 100, `train.patience: 150`, validation MAPE only). Each
gets 34 variants, the number the critic side gets over its five stages: the reported setting
plus 33 of the 35 other combinations, drawn with a fixed seed, of

- Transformer: dropout {0.1 (reported), 0, 0.2, 0.3}, weight decay {0, 1e-4, 1e-3},
  feed-forward width {2048 (reported), 512, 128};
- LSTM: output dropout {0 (reported), 0.1, 0.2, 0.3}, weight decay {0, 1e-4, 1e-3},
  gradient clipping {off (reported), 1.0, 0.1}.

*Forward:* per architecture, the variant with the lowest average score, if it beats the
reported setting by the same noise rule as on the critic side (better in both cells and more
than 1% lower on average; there are 2 cells per architecture here); otherwise the reported
setting stays.

### 2a, stage 1 result: `critic_s1_weight` and `nocritic_reference`

*Recorded 2026-11-16. 64 + 8 trials, none diverged. Files: `results/v2/critic_s1_weight/`
(`screen.md`, `screen_cells.csv`, `screen_variants.csv`, `trials/`) and
`results/v2/nocritic_reference/`.*

In plain terms: the critic does best when it is turned down almost to nothing, and even then it
only matches the model that has no critic.

Average change in validation MAPE against the reported weight 1 (negative is better), and the
validation MAPE (%) in each cell (mean of tuning seeds 0 and 100):

| weight | average change | Google TF | Google LSTM | Azure TF | Azure LSTM |
| --- | --- | --- | --- | --- | --- |
| 1 (reported) | 0.0% | 18.47 | 19.99 | 2.478 | 2.912 |
| 0.3 | -3.0% | 18.75 | 19.92 | 2.478 | 2.548 |
| 0.1 | -4.0% | 18.49 | 19.90 | 2.440 | 2.516 |
| 0.03 | -4.2% | 18.77 | 19.32 | 2.463 | 2.517 |
| 0.01 | -5.3% | 18.43 | 19.85 | 2.420 | 2.424 |
| 0.003 | -4.3% | 18.52 | 19.85 | 2.426 | 2.505 |
| 0.001 | -5.6% | 18.03 | 19.83 | 2.452 | 2.416 |
| 0 (critic has no influence) | -3.1% | 18.47 | 19.88 | 2.455 | 2.601 |
| no critic, own tuned settings | | 18.22 | 19.49 | 2.392 | 2.491 |

What this shows:

- **By the rule, `w_best` = 0.001 and `w_knob` = 0.01** (the largest weight whose average is
  within 1 point of the best: -5.3% against -5.6%). Both pass the noise rule against weight 1
  (better in all 4 cells, more than 1% lower on average).
- **Most of the gain is one cell.** Azure LSTM improves by 12-17% at every weight below 1,
  because the reported weight makes that cell unstable (seed gap 0.88 at weight 1, 0.01-0.3
  below). In the other three cells every weight is within about 3% of weight 1, which is inside
  the seed-to-seed spread.
- **There is no dose-response.** From 0.3 down to 0.001 the averages lie between -3.0% and
  -5.6% in no clear order, and weight 0 (critic trained, no influence) already gives -3.1%.
  Less critic is never worse; that is the pattern of a term that adds noise, not information
  (code review, finding 1).
- **No weight beats the no-critic model.** The best critic weight per cell against the
  no-critic reference: Google TF 18.03 vs 18.22, Google LSTM 19.32 vs 19.49, Azure TF 2.420 vs
  2.392, Azure LSTM 2.416 vs 2.491. All four differences are smaller than the seed gap in the
  same cells (0.1-0.55), and the critic numbers are the best of 8 variants while the reference
  is one setting. (The critic and no-critic cells use their own tuned learning rates, so this
  row is a level to compare with, not a controlled contrast.)

Carried forward: stages 2, 3 and 5 use weight 0.01 (`w_knob`); stages 2 and 5 also use weight 1
as planned. Their files (`configs/screens/critic_s2_warmup.yaml`, `critic_s3_knobs.yaml`,
`critic_s5_input.yaml`) are committed with this note, before they run. Stage 4 follows stage 3.
Each later stage is compared with the stage-1 trials `w0.01` and `w1` of the same cell and
tuning seeds; the per-stage `screen.md` files show changes against the stage's first variant
only.

### 2a (diagnosis). `critic_diagnostics`: per-epoch logs of what the critic does

*Written before the runs (2026-11-16).* Four runs of the paper's final model (Transformer,
critic, MADGRAD), seed 1, with `--diagnostics`: `google-10m` and `azure_functions-5m`, each at
the reported adversarial weight 1 and at the stage-1 weight 0.01. Nothing is selected from
these runs. They are the source of the critic-diagnostics figure: per epoch, the L1 loss, the
critic term, the size of each one's gradient on the forecast, the angle between the two
gradients, the critic's loss, its Wasserstein estimate and the gradient penalty. Their test
MAPE is not reported as a result (the weight-1 runs repeat reported runs).

### 3. `more_data`: four more series through the unchanged pipeline

*Written before tuning started (2026-11-16).*

**Question.** Do the effects seen on the two reported series (the Transformer helps on Google
and not on Azure Functions; the critic hurts on both) hold on other workloads, and which
properties of a workload go with a Transformer gain?

**Series.** `google-30m` and `azure_functions-10m` (in the paper's Table 3; already
preprocessed), `azure_vm_2017-10m` (in the paper's Table 3; added in this branch) and
`wikipedia-60m` (not in the paper; a strongly seasonal web workload that stands in for the
paper's Wikipedia trace, which is offline). The 30 and 60 minute versions of Azure VM 2017 are
left out to keep the cost down.

**Protocol.** Exactly the pipeline of the reported runs, nothing new switched on:

1. `python -m cwp.tune --dataset D --interval I --arch A --critic C --optimizer O` for all 8
   configurations of each series: the grid of `configs/tuning.yaml` (32 trials each, tuning
   seed 0, lowest validation MAPE wins), written to `configs/tuned/`.
2. `python -m cwp.train ... --seed 1..5 --exp more_data` for the 8 configurations: 160 runs in
   `results/v2/more_data/`. The reported runs in `results/` and `results/summary/` are not
   touched, and `configs/plan.yaml` stays as it is.

**Reporting.** `python -m cwp.evaluate v2 more_data --name more_data` (per-series tables next
to the paper's values where it has them), `python -m cwp.analysis effects --exp more_data`
(factorial effects with Holm correction), and the effects across all six series together with
the reported runs (same pipeline and settings, labeled by source). The workload analysis
relates each series' Transformer effect to the training-part properties in
`results/summary/dataset_stats.md` (seasonal strength, autocorrelation, spikiness). With six
series this is a description, not a test. Diverged runs are reported and counted, not dropped.

### 4a. `seeds_6_10`: five more seeds of the reported design

*Written before the runs (2026-11-16).* The 8 configurations on `google-10m` and
`azure_functions-5m` at their tuned settings, seeds 6 to 10, saved in `results/v2/seeds_6_10/`
(80 runs). Nothing is selected. The report gives every borderline effect twice: with the 5
reported seeds (`results/summary/`, unchanged) and with all 10 (`results/v2/summary/`,
labeled). Ten seeds also make the Wilcoxon test usable (with 5 it cannot go below 0.0625).
Effects that were significant with 5 seeds and are not with 10, or the other way round, are
both reported.

### 4b. `adam_control`: the critic with standard Adam

*Written before the runs (2026-11-16).* In the reported design, switching the critic on with
Adam also switches Adam to the paper's WGAN settings (beta1 = 0), so those cells change two
things at once (code review, finding 2). This control trains the two critic + Adam
configurations with standard Adam (`optim.adam_with_critic.enabled=false`), tuned learning
rate reused, both datasets, seeds 1 to 5, in `results/v2/adam_control/` (20 runs). It is
compared with the reported critic + Adam cells (beta1 = 0) and the reported no-critic + Adam
cells. The learning rate was tuned under beta1 = 0, which favors the reported cells; the
report says so.

**`critic_diagnostics` result (recorded 2026-11-16).** Files: `results/v2/critic_diagnostics/`
(weight 1), `results/v2/critic_diagnostics_w0p01/` (weight 0.01; a first attempt under a name
with a dot in it was refused by the tool before training, so nothing was lost), figure
`figures/critic_diagnostics.pdf`. Medians over the 1,000 epochs:

| run (final model, seed 1) | critic gradient / L1 gradient | cosine between them | Wasserstein estimate |
| --- | --- | --- | --- |
| Google, weight 1 | 0.155 | -0.046 | 0.0005 |
| Google, weight 0.01 | 0.009 | 0.020 | 0.017 |
| Azure, weight 1 | 0.268 | 0.077 | 0.008 |
| Azure, weight 0.01 | 0.006 | 0.026 | 0.018 |

In plain terms: at the reported weight the critic pushes the forecaster with a force about a
sixth to a quarter of the L1 loss's, in a direction unrelated to it (cosine near 0), and on
Google that push swings by a factor of 10 from epoch to epoch. The critic's own estimate of
the distance between real and forecast sequences stays near zero: the forecasts already look
like real sequences to it, so it has nothing to teach. At weight 0.01 the push is 1% of the L1
gradient and steady. The Google weight-1 run repeats the reported run exactly (test MAPE
24.19%). These are single runs chosen for the figure; their test values are not results.

### 2a (matched side) result: `nocritic_reg_lstm` and `nocritic_reg_transformer`

*Recorded 2026-11-16. 136 + 136 trials, none diverged. Files: `results/v2/nocritic_reg_lstm/`
and `results/v2/nocritic_reg_transformer/` (`screen.md` lists all 34 variants of each).*

In plain terms: giving the no-critic models the same number of extra tries changed almost
nothing. Their reported settings were already close to the best we can find.

| architecture | best of the 33 new variants | Google | Azure | average | passes the rule? |
| --- | --- | --- | --- | --- | --- |
| LSTM | gradient clipping at 1.0 (no output dropout, no weight decay) | -1.7% | -0.7% | -1.2% | yes (both cells, more than 1%) |
| Transformer | weight decay 1e-4, feed-forward width 128 | -1.6% | +0.2% | -0.7% | no |

- **LSTM:** the three best variants are the three with clipping at 1.0 and no weight decay
  (-1.2%, -1.2%, -0.8%); 30 of the 33 new variants are worse than the reported setting.
  Carried forward: `train.grad_clip: 1.0` for the no-critic LSTM. A gain of 1.2% as the best
  of 33 tries is about what selection alone would produce at this noise level (seed gap in
  these cells: 2.6% and 7.5% of the value), so little should be expected of it on the test part.
- **Transformer:** nothing passes; the reported setting stays. 29 of the 33 new variants are
  worse than it. More dropout, no dropout and weight decay 1e-3 all hurt.

### 2a, stages 2, 3 and 5 result

*Recorded 2026-11-16. 32 + 128 + 32 trials. Files: `results/v2/critic_s2_warmup/`,
`critic_s3_knobs/`, `critic_s5_input/`; all stages in one table against the reported weight 1
in `results/v2/summary/critic_screen.md` and against stage 1's weight 0.01 in
`results/v2/summary/critic_screen_vs_w0p01.md`
(`python -m cwp.analysis screens ... --reference critic_s1_weight:w1 --name critic_screen`).*

In plain terms: once the critic's weight is turned down to 0.01, nothing else we changed about
the critic made a difference we can tell from noise. A critic that sees more of the data did
not help either, and at the reported weight it made things worse.

Average change in validation MAPE over the 4 cells (negative is better):

| stage | variant | vs. weight 1 (reported) | vs. weight 0.01 (stage 1) | beats weight 0.01 by the rule? |
| --- | --- | --- | --- | --- |
| 1 | weight 0.001 | -5.6% | -0.3% | no |
| 1 | weight 0.01 | -5.3% | 0 | – |
| 2 | weight 0.01, 50 epochs L1 only, then a 100-epoch ramp | -5.5% | -0.3% | no (2 of 4 cells) |
| 2 | weight 0.01, 50 epochs L1 only | -5.1% | +0.2% | no |
| 2 | weight 1, 50 epochs L1 only, then a 100-epoch ramp | -2.8% | +2.6% | no |
| 2 | weight 1, 50 epochs L1 only | -2.7% | +2.7% | no |
| 3 | best of 16: 5 critic steps, penalty weight 1, width 32, same learning rate | -5.3% | -0.03% | no (1 of 4 cells) |
| 3 | median of 16 | -3.3% | +2.0% | no |
| 3 | worst of 16: 5 critic steps, penalty 10, width 128, critic learning rate x4 | diverged | diverged | no |
| 5 | first-difference critic, weight 0.01 | -3.4% | +2.0% | no |
| 5 | conditional critic, weight 0.01 | -2.9% | +2.5% | no |
| 5 | conditional critic, weight 1 | +2.0% | +7.6% | no |
| 5 | first-difference critic, weight 1 | +4.1% | +9.9% | no |

What this shows:

- **Warm-up does not rescue the reported weight.** Letting the forecaster train alone first
  and then switching the critic on at weight 1 recovers about half of what turning the weight
  down recovers (-2.7% against -5.3%), and is worse than weight 0.01 in every cell.
- **The critic's own settings do not matter at weight 0.01.** No combination of 16 passes the
  rule. The pattern inside the 16 is that 1 critic step per forecaster step is worse than 5
  (all eight `n1` combinations are worse than weight 0.01 in every cell): a less-trained
  critic is a noisier one.
- **One failure.** 5 critic steps, penalty 10, width 128 and a critic learning rate 4 times
  the forecaster's diverged on the Google LSTM (validation MAPE 1.5 million % on one tuning
  seed) and reached 3.24% on the Azure LSTM. It is kept in the tables as it is.
- **A richer critic does not help.** The conditional critic and the first-difference critic at
  weight 0.01 land where weight 0 does (-2.9% and -3.4% against -3.1% for a critic with no
  influence). At weight 1 both are worse than the reported critic. So the reason the critic
  does not help is not that it sees too little.

Carried forward from stage 3 by its rule: the stage-1 setting (weight 0.01, reported critic
settings). Stage 4 (`configs/screens/critic_s4_norm.yaml`, committed with this note) adds
spectral and layer normalization to it.
