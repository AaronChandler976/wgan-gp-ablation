# Notes for the report

Suggested text, tables and figures for each section of the final report. Every number names
the file in `results/` it comes from. Numbers from different setups are never in one table
without a label; keep it that way when copying into LaTeX.

Three kinds of runs appear below. Use these names in the report so a reader can tell them
apart:

| name | what it is | where |
| --- | --- | --- |
| **reported runs** | the original 80 runs: 8 configurations, Google 10 min and Azure Functions 5 min, seeds 1 to 5, paper settings | `results/*.json`, `results/summary/` |
| **v2 experiments** | everything added later, each under its own name | `results/v2/<name>/`, tables in `results/v2/summary/` |
| **screens** | short tuning runs scored on the validation part only | `results/v2/critic_s*`, `nocritic_*` |

The plan, the selection rule and the result of every v2 experiment are in
`docs/EXPERIMENTS_V2.md`. What the released code does differently from the paper is in
`docs/REPLICATION_NOTES.md`; the code review is in `docs/CODE_REVIEW.md`.

---

## 1. Problem statement

**In plain terms.** A cloud provider wants to know how many jobs will arrive in the next few
minutes so it can start the right number of machines. Arbat et al. (AAAI 2022) proposed a
forecaster, the WGAN-gp Transformer, and reported that it beats the LSTM-based LoadDynamics.
But their model differs from the baseline in three ways at once, so the paper cannot say which
change is responsible.

Suggested text:

> The WGAN-gp Transformer replaces LoadDynamics' LSTM with a Transformer, adds an adversarial
> critic trained with the WGAN-gp objective, and swaps Adam for MADGRAD. The paper compares
> only the two end points. We ask which of the three changes produces the improvement, by
> training all 2 x 2 x 2 = 8 combinations under one pipeline, one tuning budget and one
> evaluation protocol, and estimating the effect of each change and of their interactions.

Points to make:

- One-step-ahead forecasting of job arrivals per interval; metric is test MAPE, as in the
  paper, with MAE, RMSE and MASE alongside (`results/summary/summary.md`).
- Why an ablation is needed: three changes, two end points, no way to attribute.
- A second question the report can answer: does the paper's result reproduce at all from its
  description and its released code? (Section 6.5.)

## 2. Related work

Short paragraphs, one per item; nothing here needs numbers from our runs.

- **LoadDynamics** (Jayakumar et al., IPDPS 2020): stacked LSTM, hyperparameters chosen per
  workload by Bayesian optimization. The baseline of the paper we study.
- **WGAN-gp Transformer** (Arbat et al., AAAI 2022): the paper under study.
- **WGAN and WGAN-gp** (Arjovsky et al. 2017; Gulrajani et al. 2017): the critic and the
  gradient penalty. Note the usual settings (5 critic updates per generator update, penalty
  weight 10, Adam with beta1 = 0), which the paper adopts.
- **Adversarial training for time-series forecasting** (for example Wu et al. 2020,
  "Adversarial Sparse Transformer"): the motivation for adding a critic to a forecaster.
- **MADGRAD** (Defazio and Jelassi 2021).
- **Transformers for forecasting, and doubts about them** (Vaswani et al. 2017; Zeng et al.
  2023, "Are Transformers Effective for Time Series Forecasting?"): our finding that the
  Transformer's gain is small fits this line of work.
- **Replication and ablation practice** (Lipton and Steinhardt 2018, "Troubling Trends in
  Machine Learning Scholarship"; Bouthillier et al. 2021 on variance from seeds and tuning):
  the reason for equal tuning budgets, several seeds and corrected tests.

Check each citation against the original before it goes into the bibliography.

## 3. Solution (method)

**In plain terms.** We built one training program with three on/off switches and ran every
combination the same way, so that a difference between two results can only come from the
switch that differs.

- **The three switches.** `--arch lstm|transformer`, `--critic none|wgan-gp`,
  `--optimizer adam|madgrad`. Table: the switch table at the top of `README.md`.
- **Same everything else.** Same data split (first 60% train, next 20% validation, last 20%
  test), same scaling (fitted on the training part only), same loss (L1, plus the critic term
  when the critic is on), same 1,000 epochs, same test points for every model.
- **Same tuning budget.** 32 grid trials per configuration and series, lowest validation MAPE
  wins (`configs/tuning.yaml`, results in `configs/tuned/`).
- **Selection never touches the test part.** Hyperparameters, the reported epoch (best
  validation MAPE) and every later choice use validation only.
- **Paper over released code.** Where the two disagree we follow the paper
  (`docs/REPLICATION_NOTES.md`, first table); the released code's choices are separate,
  labeled experiments.
- **Statistics.** Effects are estimated on log MAPE from the full factorial, per seed, so every
  effect has a per-seed value. Tests: t-test over seeds, Wilcoxon signed-rank with 10 seeds,
  Holm correction, and a random-effects average across series with a test of whether the
  effect differs between series (`src/cwp/analysis.py`).

Suggested figure: a small diagram of the 2 x 2 x 2 cube with LoadDynamics and the paper's model
at opposite corners (to be drawn in LaTeX; no data needed).

## 4. List of experiments

| # | experiment | runs | question | result section |
| --- | --- | --- | --- | --- |
| E1 | Reported factorial (2 series, 8 configurations, 5 seeds) | 80 | Which part helps? | 6.1 |
| E2 | Seeds 6 to 10 (`seeds_6_10`) | 80 | Do the conclusions hold with 10 seeds? | 6.1 |
| E3 | Four more series (`more_data`) | 160 + 32 tunings | Do they hold on other workloads? | 6.2 |
| E4 | Standard-Adam control (`adam_control`) | 20 | Is the critic's harm an artifact of Adam's beta1 = 0? | 6.3 |
| E5 | Critic screen, 5 stages, 34 variants (validation only) | 272 trials | Can any critic setting be made to help? | 6.4 |
| E6 | Matched no-critic screen, 34 variants (validation only) | 272 trials | Fair budget for the other side | 6.4 |
| E7 | Critic diagnostics | 4 | What does the critic do during training? | 6.4 |
| E8 | Recheck of the screen winners on fresh tuning seeds | 80 trials | Was the screen's gain real? | 6.4 |
| E9 | Critic v2 full design (`critic_v2`, `critic_v2_diffs`) | 120 | With its best setting, does the critic help on test? | 6.4 |
| E10 | Released code, all choices and data handling only (`released_full`, `released_data`) | 40 | Does the released code explain the paper's numbers? | 6.5 |
| E11 | Released code, one choice at a time (`released_item_*`) | 78 | Which choice matters? | 6.5 |
| E12 | Provisioning analysis (no new runs) | - | Do the differences matter for autoscaling? | 6.6 |
| E13 | Inference timing sweep | - | Is the Transformer faster? | 6.7 |

## 5. Setup and data

- **Hardware and software.** One NVIDIA RTX 5070 Ti, PyTorch 2.11 (header of
  `results/summary/timing_sweep_cuda.md`). Each result file records the commit, the settings
  and the seed.
- **Series** (`results/summary/dataset_stats.md`; properties are of the training part):

| series | interval | points | days | naive MAPE | seasonal strength | in the paper? |
| --- | --- | --- | --- | --- | --- | --- |
| Google Cluster 2011 | 10 min | 4,176 | 29 | 25.2 | 0.32 | yes |
| Google Cluster 2011 | 30 min | 1,392 | 29 | 23.8 | 0.30 | yes |
| Azure Functions 2019 | 5 min | 4,032 | 14 | 3.9 | 0.74 | yes |
| Azure Functions 2019 | 10 min | 2,016 | 14 | 4.1 | 0.75 | yes |
| Azure VM 2017 | 10 min | 4,320 | 30 | 36.3 | 0.32 | yes (our rebuild of the series is not checked against the paper's) |
| Wikipedia pageviews | 60 min | 4,344 | 181 | 4.2 | 0.91 | no (stands in for the paper's Wikipedia trace, which is offline) |

  All are public. Sources and licenses: `data/README.md`.
- **Two kinds of workload.** Google and Azure VM are bursty and hard (a naive forecast is off
  by 24 to 36%); Azure Functions and Wikipedia are smooth and seasonal (about 4%). Say this
  early: it explains why MAPE values differ by a factor of ten between series.
- **Baselines.** Naive, seasonal naive and moving averages on the same test points
  (`results/summary/summary.md`). Every trained model beats the naive forecast on both
  reported series (Google 27.6, Azure Functions 3.45).

## 6. Results

### 6.1 Which part helps? (reported runs, then 10 seeds)

**In plain terms.** The Transformer helps on Google and does nothing on Azure Functions. The
critic makes things worse on both. MADGRAD does little.

Table A, test MAPE (%) of all 8 configurations, mean ± SD over 5 seeds: copy both tables of
`results/summary/summary.md`. The rows the text needs:

| configuration | Google 10 min | Azure Functions 5 min |
| --- | --- | --- |
| LSTM, no critic, Adam (LoadDynamics) | 22.69 ± 0.34 | 2.014 ± 0.054 |
| Transformer, no critic, MADGRAD (lowest on both) | 20.66 ± 0.21 | 2.012 ± 0.021 |
| Transformer, critic, MADGRAD (the paper's model) | 22.62 ± 1.11 | 2.133 ± 0.057 |
| naive forecast | 27.60 | 3.452 |

Table B, main effects (change in test MAPE from switching the part on; negative is better).
5 seeds: `results/v2/summary/effects_reported.md`. 10 seeds:
`results/v2/summary/effects_reported+seeds_6_10.md`. Holm correction over the 6 main effects.

| series | part | 5 seeds | Holm p | 10 seeds (95% interval) | same sign | Holm p | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Google 10 min | Transformer | -4.8% | 0.012 | -5.2% (-6.3 to -4.1) | 10/10 | < 0.0001 | 0.002 |
| Google 10 min | critic | +4.6% | 0.094 | +4.4% (+2.9 to +6.0) | 10/10 | 0.0004 | 0.002 |
| Google 10 min | MADGRAD | -0.3% | 1.0 | -1.3% (-2.8 to +0.2) | 7/10 | 0.15 | 0.11 |
| Azure Functions 5 min | Transformer | +0.1% | 1.0 | +0.6% (-0.6 to +1.8) | 6/10 | 0.28 | 0.43 |
| Azure Functions 5 min | critic | +6.1% | 0.012 | +6.4% (+5.3 to +7.6) | 10/10 | < 0.0001 | 0.002 |
| Azure Functions 5 min | MADGRAD | -2.6% | 0.24 | -3.4% (-5.1 to -1.6) | 10/10 | 0.006 | 0.002 |

Suggested text:

> With five seeds the critic's harm on Google was significant before correction (p = 0.024)
> and not after (Holm p = 0.094). We therefore ran five more seeds. With ten, every seed shows
> the same sign for the Transformer effect on Google and for the critic effect on both series,
> and all three survive correction. No conclusion drawn from five seeds changed; MADGRAD's
> small gain on Azure Functions (-3.4%) became significant.

- The paper's model (all three on) is no better than LoadDynamics on Google (22.62 against
  22.69) and worse on Azure Functions (2.133 against 2.014). The best configuration on both is
  the Transformer **without** the critic.
- Interactions are small and none survives correction. At 10 seeds the two with p < 0.05
  before correction are both on Azure Functions: critic x optimizer (-2.3%, Holm 0.084) and
  the three-way interaction (+1.8%, Holm 0.13). MADGRAD softens the critic's harm for the
  LSTM there.

Figures: `figures/mape_by_configuration.pdf` (all runs), `figures/factorial_main_effects.pdf` (main effects,
5 seeds), `figures/forecast_best_worst.pdf` (what the forecasts look like).

### 6.2 Other workloads (`more_data`)

**In plain terms.** On four more series the same thing happens: the critic hurts on all six,
the Transformer is a little better on five of six, MADGRAD is a coin flip.

Table C: the per-series table and the pooled table in `docs/EXPERIMENTS_V2.md`, section
"3 result" (sources `results/v2/summary/more_data.md`, `effects_more_data.md`,
`pooled_reported+more_data.md`). Label the columns by source: the first two series are the
reported runs, the other four are `more_data`; same pipeline and settings.

| part | effect on each of the six series | pooled | 95% interval | p |
| --- | --- | --- | --- | --- |
| Transformer | +0.1, -4.8, -2.5, -1.3, -5.4, -6.2 | -3.3% | -5.8 to -0.6 | 0.015 |
| critic | +6.1, +4.6, +14.0, +5.6, +8.0, +4.2 | +5.9% | +4.4 to +7.5 | < 0.0001 |
| MADGRAD | -2.6, -0.3, +9.2, +0.2, +6.1, -1.3 | +0.6% | -1.3 to +2.6 | 0.55 |

(Order: Azure Functions 5 min, Google 10 min, Azure Functions 10 min, Azure VM 10 min,
Google 30 min, Wikipedia 60 min.)

Figure: `figures/effects_all_series.pdf`. This is the single best figure for the paper's main
claim.

Points to make:

- The critic's sign is the same on all six series and in every seed of the four new ones.
- Single-series tests on the new series are weak with 5 seeds (after Holm only the critic on
  Azure VM stays under 0.05); the pooled estimate is the statement to lean on.
- **Instability.** The LSTM with critic and MADGRAD diverges late in training on all four new
  series (last-epoch test MAPE in the millions of percent, column `mape_last_epoch` of
  `more_data.md`). Picking the epoch on validation hides this; reporting the last epoch, as the
  released code does, would not.
- **Workload properties** (`results/v2/summary/workload_reported+more_data.md`). The
  Transformer's gain does not follow seasonality, autocorrelation or burstiness (rank
  correlations between -0.2 and +0.03 over six series). The tempting story from two series,
  "the Transformer helps on bursty workloads", is not supported by six. Say so; it is an honest
  negative result.
- Against the paper: on Azure Functions 10 min our values are close to the paper's (1.80 to
  2.14 against 2.06 and 1.85); on Google 30 min ours are about twice the paper's (19.2 to 24.3
  against 9.12 and 8.34), as on Google 10 min.

### 6.3 Is the critic's harm an artifact of the Adam settings? (`adam_control`)

**In plain terms.** When the critic is on, the paper also changes Adam's settings, so in those
cells two things change at once. We reran them with ordinary Adam. The critic cells are still
no better than the no-critic cells.

Table D: `docs/EXPERIMENTS_V2.md`, section "4b result" (source
`results/v2/summary/adam_control.md`). One sentence in the text is enough:

> Training the critic with standard Adam instead of beta1 = 0 changes test MAPE by less than
> the seed-to-seed spread in three of four cells and leaves all four at or above their
> no-critic counterparts.

### 6.4 Can the critic be made to help?

**In plain terms.** We gave the critic every reasonable chance: a smaller weight, a warm-up,
different training settings, normalisation, and two richer designs. The best thing we found to
do with the critic is to turn its influence down until it is almost off.

Order of the argument:

1. **Diagnosis** (`results/v2/critic_diagnostics/`, figure `figures/critic_diagnostics.pdf`).
   At the paper's weight the critic's gradient is about a sixth to a quarter of the L1
   gradient (median ratio 0.155 on Google, 0.268 on Azure) and points in an unrelated
   direction (cosine -0.05 and 0.08). The critic's own distance estimate stays near zero: it
   cannot tell forecasts from real sequences, so its signal is noise added to the L1 gradient.
2. **Screen** (`results/v2/summary/critic_screen.md`, figure `figures/critic_weight_screen.pdf`).
   34 variants in 5 stages, 272 trials, validation only, with a rule fixed beforehand (better
   in at least 3 of 4 cells and more than 1% lower on average).

   | stage | what was varied | best variant | change in validation MAPE against the paper's setting |
   | --- | --- | --- | --- |
   | 1 | adversarial weight, 1 down to 0 | weight 0.001 | -5.6% |
   | 2 | warm-up and ramp | weight 0.01 with ramp | -5.5% |
   | 3 | critic updates, penalty weight, width, learning-rate ratio | 5 updates, penalty 1, width 32 | -5.3% |
   | 4 | spectral and layer normalisation | spectral | -3.9% |
   | 5 | conditional and first-difference critics | first differences at weight 0.01 | -3.4% |

   Three facts carry the argument: the smaller the weight the better; switching the weight to
   exactly zero already gives -3.1%; and no variant reaches the no-critic reference in the same
   cells (`results/v2/nocritic_reference/`). Most of the gain is in one cell (LSTM on Azure
   Functions, -12 to -17%).
3. **Fair budget for the other side** (`results/v2/nocritic_reg_lstm/`,
   `nocritic_reg_transformer/`). The no-critic models got 34 variants of their own knobs
   (dropout, weight decay, gradient clipping and so on). Gradient clipping at 1.0 passes the
   rule for the LSTM (-1.2%); nothing passes for the Transformer.
4. **Recheck and full design.** Results in section 6.4.1 below.

Report every variant, including the ones that failed (one stage-3 setting diverged on Google;
both richer critics at weight 1 were worse than the paper's critic). The full list is in
`critic_screen.md`.

#### 6.4.1 Critic v2 on the test part

**In plain terms.** After the search, the best way to use the critic is to turn it down
until it barely acts. Doing that makes the critic models better than in our reported runs, but
they still do not beat the same models without a critic.

Sources: `results/v2/summary/critic_v2.md`, `effects_critic_v2.md`, `pooled_critic_v2.md`,
`pairs_critic_v2.md`, `versus_reported_critic_v2.md`, the files ending in
`critic_v2_diffs+nocritic_of_critic_v2`, and the two recheck tables
(`results/v2/critic_recheck/screen.md`, `results/v2/nocritic_recheck/screen.md`). Full record:
`docs/EXPERIMENTS_V2.md`, "2b result" and "2c result".

Table D2 (suggested): effect of switching the critic on, test MAPE, 5 seeds.

| design | Google 10 min | Azure Functions 5 min | pooled |
| --- | --- | --- | --- |
| Paper's setting (weight 1) | +4.6% | +6.1% | +5.7% (4.2 to 7.1) |
| Critic v2 (weight 0.001), no-critic side given the same search | +3.2% (1.4 to 5.0), Holm 0.037 | +1.7% (-0.4 to 3.8), not significant | +2.5% (1.1 to 3.9) |
| Critic v2-diffs (critic sees first differences, weight 0.01) | +2.6% (0.3 to 4.9), Holm 0.13 | +0.9% (-0.3 to 2.2), not significant | +1.6% (0.0 to 3.3) |

Each row is a different setup; do not average across rows. Figure: `figures/critic_designs.pdf` shows the same
three rows with their intervals.

Points to make:

- **Recheck on unseen tuning seeds.** The screen's gain for weight 0.001 was -5.6% in
  validation MAPE; on fresh seeds the same cells give -3.4%, and -2.4% over all eight critic
  cells. About half of the screen's gain was selection. Say this openly: it is the reason the
  recheck was run.
- **On the test part the lower weight helps every critic cell** against our reported runs (by
  1 to 6% for critic v2, 1 to 8% for v2-diffs). The largest is the paper's final model on
  Google: 22.62 to 21.19 (v2) and 20.84 (v2-diffs).
- **It still does not beat no critic.** In 7 of 8 paired comparisons the critic v2 cell is
  worse than its no-critic partner; the exception is a tie (LSTM with MADGRAD on Azure, -0.015
  points, p 0.62). The best critic cell on Google (20.84) is behind the Transformer without a
  critic (20.66).
- **Fairness of the comparison.** The no-critic LSTM was given gradient clipping from its own
  34-variant search. The recheck did not confirm that this helps (-0.4%, 2 of 4 cells), and on
  the test part it moved the LSTM cells by -3.1% to +2.0%, none significant. Report this; it is
  part of why the critic effect on Google remains +3.2%.
- **Why the result is believable.** Stage 1 of the screen showed that a critic with weight
  exactly zero already gives most of the gain. A critic at weight 0.001 is close to no critic.
  Every step of the search moved toward removing the critic's influence.
- **Instability remains.** The LSTM with critic and MADGRAD still diverges late in training
  under critic v2 on both series. Selection on validation hides this in the MAPE tables, so
  state it in the text.
- Provisioning cost is unchanged (`autoscale_critic_v2.md`).

### 6.5 Does the released code explain the paper's numbers?

**In plain terms.** Our Google errors are about twice the paper's. The released code handles
data differently from the paper's description, in ways that could make errors look smaller.
We tested that explanation and it fails.

Table E: the two tables under "Results of 1a and 1b" in `docs/EXPERIMENTS_V2.md` (source
`results/v2/summary/released_code.md`). Figure: `figures/released_code_comparison.pdf`.

Points to make:

- List of differences between paper and released code: `docs/REPLICATION_NOTES.md`. The most
  important are scaling fitted separately on the test part, test values un-scaled with the
  validation part's range, the last epoch reported, 1 critic update instead of 5, penalty
  weight 0.1 instead of 10, and the sign of the critic term.
- **Hypothesis, written before the runs:** data handling explains the gap. **Rejected** by the
  rule fixed in advance: with the released data handling, in the released code's own units,
  Google 10 min gives 24.9 for both models (paper: 11.49 and 10.58; ours: 22.69 and 22.62).
- The full set of released choices is worse still (Google 27.7 LSTM, 30.1 final model).
- One thing does reproduce: on Azure Functions 5 min the paper has its own model behind the
  LSTM (3.05 against 1.63), and the released profile gives the same order (3.82 against 2.44).
- Reporting the last epoch instead of the best validation epoch costs 2 to 5 MAPE points on
  Google.
- These rows use the test part's minimum and maximum and are **not fair evaluations**; they
  are shown to explain, not to compete.

#### 6.5.1 One released choice at a time

**In plain terms.** We switched on the released code's choices one by one. Three of them break
the paper's final model on their own, two make it slightly better by weakening the critic, and
none brings the error near the paper's numbers.

Sources: `results/v2/summary/released_items.md`, `versus_reported_released_items.md`. Full
record with both tables: `docs/EXPERIMENTS_V2.md`, "1c result". 78 runs, seeds 1 to 3; 77 with
a finite test MAPE.

Table E2 (suggested): the paper's final model with one released choice switched on. Change in
test MAPE against our reported runs of the same seeds (Google 22.47, Azure 2.13).

| released choice | Google 10 min | Azure Functions 5 min |
| --- | --- | --- |
| Sign of the critic term (L1 plus the critic score) | diverges (57598) | diverges (580) |
| Learning rate 0.01 | 38.64 (+66%) | 6.19 (+184%; 1 of 3 runs not finite) |
| Positional encoding by position in the batch | 25.81 (+15%) | 3.33 (+56%) |
| Input value copied to every embedding dimension | 24.46 (+9%) | 2.25 (+5%) |
| No shuffling | 23.02 (+3%) | 2.17 (+2%) |
| Scaler fitted per split (not a fair evaluation) | 23.02 (+3%) | 2.20 (+3%) |
| Windows inside each split (48 fewer test points) | 22.69 (+1%) | 2.09 (-2%) |
| 1 critic step, penalty weight 0.1 | 21.01 (-6%) | 2.14 (+1%) |
| Tanh critic applied to each value | 20.92 (-7%) | 2.04 (-4%) |

Points to make:

- With 3 seeds only the large rows are reliable (the first three). Say so in the caption.
- The sign of the critic term is the most consequential discrepancy between paper and code.
  With the paper's critic it makes training run away. In the full released profile it did not
  diverge; a likely reason is that the released critic's Tanh output bounds its score. We did
  not test that combination separately, so word it as a likely reason.
- The two choices that lower the error both weaken the critic, and they land next to the
  Transformer without a critic (20.66 on Google). This is the same message as section 6.4.
- For the LSTM baseline the four applicable choices are within noise, except per-split scaling
  on Azure (+16%, worse in 3 of 3 seeds).
- No single choice brings Google near the paper's 10 to 11%. The lowest value is 20.92.

### 6.6 Does it matter for autoscaling?

**In plain terms.** If one machine is started per forecast job, how often is the system short,
and how much spare capacity is needed to make that rare? The answer depends on the workload
far more than on the model.

Source: `results/v2/summary/autoscale_reported.md` (5 seeds), `autoscale_more_data.md`; figure
`figures/autoscale_tradeoff.pdf`.

| series | extra capacity for at most 5% short intervals (range over the 8 configurations) | idle machines at that setting |
| --- | --- | --- |
| Google 10 min | 54 to 61% | 45 to 53% |
| Azure Functions 5 min | 5.2 to 5.8% | 5.0 to 5.4% |
| Google 30 min | 55 to 67% | 45 to 61% |
| Azure Functions 10 min | 4.8 to 11.0% | 4.4 to 8.9% |
| Wikipedia 60 min | 2.8 to 3.2% | 3.3 to 3.7% |
| Azure VM 10 min | over 150% (some runs do not reach 5% at 200%) | over 100% |

- On Google the lowest-MAPE configuration is not the cheapest to provision: MAPE rewards
  forecasts that sit low, which leaves more intervals short (under-provisioned in 58.9% of
  intervals against 52.3% for LoadDynamics; `autoscale_reported_runs.csv`).
- The headroom is read off the test part, so it describes each forecast's trade-off curve and
  is not a tuned policy. Say this in the caption.

### 6.7 Inference time

Source: `results/summary/timing_sweep_cuda.md`, `timing_sweep_cpu.md`; figure
`figures/timing_sweep.pdf`. At a history of 672 steps on the GPU a fused LSTM takes 0.18 to
0.58 ms per prediction and the Transformer 0.29 to 0.35 ms; the same LSTM run one step at a
time takes 9.8 to 18.8 ms. The Transformer is faster only than an LSTM that is not run as one
fused kernel.

## 7. Conclusions

1. **The critic does not help.** With the paper's setting it raises test MAPE on all six
   series (pooled +5.9%, 95% interval +4.4 to +7.5), in every seed of the ten-seed design, and
   with either Adam setting. After a 34-variant search for the best way to use it, checked on
   unseen tuning seeds, it still raises the error (+2.5% pooled over the two series, +3.2% on
   Google with Holm p 0.037). The best critic is the one with the least influence.
2. **The architecture is the only part with a real gain, and it is modest and uneven.** Pooled
   -3.3%; -5.2% on Google 10 min (10 of 10 seeds), nothing on Azure Functions 5 min. The gain
   does not follow seasonality or burstiness across six series.
3. **MADGRAD is not a reliable improvement** (pooled +0.6%, differs by series).
4. **The paper's absolute numbers on Google do not reproduce** from the paper's description or
   from its released code. The released data handling does not explain the gap, and no single
   released choice does either; three of them break the model on their own.
5. **For provisioning, the workload matters far more than the model.**
6. **Recommendation that follows:** for one-step workload forecasts of this kind, train the
   Transformer with a plain L1 loss. With MADGRAD it has the lowest mean test MAPE of the
   eight configurations on five of six series (on Azure Functions 5 min it ties with the LSTM,
   2.01 for both); on Azure Functions 10 min it loses clearly to the LSTM without a critic
   (2.10 against 1.80). The critic adds training cost and instability without a gain.
   Sources: `results/summary/`, `results/v2/summary/more_data.md`.

## 8. Limitations

- One-step-ahead forecasts only; the critic sees one forecast value appended to the history. A
  critic may have more to say about multi-step forecasts. Not tested.
- Five seeds on the four added series; ten only on the two reported ones.
- Six series, four traces. The workload-property analysis is a description, not a test.
- The search spaces are ours. The paper's grid (its Table 2) is only partly specified, and the
  released code has no search.
- The Azure VM series is our rebuild from the public trace; the Wikipedia series is a
  substitute for the paper's.
- The provisioning analysis is a simple one-machine-per-job model with a fixed headroom, read
  off the test part.
- The critic screen covers the variants listed in `docs/EXPERIMENTS_V2.md`; other critic
  designs (sequence critics over multi-step outputs, feature matching) are untested.
- The critic search and the critic v2 design were run on the two reported series only, with 5
  seeds. On Azure Functions the remaining harm (+1.7%) cannot be told from zero.
- The one-choice-at-a-time runs of the released code use 3 seeds; only the large effects are
  reliable.
- The last-epoch error was not recorded for the 80 reported runs, so late divergence is
  documented from the follow-up runs only.
- We could not obtain the paper's exact Google input series or its hyperparameter values, so
  the reason for the gap to its Table 3 stays open.

## 9. Future work

- Multi-step forecasts with a critic over the whole forecast sequence, where an adversarial
  loss has a plausible role.
- Probabilistic forecasts and a provisioning-aware loss (quantile or asymmetric), since
  section 6.6 shows that under-provisioning, not MAPE, is what costs.
- Newer simple baselines (linear models on the window, patch-based Transformers).
- More traces (Alibaba, newer Google traces) to test the workload-property question properly.
- Contact the paper's authors for the input series and settings behind Table 3.
