# Code review (Phase 0)

A review of `src/cwp/` for bugs and for anything that could quietly bias the comparison between
the 8 configurations. Reviewed at commit `d5e70a8` (the code that produced the 80 reported
runs, which were trained at `38a9461`). Line numbers refer to the `v2-experiments` branch.

**In one paragraph.** No bug was found that invalidates the 80 reported runs: the WGAN-gp loop,
the metrics and the statistics do what the README says. One reproducibility bug was found and
fixed (finding 14); it does not change any reported number. Three things affect how the
results should be read. (1) The critic's term is *not* drowning out the L1 loss, which was the
suspicion: its pull on the predictions is 6-34% as strong as the L1 loss's. But that pull
points in a direction unrelated to the L1 loss, and the critic can barely tell real from
predicted values, so it acts as noise on the forecaster (finding 1). (2) Hyperparameters were
chosen from one noisy trial each, and the differences between the top trials are smaller than
the seed-to-seed noise (finding 3). (3) Turning the critic on also changes Adam's settings, so
the "critic" and "optimizer" effects are partly mixed in the Adam cells (finding 2). The 80
result files and their summaries are untouched.

Severity labels: **bug** (the code does something other than intended), **bias risk** (the code
is correct but the comparison may favor one side), **minor** (cosmetic, power or documentation).

| # | Finding | Severity | Changes the reported numbers? |
| --- | --- | --- | --- |
| 1 | The critic's gradient is 6-34% of the L1 gradient and unrelated to it in direction; the critic barely separates real from predicted | bias risk | No (helps explain them) |
| 2 | Critic on also switches Adam to beta1 = 0 for the forecaster | bias risk | No; mixes two effects in the Adam cells |
| 3 | Tuned settings are within tuning noise (one seed per trial) | bias risk | No; adds noise to every cell |
| 4 | The critic's own settings were never tuned and it shares the forecaster's learning rate | bias risk | No (known; Phase 2) |
| 5 | Critic batches share the batch-order random generator, so seed pairing is weak | minor | No |
| 6 | On Google, 32 of 40 runs select an epoch in the first 300 of 1,000, then overfit | minor | No |
| 7 | Some critic runs diverge late; `Infinity` is written into result JSON | minor | No |
| 8 | Wilcoxon column is blank by design (needs 6 seeds) | minor | No |
| 9 | Factorial p-values are not corrected for multiple tests | minor | Google critic effect: p = 0.024 becomes 0.094 |
| 10 | Diebold-Mariano compares two single runs, not two configurations | minor | No (not used in the summary) |
| 11 | Gradient-penalty norm has no epsilon | minor | No |
| 12 | Transformer dropout 0.1 and `dim_ff` 2048 are fixed; LSTM dropout is 0 | bias risk | Unknown; Phase 2 tunes them |
| 13 | `REPLICATION_NOTES.md` misses two things the released code does | minor (docs) | No |
| 14 | A critic run started on its own did not reproduce the reported run (**fixed**) | bug | No; all 8 reruns now match the reported values exactly |

Things that were checked and are correct are listed at the end.

---

## 1. How strong is the critic's term compared with the L1 loss?

*Severity: bias risk. `src/cwp/train.py:211-215`, `configs/default.yaml` (`train.adv_weight: 1.0`).*

**Plain version.** The worry was that the critic's score is a big, unbounded number while the
L1 loss is small, so the critic might overwhelm the L1 loss. It does not. What moves the
forecaster is not how big a loss *value* is but how hard it pulls on each prediction (its
gradient). Measured that way, the critic pulls 6-34% as hard as the L1 loss. The problem is
different: the critic's pull has no relation to the direction that reduces the error, because
the critic cannot really tell a predicted value from a real one. It is a source of noise.

**How it was measured.** `python -m cwp.train ... --tag diag --diagnostics` (new, off by
default, tested not to change the trained model) reran 6 reported critic runs at seed 1 and
logged per epoch: the two loss terms, their gradients with respect to the predictions, the
critic loss, the Wasserstein estimate (critic score on real minus on predicted) and the
gradient penalty. All 6 reruns reproduce the reported test MAPE and the whole validation
history exactly, so these diagnostics describe the reported runs. Files:
`results/scratch/*__diag.json` (not in git). Values below are averages over the 10 epochs up
to the selected epoch; scaled units, where the series spans (-1, 1).

| Run (seed 1) | L1 loss | Critic term (value) | Critic gradient / L1 gradient | Cosine between the two gradients | Wasserstein estimate | Gradient penalty |
| --- | --- | --- | --- | --- | --- | --- |
| Google, Transformer + critic + MADGRAD (paper's model) | 0.117 | 3.54 | 0.09 | -0.09 | -0.0001 | 0.0000 |
| Google, Transformer + critic + Adam | 0.107 | 1.69 | 0.27 | -0.03 | 0.012 | 0.0004 |
| Google, LSTM + critic + MADGRAD | 0.109 | -3.03 | 0.08 | -0.08 | -0.0001 | 0.0000 |
| Google, LSTM + critic + Adam | 0.102 | -0.42 | 0.06 | 0.00 | -0.0001 | 0.17 |
| Azure, Transformer + critic + MADGRAD (paper's model) | 0.082 | -1.00 | 0.27 | 0.00 | 0.007 | 0.0003 |
| Azure, LSTM + critic + MADGRAD | 0.072 | 1.23 | 0.34 | 0.07 | 0.013 | 0.0004 |

How to read it:

- **Value.** The critic term is 4 to 27 times the L1 loss in absolute value (first 200
  epochs). This is the "scale mismatch" in the review brief. It has no effect by itself: a
  constant offset in the critic's score changes the value and not the gradient.
- **Gradient size.** The L1 loss pulls every prediction with strength exactly 1 (up or down).
  The critic's pull is its slope along the predicted value, which the gradient penalty keeps
  below about 1; measured, it is 0.06-0.34. Early in training it is larger on Azure (0.37-0.57
  averaged over epochs 1-200).
- **Direction.** A cosine of +1 would mean the critic pushes each prediction the same way the
  L1 loss does, 0 that the two are unrelated. It is between -0.09 and +0.07 at the selected
  epoch. (On Azure it is +0.13 to +0.27 over the first 200 epochs, so there the critic agrees
  weakly with the L1 loss early on, then stops.)
- **Wasserstein estimate.** At most 0.013, against an L1 error of 0.07-0.12, and about zero in
  three of the four Google runs: the critic gives real and predicted sequences the same score.
  Real and predicted inputs share the whole history and differ in one value, so there is
  little for it to see.
- **Gradient penalty.** Near zero except for Google LSTM + critic + Adam (0.17), whose tuned
  learning rate is 0.03 with beta1 = 0: there the critic never settles.

**Why this fits what theory says (my reading, not a measured result).** The forecaster has no
random input: for one history it outputs one number. The Wasserstein distance between a single
number and the true distribution of the next value is smallest at that distribution's median,
which is exactly what the L1 loss already aims for. So in this setup a perfect critic could at
best repeat the L1 loss, and an imperfect one adds noise.

**How sensitive critic training is.** Finding 14 gave an accidental measurement. A rounding-level
difference in the first GPU operation changed critic runs by this much, with everything else
(seed, data, settings) identical, while runs without the critic did not change in any digit:

| Run (seed 1), test MAPE | Reported | Same run with the rounding difference |
| --- | --- | --- |
| Google, Transformer + critic + MADGRAD | 24.19 | 21.61 |
| Google, Transformer + critic + Adam | 22.79 | 22.98 |
| Google, LSTM + critic + MADGRAD | 23.51 | 23.34 |
| Google, LSTM + critic + Adam | 21.99 | 22.89 |
| Azure, Transformer + critic + MADGRAD | 2.042 | 2.044 |
| Azure, LSTM + critic + MADGRAD | 2.008 | 2.572 |

A 2.6-point swing on Google is larger than the critic's main effect. The seeds still work as
independent repeats, so the tests in `summary.md` remain valid, but single critic runs carry
little information and the critic configurations need more seeds than the others.

**What this means for Phase 2.** Lowering `adv_weight` will shrink the noise and should move
the critic configurations toward the no-critic ones, but by this evidence it cannot make the
critic *help*; only a critic that sees something informative can (conditional critic, first
differences, multi-step forecasts). The sweep is still worth running, because "the critic's
best weight is the smallest one" is a clear result for the report.

## 2. Turning the critic on also changes Adam

*Severity: bias risk. `src/cwp/models.py:108-115`, `src/cwp/config.py:184-186`.*

**Plain version.** In the four Adam configurations, "critic on" is two changes at once: the
critic is added, and Adam loses its momentum (beta1 goes from 0.9 to 0). So in those cells we
cannot tell which of the two caused the difference.

**Detail.** `build_optimizer` uses `optim.adam_with_critic` (betas = (0, 0.9), the paper's WGAN
setting) for both the forecaster and the critic whenever the critic is on, and standard Adam
(betas = (0.9, 0.999)) otherwise. The paper's learning rate for that setting (0.0001) is not
used: `resolve_config` overwrites it with the tuned learning rate, which is 0.03 for
`google-10m__lstm-critic-adam`. This follows the paper, and the README says so, but it has two
consequences for the factorial analysis:

- The critic effect in the Adam cells is "critic + beta1 = 0" against "no critic + beta1 = 0.9".
- The optimizer effect in the critic cells is "MADGRAD" against "Adam with beta1 = 0", a
  different Adam than in the no-critic cells. The Azure `critic x optimizer` interaction
  (p = 0.019) may come from this and should not be interpreted yet.

**Impact on the conclusions.** The MADGRAD cells do not have this problem, and there the critic
still hurts the Transformer (from `summary.md`, one-change comparisons):

| Critic effect on test MAPE (points) | LSTM | Transformer |
| --- | --- | --- |
| Google, MADGRAD (clean) | +0.81 (p = 0.15) | +1.96 (p = 0.020) |
| Google, Adam (mixed) | +0.18 (p = 0.82) | +1.02 (p = 0.022) |
| Azure, MADGRAD (clean) | +0.01 (p = 0.76) | +0.12 (p = 0.005) |
| Azure, Adam (mixed) | +0.23 (p = 0.024) | +0.14 (p = 0.001) |

So "the critic does not help" does not depend on the Adam cells. What is not established is
whether the critic hurts the LSTM.

**Proposed fix (additive).** A control experiment: the two Adam + critic configurations with
standard Adam (`optim.adam_with_critic.enabled=false`), 5 seeds, both datasets, under a tag.
It is listed as optional in the compute plan.

## 3. The tuned settings are within tuning noise

*Severity: bias risk. `src/cwp/tune.py:41-47` (one seed per trial, pick the minimum).*

**Plain version.** Each of the 32 tuning trials was run once. Running the same setting with a
different seed moves validation MAPE by more than the gap between the best trial and the next
ones, so the "best" setting is partly a lucky draw.

**Detail.** For each configuration I compared (a) the spread of validation MAPE over the 5
evaluation seeds at the tuned setting (from the 80 result files) with (b) the 32 trial values
in `configs/tuned/*.yaml`.

| Dataset | Seed-to-seed SD of validation MAPE at one setting | Gap between best and second-best trial | Trials within 2 SD of the best |
| --- | --- | --- | --- |
| Google 10 min | 0.15 to 0.77 points | 0.04 to 0.41 points | 1 to 20 of 32 (median 13.5) |
| Azure Functions 5 min | 0.03 to 0.13 points | 0.004 to 0.024 points | 5 to 20 of 32 (median 6) |

A second sign is the "winner's curse": the winning trial's validation MAPE (seed 0) is lower
than the same setting's average over seeds 1-5, because the winner was picked for being low.

| Google 10 min | Winning trial (seed 0) | Same setting, seeds 1-5 | Gap |
| --- | --- | --- | --- |
| No-critic configurations (mean of 4) | 18.62 | 19.04 | 0.42 |
| Critic configurations (mean of 4) | 18.93 | 19.99 | 1.07 |
| `transformer-critic-adam` (worst case) | 18.41 | 20.34 ± 0.39 | 1.93 |

The gap is larger for the critic configurations, because their trials are noisier. On Azure
the gap is small for both (about 0.05 points).

**Impact.** This adds noise to all 16 tuned choices; it does not favor one side by design, but
noisier configurations (the critic ones) are more likely to end up with a setting that is not
really their best. One comparison is free of it: `google-10m` `transformer-nocritic-adam` and
`transformer-critic-adam` happened to get the *same* tuned setting (lr 0.0001, window 48,
`d_model` 128), and there the critic still raises test MAPE by 1.02 points (0 of 5 seeds
improved, p = 0.022). The other 7 critic / no-critic pairs differ in at least one tuned value.

**Proposed fix (for all new tuning in v2).** Keep the 32 one-seed trials, then rerun the best 4
settings with 2 more tuning seeds and pick by the 3-seed mean. This costs about 25% more tuning
time. The original 16 tuned files stay as they are.

## 4. The critic's own settings were never tuned

*Severity: bias risk (already known). `src/cwp/train.py:170`, `configs/tuning.yaml`.*

The critic uses the forecaster's optimizer and learning rate (`opt_d` is built with the same
call). With the LSTM grid that means a critic learning rate of up to 0.03. `adv_weight`,
`n_critic`, `gp_lambda` and `model.critic.hidden` were left at their defaults. Phase 2
addresses this; rule 3 (matched budget for the no-critic side) applies.

## 5. Seed pairing between critic and no-critic runs is weak

*Severity: minor. `src/cwp/train.py:173, 184, 193`.*

**Plain version.** Runs with the same seed are treated as a matched pair in the statistics.
With and without the critic they start from the same weights, but they see the training data
in a different order from the second epoch on, so the "pair" shares less than it could.

**Detail.** The critic's batches are drawn with `torch.randint(..., generator=order)` from the
same generator that produces each epoch's batch order. I checked: the forecaster's initial
weights are identical with and without the critic, the first epoch's order is identical, and
every later epoch's order differs. In addition, 7 of the 8 critic / no-critic pairs have
different tuned settings (history length, layer sizes), so even the initial weights differ.

**Impact.** None on validity. A paired test on weakly paired data is still correct; it only
loses power (4 degrees of freedom). As a check, an unpaired Welch test on the same log-MAPE
values gives the same picture: Google arch p = 0.0008, Google critic p = 0.003, Azure critic
p = 0.0001, Azure optimizer p = 0.085.

**Proposed fix.** Draw critic batches from their own generator in the new (v2) critic code, so
that batch order no longer depends on the critic. Not applied to the default path, because it
would change what a rerun of the 40 reported critic runs produces.

## 6. Epoch selection and early stopping

*Severity: minor. `src/cwp/train.py:231-248`, `configs/default.yaml:47-48`, `configs/tuning.yaml:21`.*

**Plain version.** The test set is never used to pick an epoch, so the reported numbers are
fair. But on Google the models are at their best very early and then overfit, so most of each
1,000-epoch run is spent after the epoch that gets reported.

**Detail.**

- Selected epochs on Google: 32 of the 40 runs pick an epoch in the first 300 (the
  Transformers with Adam all pick 60-120). After that the training L1 keeps falling (for
  example from 0.10 to 0.03 for the Transformers with Adam) while validation MAPE rises from
  18-20% at the best epoch to 20-32% at epoch 1,000. On Azure the selected epochs are spread
  over 80-1,000 and the validation curve is flat with noise.
- With `eval_every: 10` and an optimum near epoch 60-90, there are only 6-9 validation checks
  before the optimum. That is coarse but treats every configuration the same.
- Picking the minimum of 100 noisy validation checks makes the *validation* MAPE optimistic
  (more so for the noisier critic runs). Test MAPE is not affected.
- Tuning `patience: 150`: replaying the 80 validation histories, stopping after 150 epochs
  without improvement would have picked the same epoch in 59 of 80 runs and used 39% of the
  compute; `patience: 300` picks the same epoch in 74 of 80 and uses 58%. So patience 150
  sometimes stops a trial before its best epoch. The validation MAPE lost that way is 0.09
  points on average on Google (1.1 in the worst run) and 0.02 on Azure, which is below the
  tuning noise in finding 3 on average but not in the worst case. With patience 300 the
  worst case is 0.28 points.
- "Last-epoch selection" (what the released code does) would make our Google numbers *worse*
  (validation 20-32% at epoch 1,000 against 18-20% at the best epoch), so it cannot explain
  the paper's lower MAPE at our learning rates.

**Proposed change.** None to the 80 runs. For Phase 2: the heavy overfitting says that
regularization (dropout, weight decay, a smaller `dim_ff`) is a sensible set of knobs for the
no-critic side's matched tuning budget. New runs will also record test metrics at the last
epoch (reporting only), so "best-validation vs last epoch" costs no extra training.

## 7. Late divergence and `Infinity` in result files

*Severity: minor. `src/cwp/train.py:233-234`.*

3 of the 5 `azure_functions-5m__lstm-critic-madgrad` runs diverge after their best epoch
(training L1 above 1 at epoch 1,000, where a healthy run is near 0.07; in one of them
validation MAPE becomes infinite). No other configuration does this. Best-validation selection
returns the good epoch, so the reported metrics are fine, but (a) this configuration is
unstable and the report should say so, and (b) that one history stores `Infinity`, which Python
reads but strict JSON parsers (`jq`, JavaScript) reject. In tuning, at least one trial of each
LSTM + critic + MADGRAD configuration never produced a usable model (best validation MAPE
above 1e8).

**Proposed fix.** Write non-finite values as `null` in new result files and keep reading the
old ones. Not done yet (it touches the result format); it will go in with the Phase 1 changes.

## 8. Why the Wilcoxon column is blank

*Severity: minor. `src/cwp/evaluate.py:96`.*

The test is only run with 6 or more seeds (`len(delta) >= 6`), on purpose: with 5 pairs the
smallest possible two-sided Wilcoxon p-value is 2/32 = 0.0625, so it can never be below 0.05.
Nothing is broken. The column fills in once Phase 4 adds seeds. The v2 summary will print this
reason under the table.

## 9. No correction for multiple tests in the factorial table

*Severity: minor. `src/cwp/evaluate.py:110-132`.*

**Plain version.** The more p-values a table has, the more likely one of them is small by
chance. The one-change table corrects for that (Holm); the factorial table does not.

**Detail.** With Holm's correction across the 6 main effects (3 per dataset):

| Main effect | p (as reported) | p (Holm, 6 tests) |
| --- | --- | --- |
| Azure: critic | 0.0020 | 0.012 |
| Google: Transformer | 0.0021 | 0.012 |
| Google: critic | 0.024 | 0.094 |
| Azure: MADGRAD | 0.078 | 0.24 |
| Google: MADGRAD, Azure: Transformer | 0.65, 0.85 | 1.0 |

So "the critic hurts on Google" is in the same direction in all 5 seeds but is not significant
at 0.05 after correction. The main-effect and interaction arithmetic itself is correct: I
recomputed every main effect independently and got the same effects and p-values.

**Proposed fix.** Add a Holm-adjusted column in the v2 summary, and add seeds in Phase 4 for
the borderline effects (Google critic, Azure MADGRAD).

## 10. Diebold-Mariano test

*Severity: minor. `src/cwp/metrics.py:73-85`, `src/cwp/evaluate.py:203`.*

The statistic and the Harvey correction are right for one-step forecasts, and the loss
differences are almost uncorrelated in time (lag-1 autocorrelation 0.04-0.06 on two pairs I
checked), so using only the lag-0 variance is fine. The limitation is what it compares: two
trained models (one seed each), not two configurations, so it says nothing about seed noise.
It is not used in `summary.md`. **Proposed:** in Phase 4, run it on seed-averaged predictions
or report it per seed pair.

## 11. Gradient-penalty norm has no epsilon

*Severity: minor. `src/cwp/train.py:98`.*

`grad.norm(2, dim=1)` has an undefined derivative when a sample's gradient is exactly zero.
With LeakyReLU this is very unlikely and no reported run shows it. **Proposed:** use
`sqrt(sum(g^2) + 1e-12)` in the v2 critic code (not in the default path, to keep reruns of the
reported runs unchanged).

## 12. Regularization settings are fixed and differ between the architectures

*Severity: bias risk. `configs/default.yaml:13, 19-20`, `src/cwp/models.py:20`.*

The Transformer trains with dropout 0.1 and a 2,048-wide feed-forward layer (for `d_model`
16-128); the LSTM has dropout 0, and LSTM dropout does nothing with one layer anyway. Neither
was tuned. Given the overfitting in finding 6, this could matter for the Transformer / LSTM
comparison in either direction. Phase 2's matched no-critic tuning budget covers these knobs.

## 13. Two things the released code does that `REPLICATION_NOTES.md` does not list

*Severity: minor (documentation). Released code at `d9519dc`, `main.py`.*

- **One scaler object is refitted three times** (`main.py:495-513`): on the training part,
  then the test part, then the validation part. Every later `inverse_transform`, including the
  one for test predictions (`main.py:431-441`), therefore uses the *validation* part's minimum
  and maximum. Test MAPE is computed on values that are not job counts. This goes beyond
  "fitted separately on each part".
- **`d_model` is a command-line argument** (`main.py:40, 314-316`); the value computed from
  the history length is never used. The notes say it is set from the history length.

I measured what the scaler handling does to MAPE on our series, with no training (same affine
map applied to the true values and to the predictions of the reported runs):

| Series | Naive forecast, true units | Naive, released-code units | Our models, true units | Our models, released-code units | Paper Table 3 (LSTM / WGAN-gp) |
| --- | --- | --- | --- | --- | --- |
| Google 10 min | 27.6 | 26.8 | 20.7-22.7 | 20.1-22.0 | 11.49 / 10.58 |
| Azure Functions 5 min | 3.45 | 7.22 | 2.01-2.13 | 4.2-4.5 | 1.63 / 3.05 |

So this quirk moves Azure MAPE a lot (the validation part has a much larger range than the
test part) and Google MAPE hardly at all. It does **not** explain the paper's Google numbers
on our series. This is a head start on Phase 1's hypothesis test; the trained version of that
test is still needed.

Also relevant to Phase 1: **no interval is empty** in any of the four processed series (the
minimum is 29 jobs for Google 10 min), so "dropping empty Google intervals" changes nothing at
10 or 30 minutes and needs no runs.

## 14. A critic run started on its own did not reproduce the reported run (fixed)

*Severity: bug (reproducibility only). `src/cwp/train.py:43-55` (`warm_up`), called at the top of `fit`.*

**Plain version.** Running a reported critic run again with the README's command gave a
different result, even with the same seed. The reported numbers were not wrong; the same run
simply came out differently depending on whether another run had been trained earlier in the
same Python process.

**Detail.** The first GPU computation in a new process rounds slightly differently from later
ones (the GPU libraries initialize on first use). Runs without the critic are not affected:
two of them, rerun standalone, match the reported runs in every digit. Critic training
amplifies the difference (table in finding 1). The reported runs were started one after
another in one process by `cwp.launch`, so a standalone `python -m cwp.train` of a critic run
was the odd one out. I confirmed the cause by experiment: the first critic run in a fresh
process differed, the second and third were identical to each other and to the reported run,
and doing any small GPU forward and backward pass first removed the difference.

**Fix (applied, commits `a17200d` and the one after it).** `fit` now runs one tiny forward
and backward pass on the GPU before it sets the seeds. It draws no random numbers (tested).
After the fix, 6 critic runs and 2 no-critic runs at seed 1, each started in its own process,
reproduce the reported test MAPE and the full validation history exactly.

**Impact on the reported results.** None. No result file changed. Remaining limits: exact
reproduction is only checked on this machine (RTX 5070 Ti, PyTorch 2.11.0); other GPUs or
library versions can round differently, and finding 1 shows that critic runs would then
differ visibly.

---

## Checked and correct

| Item | Where | What was checked |
| --- | --- | --- |
| Critic / forecaster update order | `train.py:189-223` | `n_critic` critic steps, then one forecaster step, per batch |
| `requires_grad` toggling | `train.py:190-191, 208-209` | Critic weights are frozen during the forecaster step; gradients still flow through the critic to the prediction |
| Sign of the adversarial term (paper Eq. 4) | `train.py:201, 215` | Critic minimizes `D(fake) - D(real) + lambda * GP`; forecaster minimizes `L1 - adv_weight * D(fake)` |
| Gradient penalty | `train.py:93-98` | One `alpha` per sample, broadcast over `[batch, window + 1]`; per-sample gradient norm. New test: equals `(|w| - 1)^2` for a linear critic |
| Fake sequences are detached for the critic step | `train.py:195-198` | `torch.no_grad()` around the forecaster |
| Scaler | `train.py:66` | Fitted on the training part only |
| Windows, no leakage | `data.py:119-131` | History never includes the target; validation / test windows reach back into earlier parts |
| Same test points for every model | all 80 `results/predictions/*.csv` | `t` runs from `val_end` to `n - 1` in every file, and `y_true` equals the processed series |
| Metrics on unscaled values | `train.py:85-90`, `metrics.py` | Recomputing all metrics from the prediction CSVs reproduces every value in the 80 result files |
| MAPE and zero actuals | `metrics.py:11-15` | No zeros in any series or split, so nothing is ever skipped |
| MASE | `metrics.py:32-39` | Scale is the naive error on the unscaled training part |
| Provisioning metrics | `metrics.py:42-49` | Computed on unscaled values, same test points; no prediction is negative in any run |
| Paired t-test, Holm | `evaluate.py:95-105`, `metrics.py:88-96` | Correct; Holm is applied over the 12 one-change comparisons of each dataset |
| Factorial effects | `evaluate.py:110-132` | Contrast signs and the divisor are right; recomputed independently |
| Environment of the 80 runs | result files | One machine, one commit (`38a9461`, clean), PyTorch 2.11.0, all with tuned settings |
