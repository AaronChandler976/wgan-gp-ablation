# Replication notes

What the paper says, what its released code does, and what this repo does. Where the paper and
the released code disagree, this repo follows the **paper**.

Sources: Arbat et al., AAAI 2022 (the paper); https://github.com/shivaniarbat/wgan-gp-transformer
at commit `d9519dc` (the released code); Jayakumar et al., IPDPS 2020 (LoadDynamics).

## WGAN-gp Transformer

| Item | Paper | Released code | This repo |
| --- | --- | --- | --- |
| Generator | Transformer, 1 encoder + 1 decoder layer | Same | Same |
| Decoder input | Last observed value | Last observed value | Same |
| Input embedding | Not stated | The scalar is copied into all `d_model` features | Learned linear layer (`input_embed: broadcast` copies the scalar instead) |
| Positional encoding | Sinusoidal, over time steps | Added by position in the **batch**, not in time (`pe[:x.size(1)]` on a `[window, batch, d]` tensor) | Sinusoidal, over time steps |
| Feed-forward size | Not stated | PyTorch default, 2048 | 2048 |
| Critic | 3 fully connected layers, LeakyReLU | Tanh after every layer including the output; applied to each value separately (`Linear(1, d)`) | 3 fully connected layers, LeakyReLU(0.2), one unbounded score for the whole sequence |
| Critic updates per generator update | 5 | 1 | 5 |
| Gradient penalty weight | 10 | 0.1 | 10 |
| Generator loss | L1 minus the critic score (Eq. 4) | L1 **plus** the critic score | L1 minus the critic score |
| Learning rate | 0.001 | 0.01 | 0.001 |
| MADGRAD | momentum 0.9, weight decay 0, eps 1e-6 | Library defaults (the same values) | Same |
| Adam (for the Adam comparison) | beta1 = 0, beta2 = 0.9, lr = 0.0001 | Not in the repo | Same, when the critic is on (`optim.adam_with_critic`) |
| Epochs | 1,000 | 1,000 | 1,000 |
| Hyperparameter search | Grid over history length, batch size, `d_model` and heads (Table 2) | None: one setting per run; `d_model`, heads, dropout, history length and batch size come from the command line (a `d_model` computed from the history length is never used) | Learning rate (0.00001-0.0003), history length (48, 96), `d_model` (16-128), 8 heads: 32 trials of up to 1,000 epochs with early stopping, the same number every LSTM configuration gets |
| Split | First 60% / next 20% / last 20% | Same | Same |
| Scaling | Not stated | Min-max to (-1, 1), fitted **separately on train, validation and test** | Min-max to (-1, 1), fitted on the training part only |
| Un-scaling for the metrics | Not stated | One scaler object is refitted on train, then test, then validation, so **every part is un-scaled with the validation part's minimum and maximum**; test MAPE is not computed in jobs per interval | Each value is un-scaled with the scaler that scaled it |
| Critic's batch | Not stated | The generator's own batch, one forward pass for both | Separate random batches for the critic (`train.critic_batch`) |
| Windows | Sliding, stride 1 | Built inside each part, so the first `n` points of validation and test are never predicted | Validation and test windows may use earlier points as history, so every point is predicted |
| Batch order | Not stated | Time order, no shuffling | Shuffled every epoch with a seeded order |
| Which epoch is reported | Not stated | The last one | The one with the best validation MAPE (`train.select`) |
| Random seed | Not stated | Fixed (`torch.manual_seed(1)`) | The seeds in `configs/plan.yaml` |

Every row where the released code differs is a config key that defaults to this repo's choice
(`data.scaler`, `data.windows`, `data.drop_empty`, `model.transformer.input_embed`,
`model.transformer.pos_encoding`, `model.critic.kind`, `train.n_critic`, `train.gp_lambda`,
`train.adv_sign`, `train.critic_batch`, `train.shuffle`, `train.select`, and the learning rate).
`configs/profiles/released_code.yaml` switches them all on (`--profile released_code`); the
experiments that use it are in `docs/EXPERIMENTS_V2.md`. Not reproduced: the released critic is
as wide as `d_model`, and its gradient penalty draws one mixing weight per time position.

Two of these change results on purpose and are worth saying in the report:

- **Scaling.** Fitting the scaler on the test part uses the test data's own minimum and maximum.
  Fitting on the training part only is the standard, leak-free choice.
- **Same test points for every model.** Because history may reach back into the previous part,
  a model with a window of 24 and one with a window of 96 are scored on exactly the same test
  points. That is what makes paired comparisons and the Diebold-Mariano test valid.

## LoadDynamics LSTM

| Item | LoadDynamics paper | This repo |
| --- | --- | --- |
| Model | Stacked LSTM plus one fully connected output layer | Same (`--arch lstm`) |
| Loss | Mean squared error | L1 by default for **every** configuration, so the loss is not a fourth difference. `train.loss=mse` gives the LoadDynamics loss. |
| Optimizer | Adam | `--optimizer adam` (or `madgrad` for the ablation) |
| Tuned settings | History length, cell size, layers, batch size | Same knobs: `data.window`, `model.lstm.hidden`, `model.lstm.layers`, `train.batch_size` |
| Search | Bayesian optimization, 100 iterations. The WGAN-gp paper used the "brute force" version, with LoadDynamics' published settings where available and a grid search otherwise. | A grid (`configs/tuning.yaml`): learning rate (0.001-0.03), history length (48, 96), cell size (16, 64) and layers (1, 2). 32 trials of up to 1,000 epochs with early stopping, the same number every Transformer configuration gets |
| Learning rate, epochs, scaling | Not stated | Same as the other configurations |
| Framework | TensorFlow 1 | PyTorch |

Settings LoadDynamics selected (its Table IV, range across intervals), for reference:

| Workload | History length | Cell size | Layers | Batch size |
| --- | --- | --- | --- | --- |
| Google | 5-472 | 8-99 | 1-3 | 18-371 |
| Azure | 68-79 | 8-47 | 1-4 | 18-700 |

## Data

| Dataset | Paper's code | This repo |
| --- | --- | --- |
| Google Cluster 2011 | Rows of `job_events` with event type 0 (SUBMIT) and time >= 600 s, counted per interval. Intervals with no events disappear. | Same events. Intervals with no events are kept as 0. At 10 and 30 minutes no interval is empty, so the two agree (`data.drop_empty` exists for completeness). |
| Azure Functions 2019 | Not in the released code | Invocations per minute summed over all functions, then summed into 5- and 10-minute intervals |

The paper does not say how it aggregated Azure Functions, so our MAPE values may not match its
Table 3. Compare the direction and rough size of the differences, not exact numbers.
