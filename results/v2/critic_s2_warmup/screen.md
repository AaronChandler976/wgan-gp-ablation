# Screen: critic_s2_warmup

32 of 32 trials finished. Validation MAPE (%) only; the test part is not used. Tuning seeds [0, 100], settings while screening: {'train.patience': 150}. Changes are against the variant `w0.01_warm50` in the same cell.

Stage 2 of the critic screen: the forecaster first trains on the L1 loss alone for 50 epochs, then the critic term is switched on, either at once or rising linearly over 100 epochs. At the stage-1 weight carried forward (0.01) and at the reported weight (1). Compare with `w0.01` and `w1` of critic_s1_weight.

## Variants, best first (average over cells)

| variant | cells | worst_cell_pct | best_cell_pct | mean_change_pct |
| --- | --- | --- | --- | --- |
| w0.01_warm50_ramp100 | 4 | 0.397 | -1.376 | -0.485 |
| w0.01_warm50 | 4 | 0.0 | 0.0 | 0.0 |
| w1_warm50_ramp100 | 4 | 3.626 | 0.582 | 2.418 |
| w1_warm50 | 4 | 4.439 | 0.641 | 2.46 |

## Every cell

| cell | variant | seeds | val_mape | seed_gap | seconds | vs_reference_pct |
| --- | --- | --- | --- | --- | --- | --- |
| google-10m__transformer-critic-madgrad | w0.01_warm50 | 2 | 18.36 | 0.006 | 327.5 | 0.0 |
| google-10m__transformer-critic-madgrad | w0.01_warm50_ramp100 | 2 | 18.285 | 0.279 | 197.5 | -0.411 |
| google-10m__transformer-critic-madgrad | w1_warm50 | 2 | 18.478 | 0.211 | 271.75 | 0.641 |
| google-10m__transformer-critic-madgrad | w1_warm50_ramp100 | 2 | 18.467 | 0.29 | 189.85 | 0.582 |
| google-10m__lstm-critic-madgrad | w0.01_warm50 | 2 | 19.609 | 0.313 | 342.45 | 0.0 |
| google-10m__lstm-critic-madgrad | w0.01_warm50_ramp100 | 2 | 19.339 | 0.247 | 736.55 | -1.376 |
| google-10m__lstm-critic-madgrad | w1_warm50 | 2 | 20.427 | 0.407 | 257.85 | 4.172 |
| google-10m__lstm-critic-madgrad | w1_warm50_ramp100 | 2 | 20.031 | 0.142 | 375.45 | 2.155 |
| azure_functions-5m__transformer-critic-madgrad | w0.01_warm50 | 2 | 2.423 | 0.149 | 664.75 | 0.0 |
| azure_functions-5m__transformer-critic-madgrad | w0.01_warm50_ramp100 | 2 | 2.433 | 0.087 | 604.25 | 0.397 |
| azure_functions-5m__transformer-critic-madgrad | w1_warm50 | 2 | 2.531 | 0.173 | 359.6 | 4.439 |
| azure_functions-5m__transformer-critic-madgrad | w1_warm50_ramp100 | 2 | 2.504 | 0.216 | 429.85 | 3.338 |
| azure_functions-5m__lstm-critic-madgrad | w0.01_warm50 | 2 | 2.479 | 0.016 | 387.1 | 0.0 |
| azure_functions-5m__lstm-critic-madgrad | w0.01_warm50_ramp100 | 2 | 2.466 | 0.016 | 431.9 | -0.543 |
| azure_functions-5m__lstm-critic-madgrad | w1_warm50 | 2 | 2.496 | 0.047 | 294.15 | 0.655 |
| azure_functions-5m__lstm-critic-madgrad | w1_warm50_ramp100 | 2 | 2.569 | 0.054 | 136.45 | 3.626 |
