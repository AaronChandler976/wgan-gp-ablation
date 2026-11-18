# Screen: critic_s5_input

32 of 32 trials finished. Validation MAPE (%) only; the test part is not used. Tuning seeds [0, 100], settings while screening: {'train.patience': 150}. Changes are against the variant `w0.01_conditional` in the same cell.

Stage 5 of the critic screen: a critic that sees more. `conditional` reads the history and the forecast through separate layers; `diffs` scores first differences. Each at weight 0.01 and at the reported weight 1. Compare with `w0.01` and `w1` of critic_s1_weight.

## Variants, best first (average over cells)

| variant | cells | worst_cell_pct | best_cell_pct | mean_change_pct |
| --- | --- | --- | --- | --- |
| w0.01_diffs | 4 | 0.42 | -2.336 | -0.502 |
| w0.01_conditional | 4 | 0.0 | 0.0 | 0.0 |
| w1_conditional | 4 | 15.516 | -0.167 | 4.964 |
| w1_diffs | 4 | 11.848 | 1.637 | 7.166 |

## Every cell

| cell | variant | seeds | val_mape | seed_gap | seconds | vs_reference_pct |
| --- | --- | --- | --- | --- | --- | --- |
| google-10m__transformer-critic-madgrad | w0.01_conditional | 2 | 18.527 | 0.496 | 146.45 | 0.0 |
| google-10m__transformer-critic-madgrad | w0.01_diffs | 2 | 18.523 | 0.037 | 111.5 | -0.025 |
| google-10m__transformer-critic-madgrad | w1_conditional | 2 | 19.059 | 0.297 | 122.65 | 2.869 |
| google-10m__transformer-critic-madgrad | w1_diffs | 2 | 20.722 | 3.064 | 114.15 | 11.848 |
| google-10m__lstm-critic-madgrad | w0.01_conditional | 2 | 19.841 | 0.065 | 150.9 | 0.0 |
| google-10m__lstm-critic-madgrad | w0.01_diffs | 2 | 19.832 | 0.093 | 121.9 | -0.045 |
| google-10m__lstm-critic-madgrad | w1_conditional | 2 | 20.301 | 0.084 | 80.65 | 2.32 |
| google-10m__lstm-critic-madgrad | w1_diffs | 2 | 20.166 | 0.232 | 66.95 | 1.637 |
| azure_functions-5m__transformer-critic-madgrad | w0.01_conditional | 2 | 2.487 | 0.2 | 155.6 | 0.0 |
| azure_functions-5m__transformer-critic-madgrad | w0.01_diffs | 2 | 2.429 | 0.127 | 159.6 | -2.336 |
| azure_functions-5m__transformer-critic-madgrad | w1_conditional | 2 | 2.483 | 0.113 | 152.1 | -0.167 |
| azure_functions-5m__transformer-critic-madgrad | w1_diffs | 2 | 2.581 | 0.237 | 107.55 | 3.768 |
| azure_functions-5m__lstm-critic-madgrad | w0.01_conditional | 2 | 2.593 | 0.503 | 136.55 | 0.0 |
| azure_functions-5m__lstm-critic-madgrad | w0.01_diffs | 2 | 2.604 | 0.095 | 91.9 | 0.42 |
| azure_functions-5m__lstm-critic-madgrad | w1_conditional | 2 | 2.995 | 1.069 | 115.05 | 15.516 |
| azure_functions-5m__lstm-critic-madgrad | w1_diffs | 2 | 2.899 | 0.255 | 58.8 | 11.809 |
