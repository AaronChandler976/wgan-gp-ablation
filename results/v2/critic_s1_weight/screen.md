# Screen: critic_s1_weight

64 of 64 trials finished. Validation MAPE (%) only; the test part is not used. Tuning seeds [0, 100], settings while screening: {'train.patience': 150}. Changes are against the variant `w1` in the same cell.

Stage 1 of the critic screen: the weight of the critic term in the generator loss, on a log scale. `w1` is the reported setting. `w0` trains the critic but gives it no influence on the forecaster, so it shows the same hyperparameters without the critic's push.

## Variants, best first (average over cells)

| variant | cells | worst_cell_pct | best_cell_pct | mean_change_pct |
| --- | --- | --- | --- | --- |
| w0.001 | 4 | -0.786 | -17.044 | -5.57 |
| w0.01 | 4 | -0.189 | -16.761 | -5.251 |
| w0.003 | 4 | 0.304 | -13.977 | -4.295 |
| w0.03 | 4 | 1.611 | -13.581 | -4.158 |
| w0.1 | 4 | 0.115 | -13.596 | -4.036 |
| w0 | 4 | 0.023 | -10.679 | -3.136 |
| w0.3 | 4 | 1.533 | -12.495 | -2.996 |
| w1 | 4 | 0.0 | 0.0 | 0.0 |

## Every cell

| cell | variant | seeds | val_mape | seed_gap | seconds | vs_reference_pct |
| --- | --- | --- | --- | --- | --- | --- |
| google-10m__transformer-critic-madgrad | w1 | 2 | 18.468 | 0.538 | 518.45 | 0.0 |
| google-10m__transformer-critic-madgrad | w0.3 | 2 | 18.751 | 0.176 | 410.7 | 1.533 |
| google-10m__transformer-critic-madgrad | w0.1 | 2 | 18.489 | 0.597 | 376.0 | 0.115 |
| google-10m__transformer-critic-madgrad | w0.03 | 2 | 18.766 | 0.035 | 304.9 | 1.611 |
| google-10m__transformer-critic-madgrad | w0.01 | 2 | 18.433 | 0.315 | 365.55 | -0.189 |
| google-10m__transformer-critic-madgrad | w0.003 | 2 | 18.524 | 0.367 | 194.0 | 0.304 |
| google-10m__transformer-critic-madgrad | w0.001 | 2 | 18.028 | 0.004 | 257.3 | -2.382 |
| google-10m__transformer-critic-madgrad | w0 | 2 | 18.472 | 0.017 | 203.2 | 0.023 |
| google-10m__lstm-critic-madgrad | w1 | 2 | 19.986 | 1.345 | 216.0 | 0.0 |
| google-10m__lstm-critic-madgrad | w0.3 | 2 | 19.916 | 0.66 | 264.25 | -0.352 |
| google-10m__lstm-critic-madgrad | w0.1 | 2 | 19.9 | 0.813 | 187.15 | -0.432 |
| google-10m__lstm-critic-madgrad | w0.03 | 2 | 19.324 | 0.16 | 299.45 | -3.315 |
| google-10m__lstm-critic-madgrad | w0.01 | 2 | 19.848 | 0.063 | 216.8 | -0.694 |
| google-10m__lstm-critic-madgrad | w0.003 | 2 | 19.846 | 0.106 | 254.1 | -0.702 |
| google-10m__lstm-critic-madgrad | w0.001 | 2 | 19.829 | 0.046 | 243.45 | -0.786 |
| google-10m__lstm-critic-madgrad | w0 | 2 | 19.878 | 0.079 | 360.25 | -0.542 |
| azure_functions-5m__transformer-critic-madgrad | w1 | 2 | 2.478 | 0.105 | 343.4 | 0.0 |
| azure_functions-5m__transformer-critic-madgrad | w0.3 | 2 | 2.478 | 0.132 | 257.7 | 0.011 |
| azure_functions-5m__transformer-critic-madgrad | w0.1 | 2 | 2.44 | 0.127 | 455.55 | -1.535 |
| azure_functions-5m__transformer-critic-madgrad | w0.03 | 2 | 2.463 | 0.12 | 466.85 | -0.615 |
| azure_functions-5m__transformer-critic-madgrad | w0.01 | 2 | 2.42 | 0.114 | 467.4 | -2.316 |
| azure_functions-5m__transformer-critic-madgrad | w0.003 | 2 | 2.426 | 0.193 | 308.4 | -2.081 |
| azure_functions-5m__transformer-critic-madgrad | w0.001 | 2 | 2.452 | 0.19 | 410.75 | -1.031 |
| azure_functions-5m__transformer-critic-madgrad | w0 | 2 | 2.455 | 0.212 | 217.1 | -0.926 |
| azure_functions-5m__lstm-critic-madgrad | w1 | 2 | 2.912 | 0.875 | 221.7 | 0.0 |
| azure_functions-5m__lstm-critic-madgrad | w0.3 | 2 | 2.548 | 0.228 | 187.7 | -12.495 |
| azure_functions-5m__lstm-critic-madgrad | w0.1 | 2 | 2.516 | 0.021 | 242.05 | -13.596 |
| azure_functions-5m__lstm-critic-madgrad | w0.03 | 2 | 2.517 | 0.321 | 263.85 | -13.581 |
| azure_functions-5m__lstm-critic-madgrad | w0.01 | 2 | 2.424 | 0.041 | 223.2 | -16.761 |
| azure_functions-5m__lstm-critic-madgrad | w0.003 | 2 | 2.505 | 0.006 | 232.7 | -13.977 |
| azure_functions-5m__lstm-critic-madgrad | w0.001 | 2 | 2.416 | 0.06 | 177.5 | -17.044 |
| azure_functions-5m__lstm-critic-madgrad | w0 | 2 | 2.601 | 0.005 | 118.4 | -10.679 |
