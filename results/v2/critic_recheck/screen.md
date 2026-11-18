# Screen: critic_recheck

64 of 64 trials finished. Validation MAPE (%) only; the test part is not used. Tuning seeds [200, 300], settings while screening: {'train.patience': 150}. Changes are against the variant `w1` in the same cell.

Confirmation of the critic screen on tuning seeds it has not seen (200, 300), and on the Adam cells, which the screen did not cover. Nothing is selected here.

## Variants, best first (average over cells)

| variant | cells | worst_cell_pct | best_cell_pct | mean_change_pct |
| --- | --- | --- | --- | --- |
| w0.001 | 8 | 1.289 | -4.347 | -2.396 |
| w0.01 | 8 | 1.195 | -4.551 | -2.357 |
| w0.01_diffs | 8 | 2.408 | -4.983 | -1.414 |
| w1 | 8 | 0.0 | 0.0 | 0.0 |

## Every cell

| cell | variant | seeds | val_mape | seed_gap | seconds | vs_reference_pct |
| --- | --- | --- | --- | --- | --- | --- |
| google-10m__transformer-critic-madgrad | w1 | 2 | 19.269 | 0.335 | 301.2 | 0.0 |
| google-10m__transformer-critic-madgrad | w0.01 | 2 | 18.392 | 0.071 | 255.05 | -4.551 |
| google-10m__transformer-critic-madgrad | w0.001 | 2 | 18.431 | 0.004 | 267.95 | -4.347 |
| google-10m__transformer-critic-madgrad | w0.01_diffs | 2 | 18.309 | 0.019 | 212.4 | -4.983 |
| google-10m__transformer-critic-adam | w1 | 2 | 19.278 | 1.686 | 204.65 | 0.0 |
| google-10m__transformer-critic-adam | w0.01 | 2 | 18.656 | 0.124 | 239.95 | -3.231 |
| google-10m__transformer-critic-adam | w0.001 | 2 | 18.547 | 0.743 | 192.15 | -3.793 |
| google-10m__transformer-critic-adam | w0.01_diffs | 2 | 18.734 | 0.645 | 232.1 | -2.823 |
| google-10m__lstm-critic-madgrad | w1 | 2 | 20.308 | 0.143 | 140.15 | 0.0 |
| google-10m__lstm-critic-madgrad | w0.01 | 2 | 19.682 | 0.938 | 213.05 | -3.082 |
| google-10m__lstm-critic-madgrad | w0.001 | 2 | 19.764 | 0.845 | 188.85 | -2.68 |
| google-10m__lstm-critic-madgrad | w0.01_diffs | 2 | 19.707 | 0.772 | 225.6 | -2.96 |
| google-10m__lstm-critic-adam | w1 | 2 | 20.217 | 0.338 | 81.8 | 0.0 |
| google-10m__lstm-critic-adam | w0.01 | 2 | 20.105 | 0.688 | 88.0 | -0.555 |
| google-10m__lstm-critic-adam | w0.001 | 2 | 20.478 | 0.447 | 74.35 | 1.289 |
| google-10m__lstm-critic-adam | w0.01_diffs | 2 | 20.309 | 0.027 | 82.55 | 0.453 |
| azure_functions-5m__transformer-critic-madgrad | w1 | 2 | 2.495 | 0.007 | 286.0 | 0.0 |
| azure_functions-5m__transformer-critic-madgrad | w0.01 | 2 | 2.468 | 0.127 | 280.75 | -1.1 |
| azure_functions-5m__transformer-critic-madgrad | w0.001 | 2 | 2.428 | 0.106 | 407.55 | -2.681 |
| azure_functions-5m__transformer-critic-madgrad | w0.01_diffs | 2 | 2.555 | 0.295 | 241.65 | 2.408 |
| azure_functions-5m__transformer-critic-adam | w1 | 2 | 2.52 | 0.013 | 524.9 | 0.0 |
| azure_functions-5m__transformer-critic-adam | w0.01 | 2 | 2.437 | 0.031 | 707.8 | -3.319 |
| azure_functions-5m__transformer-critic-adam | w0.001 | 2 | 2.436 | 0.036 | 653.9 | -3.344 |
| azure_functions-5m__transformer-critic-adam | w0.01_diffs | 2 | 2.442 | 0.025 | 610.4 | -3.118 |
| azure_functions-5m__lstm-critic-madgrad | w1 | 2 | 2.547 | 0.045 | 185.1 | 0.0 |
| azure_functions-5m__lstm-critic-madgrad | w0.01 | 2 | 2.443 | 0.065 | 251.2 | -4.071 |
| azure_functions-5m__lstm-critic-madgrad | w0.001 | 2 | 2.446 | 0.02 | 179.75 | -3.964 |
| azure_functions-5m__lstm-critic-madgrad | w0.01_diffs | 2 | 2.517 | 0.233 | 155.05 | -1.186 |
| azure_functions-5m__lstm-critic-adam | w1 | 2 | 2.473 | 0.033 | 91.7 | 0.0 |
| azure_functions-5m__lstm-critic-adam | w0.01 | 2 | 2.503 | 0.131 | 125.15 | 1.195 |
| azure_functions-5m__lstm-critic-adam | w0.001 | 2 | 2.486 | 0.117 | 125.15 | 0.514 |
| azure_functions-5m__lstm-critic-adam | w0.01_diffs | 2 | 2.501 | 0.157 | 128.85 | 1.127 |
