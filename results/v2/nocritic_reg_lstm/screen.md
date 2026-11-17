# Screen: nocritic_reg_lstm

136 of 136 trials finished. Validation MAPE (%) only; the test part is not used. Tuning seeds [0, 100], settings while screening: {'train.patience': 150}. Changes are against the variant `base` in the same cell.

Matched tuning budget for the no-critic LSTM: 34 variants (the reported setting plus 33 of the 35 other combinations of output dropout, weight decay and gradient clipping, drawn with a fixed seed).

## Variants, best first (average over cells)

| variant | cells | worst_cell_pct | best_cell_pct | mean_change_pct |
| --- | --- | --- | --- | --- |
| out_dropout0.0_weight_decay0.0_grad_clip1.0 | 2 | -0.655 | -1.71 | -1.184 |
| out_dropout0.3_weight_decay0.0_grad_clip1.0 | 2 | -0.95 | -1.394 | -1.172 |
| out_dropout0.1_weight_decay0.0_grad_clip1.0 | 2 | -0.179 | -1.397 | -0.79 |
| base | 2 | 0.0 | 0.0 | 0.0 |
| out_dropout0.1_weight_decay0.0001_grad_clip0.0 | 2 | 2.768 | -2.37 | 0.166 |
| out_dropout0.1_weight_decay0.0_grad_clip0.1 | 2 | 2.399 | -1.248 | 0.559 |
| out_dropout0.3_weight_decay0.0_grad_clip0.1 | 2 | 3.14 | -1.849 | 0.614 |
| out_dropout0.0_weight_decay0.0001_grad_clip0.0 | 2 | 2.775 | -1.109 | 0.814 |
| out_dropout0.0_weight_decay0.0001_grad_clip1.0 | 2 | 2.368 | 0.226 | 1.291 |
| out_dropout0.2_weight_decay0.0001_grad_clip0.0 | 2 | 3.176 | -0.183 | 1.483 |
| out_dropout0.2_weight_decay0.0001_grad_clip1.0 | 2 | 3.553 | -0.462 | 1.525 |
| out_dropout0.3_weight_decay0.0001_grad_clip0.0 | 2 | 3.452 | 1.94 | 2.693 |
| out_dropout0.2_weight_decay0.0_grad_clip0.1 | 2 | 7.777 | -2.109 | 2.715 |
| out_dropout0.1_weight_decay0.0_grad_clip0.0 | 2 | 5.629 | -0.082 | 2.734 |
| out_dropout0.0_weight_decay0.0001_grad_clip0.1 | 2 | 4.255 | 2.979 | 3.615 |
| out_dropout0.1_weight_decay0.0001_grad_clip1.0 | 2 | 6.395 | 1.112 | 3.72 |
| out_dropout0.3_weight_decay0.0_grad_clip0.0 | 2 | 8.785 | -1.068 | 3.741 |
| out_dropout0.3_weight_decay0.0001_grad_clip1.0 | 2 | 6.404 | 3.2 | 4.79 |
| out_dropout0.1_weight_decay0.0001_grad_clip0.1 | 2 | 8.604 | 3.156 | 5.845 |
| out_dropout0.3_weight_decay0.0001_grad_clip0.1 | 2 | 11.456 | 3.353 | 7.328 |
| out_dropout0.2_weight_decay0.0_grad_clip0.0 | 2 | 22.269 | -1.229 | 9.894 |
| out_dropout0.2_weight_decay0.0001_grad_clip0.1 | 2 | 16.652 | 3.53 | 9.895 |
| out_dropout0.0_weight_decay0.001_grad_clip1.0 | 2 | 44.746 | 3.885 | 22.626 |
| out_dropout0.0_weight_decay0.001_grad_clip0.0 | 2 | 44.858 | 4.558 | 23.069 |
| out_dropout0.1_weight_decay0.001_grad_clip0.0 | 2 | 45.61 | 5.855 | 24.151 |
| out_dropout0.0_weight_decay0.001_grad_clip0.1 | 2 | 48.048 | 4.453 | 24.355 |
| out_dropout0.1_weight_decay0.001_grad_clip1.0 | 2 | 45.541 | 6.384 | 24.431 |
| out_dropout0.3_weight_decay0.001_grad_clip0.0 | 2 | 46.198 | 6.914 | 25.023 |
| out_dropout0.2_weight_decay0.001_grad_clip1.0 | 2 | 46.056 | 7.124 | 25.085 |
| out_dropout0.2_weight_decay0.001_grad_clip0.0 | 2 | 46.015 | 7.206 | 25.115 |
| out_dropout0.3_weight_decay0.001_grad_clip1.0 | 2 | 46.124 | 7.329 | 25.233 |
| out_dropout0.1_weight_decay0.001_grad_clip0.1 | 2 | 48.604 | 7.597 | 26.449 |
| out_dropout0.2_weight_decay0.001_grad_clip0.1 | 2 | 49.902 | 7.641 | 27.026 |
| out_dropout0.3_weight_decay0.001_grad_clip0.1 | 2 | 50.614 | 7.905 | 27.484 |

## Every cell

| cell | variant | seeds | val_mape | seed_gap | seconds | vs_reference_pct |
| --- | --- | --- | --- | --- | --- | --- |
| google-10m__lstm-nocritic-madgrad | base | 2 | 19.493 | 0.499 | 73.75 | 0.0 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.1_weight_decay0.001_grad_clip0.1 | 2 | 20.974 | 0.032 | 34.75 | 7.597 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.3_weight_decay0.0_grad_clip0.1 | 2 | 19.132 | 0.004 | 86.65 | -1.849 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.2_weight_decay0.0001_grad_clip1.0 | 2 | 20.185 | 0.234 | 64.8 | 3.553 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.0_weight_decay0.001_grad_clip1.0 | 2 | 20.25 | 0.134 | 80.45 | 3.885 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.1_weight_decay0.0001_grad_clip0.1 | 2 | 20.108 | 0.052 | 88.85 | 3.156 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.0_weight_decay0.001_grad_clip0.0 | 2 | 20.381 | 0.362 | 46.15 | 4.558 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.0_weight_decay0.001_grad_clip0.1 | 2 | 20.361 | 0.136 | 40.95 | 4.453 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.3_weight_decay0.001_grad_clip1.0 | 2 | 20.921 | 0.273 | 21.8 | 7.329 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.2_weight_decay0.001_grad_clip0.1 | 2 | 20.982 | 0.007 | 19.85 | 7.641 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.2_weight_decay0.0_grad_clip0.0 | 2 | 19.253 | 0.075 | 24.0 | -1.229 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.3_weight_decay0.0001_grad_clip1.0 | 2 | 20.117 | 0.101 | 22.05 | 3.2 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.0_weight_decay0.0001_grad_clip0.1 | 2 | 20.074 | 0.185 | 24.05 | 2.979 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.3_weight_decay0.0001_grad_clip0.1 | 2 | 20.146 | 0.058 | 18.6 | 3.353 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.1_weight_decay0.0001_grad_clip0.0 | 2 | 20.032 | 0.146 | 26.75 | 2.768 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.2_weight_decay0.0001_grad_clip0.1 | 2 | 20.181 | 0.045 | 23.4 | 3.53 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.2_weight_decay0.0001_grad_clip0.0 | 2 | 20.112 | 0.086 | 27.7 | 3.176 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.3_weight_decay0.001_grad_clip0.1 | 2 | 21.034 | 0.028 | 22.05 | 7.905 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.3_weight_decay0.001_grad_clip0.0 | 2 | 20.841 | 0.008 | 16.85 | 6.914 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.0_weight_decay0.0001_grad_clip0.0 | 2 | 20.034 | 0.07 | 21.75 | 2.775 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.3_weight_decay0.0_grad_clip1.0 | 2 | 19.308 | 0.237 | 26.05 | -0.95 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.3_weight_decay0.0_grad_clip0.0 | 2 | 19.285 | 0.127 | 25.8 | -1.068 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.2_weight_decay0.0_grad_clip0.1 | 2 | 19.082 | 0.173 | 32.05 | -2.109 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.0_weight_decay0.0001_grad_clip1.0 | 2 | 19.954 | 0.118 | 46.7 | 2.368 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.2_weight_decay0.001_grad_clip0.0 | 2 | 20.898 | 0.167 | 25.85 | 7.206 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.2_weight_decay0.001_grad_clip1.0 | 2 | 20.882 | 0.315 | 26.45 | 7.124 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.0_weight_decay0.0_grad_clip1.0 | 2 | 19.365 | 0.22 | 47.9 | -0.655 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.3_weight_decay0.0001_grad_clip0.0 | 2 | 20.166 | 0.046 | 59.15 | 3.452 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.1_weight_decay0.001_grad_clip1.0 | 2 | 20.737 | 0.049 | 31.7 | 6.384 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.1_weight_decay0.001_grad_clip0.0 | 2 | 20.634 | 0.03 | 64.55 | 5.855 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.1_weight_decay0.0_grad_clip0.0 | 2 | 19.477 | 0.22 | 53.9 | -0.082 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.1_weight_decay0.0_grad_clip0.1 | 2 | 19.25 | 0.325 | 49.7 | -1.248 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.1_weight_decay0.0001_grad_clip1.0 | 2 | 19.71 | 0.396 | 130.85 | 1.112 |
| google-10m__lstm-nocritic-madgrad | out_dropout0.1_weight_decay0.0_grad_clip1.0 | 2 | 19.458 | 0.058 | 49.25 | -0.179 |
| azure_functions-5m__lstm-nocritic-madgrad | base | 2 | 2.491 | 0.187 | 70.1 | 0.0 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.1_weight_decay0.001_grad_clip0.1 | 2 | 3.702 | 0.023 | 48.7 | 48.604 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.3_weight_decay0.0_grad_clip0.1 | 2 | 2.569 | 0.341 | 76.25 | 3.14 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.2_weight_decay0.0001_grad_clip1.0 | 2 | 2.479 | 0.011 | 67.5 | -0.462 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.0_weight_decay0.001_grad_clip1.0 | 2 | 3.606 | 0.004 | 34.95 | 44.746 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.1_weight_decay0.0001_grad_clip0.1 | 2 | 2.705 | 0.38 | 58.5 | 8.604 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.0_weight_decay0.001_grad_clip0.0 | 2 | 3.608 | 0.006 | 24.6 | 44.858 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.0_weight_decay0.001_grad_clip0.1 | 2 | 3.688 | 0.007 | 30.2 | 48.048 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.3_weight_decay0.001_grad_clip1.0 | 2 | 3.64 | 0.009 | 18.0 | 46.124 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.2_weight_decay0.001_grad_clip0.1 | 2 | 3.734 | 0.022 | 18.75 | 49.902 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.2_weight_decay0.0_grad_clip0.0 | 2 | 3.046 | 0.184 | 10.2 | 22.269 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.3_weight_decay0.0001_grad_clip1.0 | 2 | 2.65 | 0.313 | 24.05 | 6.404 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.0_weight_decay0.0001_grad_clip0.1 | 2 | 2.597 | 0.248 | 35.6 | 4.255 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.3_weight_decay0.0001_grad_clip0.1 | 2 | 2.776 | 0.24 | 23.1 | 11.456 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.1_weight_decay0.0001_grad_clip0.0 | 2 | 2.432 | 0.115 | 24.7 | -2.37 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.2_weight_decay0.0001_grad_clip0.1 | 2 | 2.906 | 0.075 | 76.05 | 16.652 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.2_weight_decay0.0001_grad_clip0.0 | 2 | 2.486 | 0.046 | 116.95 | -0.183 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.3_weight_decay0.001_grad_clip0.1 | 2 | 3.752 | 0.031 | 84.0 | 50.614 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.3_weight_decay0.001_grad_clip0.0 | 2 | 3.642 | 0.003 | 76.1 | 46.198 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.0_weight_decay0.0001_grad_clip0.0 | 2 | 2.463 | 0.088 | 112.15 | -1.109 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.3_weight_decay0.0_grad_clip1.0 | 2 | 2.456 | 0.179 | 124.2 | -1.394 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.3_weight_decay0.0_grad_clip0.0 | 2 | 2.71 | 0.593 | 93.9 | 8.785 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.2_weight_decay0.0_grad_clip0.1 | 2 | 2.685 | 0.396 | 116.0 | 7.777 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.0_weight_decay0.0001_grad_clip1.0 | 2 | 2.497 | 0.051 | 86.2 | 0.226 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.2_weight_decay0.001_grad_clip0.0 | 2 | 3.637 | 0.01 | 59.15 | 46.015 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.2_weight_decay0.001_grad_clip1.0 | 2 | 3.638 | 0.008 | 51.15 | 46.056 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.0_weight_decay0.0_grad_clip1.0 | 2 | 2.448 | 0.004 | 69.85 | -1.71 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.3_weight_decay0.0001_grad_clip0.0 | 2 | 2.539 | 0.036 | 52.75 | 1.94 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.1_weight_decay0.001_grad_clip1.0 | 2 | 3.625 | 0.004 | 33.1 | 45.541 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.1_weight_decay0.001_grad_clip0.0 | 2 | 3.627 | 0.001 | 30.3 | 45.61 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.1_weight_decay0.0_grad_clip0.0 | 2 | 2.631 | 0.474 | 53.15 | 5.629 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.1_weight_decay0.0_grad_clip0.1 | 2 | 2.551 | 0.365 | 71.3 | 2.399 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.1_weight_decay0.0001_grad_clip1.0 | 2 | 2.65 | 0.47 | 49.5 | 6.395 |
| azure_functions-5m__lstm-nocritic-madgrad | out_dropout0.1_weight_decay0.0_grad_clip1.0 | 2 | 2.456 | 0.102 | 60.8 | -1.397 |
