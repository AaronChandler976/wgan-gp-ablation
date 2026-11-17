# Screen: nocritic_reg_transformer

136 of 136 trials finished. Validation MAPE (%) only; the test part is not used. Tuning seeds [0, 100], settings while screening: {'train.patience': 150}. Changes are against the variant `base` in the same cell.

Matched tuning budget for the no-critic Transformer: 34 variants (the reported setting plus 33 of the 35 other combinations of dropout, weight decay and feed-forward width, drawn with a fixed seed).

## Variants, best first (average over cells)

| variant | cells | worst_cell_pct | best_cell_pct | mean_change_pct |
| --- | --- | --- | --- | --- |
| dropout0.1_weight_decay0.0001_dim_ff128 | 2 | 0.225 | -1.602 | -0.693 |
| dropout0.1_weight_decay0.0001_dim_ff2048 | 2 | -0.393 | -0.951 | -0.672 |
| dropout0.1_weight_decay0.0001_dim_ff512 | 2 | 0.045 | -1.008 | -0.483 |
| dropout0.1_weight_decay0.0_dim_ff512 | 2 | 0.1 | -0.665 | -0.283 |
| base | 2 | 0.0 | 0.0 | 0.0 |
| dropout0.2_weight_decay0.0001_dim_ff2048 | 2 | 1.767 | -0.225 | 0.766 |
| dropout0.0_weight_decay0.0001_dim_ff512 | 2 | 1.099 | 0.521 | 0.81 |
| dropout0.0_weight_decay0.0_dim_ff2048 | 2 | 2.962 | -1.052 | 0.935 |
| dropout0.3_weight_decay0.0_dim_ff512 | 2 | 2.433 | -0.127 | 1.145 |
| dropout0.0_weight_decay0.0_dim_ff512 | 2 | 1.718 | 0.979 | 1.348 |
| dropout0.2_weight_decay0.0_dim_ff2048 | 2 | 2.925 | 0.231 | 1.569 |
| dropout0.0_weight_decay0.0001_dim_ff2048 | 2 | 4.63 | -1.347 | 1.598 |
| dropout0.0_weight_decay0.0_dim_ff128 | 2 | 3.915 | -0.562 | 1.652 |
| dropout0.0_weight_decay0.0001_dim_ff128 | 2 | 4.836 | -0.878 | 1.939 |
| dropout0.2_weight_decay0.0001_dim_ff512 | 2 | 6.115 | 0.965 | 3.508 |
| dropout0.3_weight_decay0.0001_dim_ff512 | 2 | 6.114 | 3.153 | 4.623 |
| dropout0.2_weight_decay0.0001_dim_ff128 | 2 | 6.584 | 3.124 | 4.84 |
| dropout0.2_weight_decay0.0_dim_ff128 | 2 | 6.267 | 4.295 | 5.277 |
| dropout0.3_weight_decay0.0001_dim_ff128 | 2 | 6.5 | 4.939 | 5.717 |
| dropout0.3_weight_decay0.0_dim_ff128 | 2 | 6.767 | 5.889 | 6.327 |
| dropout0.1_weight_decay0.001_dim_ff2048 | 2 | 11.691 | 1.406 | 6.424 |
| dropout0.0_weight_decay0.001_dim_ff512 | 2 | 7.517 | 7.236 | 7.376 |
| dropout0.3_weight_decay0.0001_dim_ff2048 | 2 | 15.827 | 2.928 | 9.187 |
| dropout0.3_weight_decay0.0_dim_ff2048 | 2 | 16.578 | 2.472 | 9.298 |
| dropout0.1_weight_decay0.001_dim_ff512 | 2 | 12.069 | 7.091 | 9.552 |
| dropout0.3_weight_decay0.001_dim_ff2048 | 2 | 11.675 | 7.488 | 9.562 |
| dropout0.0_weight_decay0.001_dim_ff2048 | 2 | 12.322 | 7.259 | 9.761 |
| dropout0.2_weight_decay0.001_dim_ff2048 | 2 | 12.792 | 7.343 | 10.034 |
| dropout0.2_weight_decay0.001_dim_ff512 | 2 | 12.417 | 8.628 | 10.506 |
| dropout0.3_weight_decay0.001_dim_ff512 | 2 | 12.677 | 8.791 | 10.717 |
| dropout0.1_weight_decay0.001_dim_ff128 | 2 | 13.228 | 9.686 | 11.443 |
| dropout0.2_weight_decay0.001_dim_ff128 | 2 | 13.933 | 10.57 | 12.239 |
| dropout0.0_weight_decay0.001_dim_ff128 | 2 | 15.113 | 11.204 | 13.142 |
| dropout0.3_weight_decay0.001_dim_ff128 | 2 | 13.983 | 12.58 | 13.279 |

## Every cell

| cell | variant | seeds | val_mape | seed_gap | seconds | vs_reference_pct |
| --- | --- | --- | --- | --- | --- | --- |
| google-10m__transformer-nocritic-madgrad | base | 2 | 18.217 | 0.552 | 87.4 | 0.0 |
| google-10m__transformer-nocritic-madgrad | dropout0.0_weight_decay0.001_dim_ff128 | 2 | 20.97 | 0.384 | 34.7 | 15.113 |
| google-10m__transformer-nocritic-madgrad | dropout0.3_weight_decay0.0_dim_ff128 | 2 | 19.289 | 2.867 | 87.05 | 5.889 |
| google-10m__transformer-nocritic-madgrad | dropout0.2_weight_decay0.0001_dim_ff512 | 2 | 19.331 | 2.604 | 80.85 | 6.115 |
| google-10m__transformer-nocritic-madgrad | dropout0.1_weight_decay0.001_dim_ff512 | 2 | 20.415 | 0.175 | 83.85 | 12.069 |
| google-10m__transformer-nocritic-madgrad | dropout0.0_weight_decay0.0001_dim_ff128 | 2 | 18.057 | 0.461 | 106.15 | -0.878 |
| google-10m__transformer-nocritic-madgrad | dropout0.1_weight_decay0.001_dim_ff2048 | 2 | 20.346 | 0.074 | 105.45 | 11.691 |
| google-10m__transformer-nocritic-madgrad | dropout0.1_weight_decay0.001_dim_ff128 | 2 | 20.626 | 0.187 | 96.1 | 13.228 |
| google-10m__transformer-nocritic-madgrad | dropout0.3_weight_decay0.001_dim_ff512 | 2 | 20.526 | 0.129 | 53.5 | 12.677 |
| google-10m__transformer-nocritic-madgrad | dropout0.2_weight_decay0.001_dim_ff128 | 2 | 20.755 | 0.038 | 34.1 | 13.933 |
| google-10m__transformer-nocritic-madgrad | dropout0.2_weight_decay0.0_dim_ff2048 | 2 | 18.259 | 0.189 | 122.5 | 0.231 |
| google-10m__transformer-nocritic-madgrad | dropout0.3_weight_decay0.0001_dim_ff512 | 2 | 19.33 | 2.515 | 82.1 | 6.114 |
| google-10m__transformer-nocritic-madgrad | dropout0.1_weight_decay0.0001_dim_ff128 | 2 | 17.925 | 0.403 | 174.45 | -1.602 |
| google-10m__transformer-nocritic-madgrad | dropout0.3_weight_decay0.0001_dim_ff128 | 2 | 19.401 | 2.735 | 93.35 | 6.5 |
| google-10m__transformer-nocritic-madgrad | dropout0.0_weight_decay0.0001_dim_ff2048 | 2 | 17.971 | 0.347 | 72.5 | -1.347 |
| google-10m__transformer-nocritic-madgrad | dropout0.2_weight_decay0.0001_dim_ff128 | 2 | 19.416 | 2.667 | 94.55 | 6.584 |
| google-10m__transformer-nocritic-madgrad | dropout0.2_weight_decay0.0001_dim_ff2048 | 2 | 18.176 | 0.123 | 155.9 | -0.225 |
| google-10m__transformer-nocritic-madgrad | dropout0.3_weight_decay0.001_dim_ff128 | 2 | 20.764 | 0.076 | 37.0 | 13.983 |
| google-10m__transformer-nocritic-madgrad | dropout0.3_weight_decay0.001_dim_ff2048 | 2 | 20.343 | 0.081 | 139.1 | 11.675 |
| google-10m__transformer-nocritic-madgrad | dropout0.1_weight_decay0.0001_dim_ff2048 | 2 | 18.145 | 0.161 | 101.9 | -0.393 |
| google-10m__transformer-nocritic-madgrad | dropout0.3_weight_decay0.0_dim_ff512 | 2 | 18.193 | 0.303 | 126.45 | -0.127 |
| google-10m__transformer-nocritic-madgrad | dropout0.3_weight_decay0.0_dim_ff2048 | 2 | 21.237 | 0.029 | 45.1 | 16.578 |
| google-10m__transformer-nocritic-madgrad | dropout0.2_weight_decay0.0_dim_ff128 | 2 | 19.358 | 2.813 | 89.9 | 6.267 |
| google-10m__transformer-nocritic-madgrad | dropout0.1_weight_decay0.0001_dim_ff512 | 2 | 18.033 | 0.048 | 123.55 | -1.008 |
| google-10m__transformer-nocritic-madgrad | dropout0.2_weight_decay0.001_dim_ff2048 | 2 | 20.547 | 0.193 | 63.15 | 12.792 |
| google-10m__transformer-nocritic-madgrad | dropout0.2_weight_decay0.001_dim_ff512 | 2 | 20.478 | 0.258 | 66.5 | 12.417 |
| google-10m__transformer-nocritic-madgrad | dropout0.1_weight_decay0.0_dim_ff512 | 2 | 18.095 | 0.006 | 127.35 | -0.665 |
| google-10m__transformer-nocritic-madgrad | dropout0.3_weight_decay0.0001_dim_ff2048 | 2 | 21.1 | 0.224 | 47.8 | 15.827 |
| google-10m__transformer-nocritic-madgrad | dropout0.0_weight_decay0.001_dim_ff512 | 2 | 19.535 | 2.002 | 132.4 | 7.236 |
| google-10m__transformer-nocritic-madgrad | dropout0.0_weight_decay0.001_dim_ff2048 | 2 | 20.461 | 0.385 | 66.15 | 12.322 |
| google-10m__transformer-nocritic-madgrad | dropout0.0_weight_decay0.0_dim_ff2048 | 2 | 18.025 | 0.423 | 76.65 | -1.052 |
| google-10m__transformer-nocritic-madgrad | dropout0.0_weight_decay0.0_dim_ff128 | 2 | 18.114 | 0.766 | 105.45 | -0.562 |
| google-10m__transformer-nocritic-madgrad | dropout0.0_weight_decay0.0001_dim_ff512 | 2 | 18.311 | 0.058 | 109.75 | 0.521 |
| google-10m__transformer-nocritic-madgrad | dropout0.0_weight_decay0.0_dim_ff512 | 2 | 18.395 | 0.067 | 82.1 | 0.979 |
| azure_functions-5m__transformer-nocritic-madgrad | base | 2 | 2.392 | 0.117 | 180.15 | 0.0 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.0_weight_decay0.001_dim_ff128 | 2 | 2.66 | 0.003 | 109.95 | 11.204 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.3_weight_decay0.0_dim_ff128 | 2 | 2.554 | 0.1 | 220.65 | 6.767 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.2_weight_decay0.0001_dim_ff512 | 2 | 2.415 | 0.12 | 219.1 | 0.965 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.1_weight_decay0.001_dim_ff512 | 2 | 2.561 | 0.238 | 180.25 | 7.091 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.0_weight_decay0.0001_dim_ff128 | 2 | 2.507 | 0.058 | 199.85 | 4.836 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.1_weight_decay0.001_dim_ff2048 | 2 | 2.425 | 0.063 | 180.2 | 1.406 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.1_weight_decay0.001_dim_ff128 | 2 | 2.623 | 0.147 | 211.45 | 9.686 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.3_weight_decay0.001_dim_ff512 | 2 | 2.602 | 0.328 | 214.1 | 8.791 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.2_weight_decay0.001_dim_ff128 | 2 | 2.644 | 0.175 | 185.7 | 10.57 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.2_weight_decay0.0_dim_ff2048 | 2 | 2.462 | 0.148 | 174.3 | 2.925 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.3_weight_decay0.0001_dim_ff512 | 2 | 2.467 | 0.172 | 222.35 | 3.153 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.1_weight_decay0.0001_dim_ff128 | 2 | 2.397 | 0.085 | 214.55 | 0.225 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.3_weight_decay0.0001_dim_ff128 | 2 | 2.51 | 0.192 | 218.0 | 4.939 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.0_weight_decay0.0001_dim_ff2048 | 2 | 2.502 | 0.157 | 124.15 | 4.63 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.2_weight_decay0.0001_dim_ff128 | 2 | 2.466 | 0.188 | 211.75 | 3.124 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.2_weight_decay0.0001_dim_ff2048 | 2 | 2.434 | 0.124 | 203.3 | 1.767 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.3_weight_decay0.001_dim_ff128 | 2 | 2.693 | 0.198 | 184.35 | 12.58 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.3_weight_decay0.001_dim_ff2048 | 2 | 2.571 | 0.333 | 198.0 | 7.488 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.1_weight_decay0.0001_dim_ff2048 | 2 | 2.369 | 0.101 | 171.3 | -0.951 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.3_weight_decay0.0_dim_ff512 | 2 | 2.45 | 0.152 | 177.4 | 2.433 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.3_weight_decay0.0_dim_ff2048 | 2 | 2.451 | 0.134 | 159.75 | 2.472 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.2_weight_decay0.0_dim_ff128 | 2 | 2.494 | 0.107 | 155.4 | 4.295 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.1_weight_decay0.0001_dim_ff512 | 2 | 2.393 | 0.072 | 160.5 | 0.045 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.2_weight_decay0.001_dim_ff2048 | 2 | 2.567 | 0.373 | 173.6 | 7.343 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.2_weight_decay0.001_dim_ff512 | 2 | 2.598 | 0.363 | 153.75 | 8.628 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.1_weight_decay0.0_dim_ff512 | 2 | 2.394 | 0.067 | 153.4 | 0.1 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.3_weight_decay0.0001_dim_ff2048 | 2 | 2.462 | 0.122 | 132.45 | 2.928 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.0_weight_decay0.001_dim_ff512 | 2 | 2.571 | 0.291 | 71.3 | 7.517 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.0_weight_decay0.001_dim_ff2048 | 2 | 2.565 | 0.218 | 75.4 | 7.259 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.0_weight_decay0.0_dim_ff2048 | 2 | 2.463 | 0.104 | 77.85 | 2.962 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.0_weight_decay0.0_dim_ff128 | 2 | 2.485 | 0.033 | 112.8 | 3.915 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.0_weight_decay0.0001_dim_ff512 | 2 | 2.418 | 0.104 | 95.6 | 1.099 |
| azure_functions-5m__transformer-nocritic-madgrad | dropout0.0_weight_decay0.0_dim_ff512 | 2 | 2.433 | 0.132 | 95.05 | 1.718 |
