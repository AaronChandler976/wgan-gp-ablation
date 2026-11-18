# Screen: critic_s3_knobs

128 of 128 trials finished. Validation MAPE (%) only; the test part is not used. Tuning seeds [0, 100], settings while screening: {'train.patience': 150}. Changes are against the variant `n1_gp1_h32_lr1` in the same cell.

Stage 3 of the critic screen: the critic's own settings at adversarial weight 0.01. 16 of the 24 combinations of critic updates per step {1, 5}, gradient penalty weight {1, 10}, critic width {32, 128} and critic learning-rate scale {0.25, 1, 4}, drawn with random.Random(0).sample. Compare with `w0.01` of critic_s1_weight.

## Variants, best first (average over cells)

| variant | cells | worst_cell_pct | best_cell_pct | mean_change_pct |
| --- | --- | --- | --- | --- |
| n5_gp1_h32_lr1 | 4 | -0.528 | -2.63 | -1.602 |
| n5_gp1_h128_lr0.25 | 4 | -0.541 | -2.537 | -1.411 |
| n5_gp1_h128_lr1 | 4 | -0.412 | -1.327 | -1.054 |
| n5_gp10_h32_lr0.25 | 4 | 0.312 | -1.782 | -0.926 |
| n5_gp10_h32_lr1 | 4 | 1.071 | -2.671 | -0.762 |
| n5_gp1_h32_lr0.25 | 4 | 0.515 | -2.248 | -0.638 |
| n1_gp1_h32_lr1 | 4 | 0.0 | 0.0 | 0.0 |
| n1_gp1_h128_lr4 | 4 | 1.511 | -0.486 | 0.166 |
| n1_gp10_h128_lr1 | 4 | 1.083 | 0.387 | 0.656 |
| n1_gp1_h32_lr4 | 4 | 2.306 | -0.107 | 0.945 |
| n1_gp10_h32_lr4 | 4 | 6.51 | -1.681 | 1.272 |
| n1_gp10_h128_lr0.25 | 4 | 5.342 | -0.214 | 1.451 |
| n1_gp1_h128_lr1 | 4 | 8.011 | -0.128 | 2.288 |
| n5_gp10_h32_lr4 | 4 | 12.797 | -1.34 | 2.289 |
| n1_gp1_h128_lr0.25 | 4 | 6.137 | 0.355 | 2.405 |
| n5_gp10_h128_lr4 | 4 | 3840515.698 | -2.052 | 1392.292 |

## Every cell

| cell | variant | seeds | val_mape | seed_gap | seconds | vs_reference_pct |
| --- | --- | --- | --- | --- | --- | --- |
| google-10m__transformer-critic-madgrad | n1_gp1_h32_lr1 | 2 | 18.518 | 0.294 | 112.75 | 0.0 |
| google-10m__transformer-critic-madgrad | n1_gp1_h32_lr4 | 2 | 18.71 | 0.462 | 114.05 | 1.038 |
| google-10m__transformer-critic-madgrad | n1_gp1_h128_lr0.25 | 2 | 18.991 | 0.345 | 117.85 | 2.557 |
| google-10m__transformer-critic-madgrad | n1_gp1_h128_lr1 | 2 | 18.677 | 0.064 | 118.05 | 0.861 |
| google-10m__transformer-critic-madgrad | n1_gp1_h128_lr4 | 2 | 18.798 | 0.259 | 116.6 | 1.511 |
| google-10m__transformer-critic-madgrad | n1_gp10_h32_lr4 | 2 | 18.467 | 0.022 | 117.6 | -0.276 |
| google-10m__transformer-critic-madgrad | n1_gp10_h128_lr0.25 | 2 | 18.658 | 0.232 | 116.2 | 0.756 |
| google-10m__transformer-critic-madgrad | n1_gp10_h128_lr1 | 2 | 18.718 | 0.497 | 148.5 | 1.083 |
| google-10m__transformer-critic-madgrad | n5_gp1_h32_lr0.25 | 2 | 18.473 | 0.17 | 281.4 | -0.241 |
| google-10m__transformer-critic-madgrad | n5_gp1_h32_lr1 | 2 | 18.177 | 0.089 | 290.95 | -1.84 |
| google-10m__transformer-critic-madgrad | n5_gp1_h128_lr0.25 | 2 | 18.418 | 0.156 | 235.3 | -0.541 |
| google-10m__transformer-critic-madgrad | n5_gp1_h128_lr1 | 2 | 18.278 | 0.013 | 290.5 | -1.295 |
| google-10m__transformer-critic-madgrad | n5_gp10_h32_lr0.25 | 2 | 18.318 | 0.48 | 314.25 | -1.081 |
| google-10m__transformer-critic-madgrad | n5_gp10_h32_lr1 | 2 | 18.369 | 0.205 | 279.25 | -0.803 |
| google-10m__transformer-critic-madgrad | n5_gp10_h32_lr4 | 2 | 18.307 | 0.278 | 221.35 | -1.138 |
| google-10m__transformer-critic-madgrad | n5_gp10_h128_lr4 | 2 | 18.59 | 0.205 | 271.35 | 0.388 |
| google-10m__lstm-critic-madgrad | n1_gp1_h32_lr1 | 2 | 19.968 | 0.343 | 108.85 | 0.0 |
| google-10m__lstm-critic-madgrad | n1_gp1_h32_lr4 | 2 | 19.947 | 0.362 | 89.75 | -0.107 |
| google-10m__lstm-critic-madgrad | n1_gp1_h128_lr0.25 | 2 | 20.102 | 0.019 | 58.4 | 0.672 |
| google-10m__lstm-critic-madgrad | n1_gp1_h128_lr1 | 2 | 19.942 | 0.343 | 92.45 | -0.128 |
| google-10m__lstm-critic-madgrad | n1_gp1_h128_lr4 | 2 | 19.962 | 0.315 | 77.8 | -0.031 |
| google-10m__lstm-critic-madgrad | n1_gp10_h32_lr4 | 2 | 20.112 | 0.052 | 67.15 | 0.722 |
| google-10m__lstm-critic-madgrad | n1_gp10_h128_lr0.25 | 2 | 19.971 | 0.327 | 81.05 | 0.018 |
| google-10m__lstm-critic-madgrad | n1_gp10_h128_lr1 | 2 | 20.116 | 0.004 | 62.2 | 0.741 |
| google-10m__lstm-critic-madgrad | n5_gp1_h32_lr0.25 | 2 | 19.856 | 0.052 | 290.45 | -0.559 |
| google-10m__lstm-critic-madgrad | n5_gp1_h32_lr1 | 2 | 19.862 | 0.117 | 396.35 | -0.528 |
| google-10m__lstm-critic-madgrad | n5_gp1_h128_lr0.25 | 2 | 19.858 | 0.058 | 521.3 | -0.549 |
| google-10m__lstm-critic-madgrad | n5_gp1_h128_lr1 | 2 | 19.886 | 0.105 | 569.1 | -0.412 |
| google-10m__lstm-critic-madgrad | n5_gp10_h32_lr0.25 | 2 | 19.74 | 0.296 | 664.2 | -1.143 |
| google-10m__lstm-critic-madgrad | n5_gp10_h32_lr1 | 2 | 19.846 | 0.108 | 582.55 | -0.608 |
| google-10m__lstm-critic-madgrad | n5_gp10_h32_lr4 | 2 | 19.869 | 0.106 | 539.15 | -0.494 |
| google-10m__lstm-critic-madgrad | n5_gp10_h128_lr4 | 2 | 766888.065 | 1533731.086 | 174.45 | 3840515.698 |
| azure_functions-5m__transformer-critic-madgrad | n1_gp1_h32_lr1 | 2 | 2.507 | 0.213 | 265.15 | 0.0 |
| azure_functions-5m__transformer-critic-madgrad | n1_gp1_h32_lr4 | 2 | 2.521 | 0.231 | 221.25 | 0.558 |
| azure_functions-5m__transformer-critic-madgrad | n1_gp1_h128_lr0.25 | 2 | 2.515 | 0.247 | 222.85 | 0.355 |
| azure_functions-5m__transformer-critic-madgrad | n1_gp1_h128_lr1 | 2 | 2.522 | 0.268 | 259.65 | 0.614 |
| azure_functions-5m__transformer-critic-madgrad | n1_gp1_h128_lr4 | 2 | 2.494 | 0.231 | 231.85 | -0.486 |
| azure_functions-5m__transformer-critic-madgrad | n1_gp10_h32_lr4 | 2 | 2.464 | 0.151 | 234.75 | -1.681 |
| azure_functions-5m__transformer-critic-madgrad | n1_gp10_h128_lr0.25 | 2 | 2.501 | 0.213 | 323.3 | -0.214 |
| azure_functions-5m__transformer-critic-madgrad | n1_gp10_h128_lr1 | 2 | 2.517 | 0.239 | 219.4 | 0.415 |
| azure_functions-5m__transformer-critic-madgrad | n5_gp1_h32_lr0.25 | 2 | 2.45 | 0.189 | 548.7 | -2.248 |
| azure_functions-5m__transformer-critic-madgrad | n5_gp1_h32_lr1 | 2 | 2.441 | 0.152 | 691.3 | -2.63 |
| azure_functions-5m__transformer-critic-madgrad | n5_gp1_h128_lr0.25 | 2 | 2.443 | 0.114 | 726.9 | -2.537 |
| azure_functions-5m__transformer-critic-madgrad | n5_gp1_h128_lr1 | 2 | 2.473 | 0.196 | 477.1 | -1.327 |
| azure_functions-5m__transformer-critic-madgrad | n5_gp10_h32_lr0.25 | 2 | 2.462 | 0.16 | 504.9 | -1.782 |
| azure_functions-5m__transformer-critic-madgrad | n5_gp10_h32_lr1 | 2 | 2.44 | 0.151 | 528.7 | -2.671 |
| azure_functions-5m__transformer-critic-madgrad | n5_gp10_h32_lr4 | 2 | 2.473 | 0.139 | 617.45 | -1.34 |
| azure_functions-5m__transformer-critic-madgrad | n5_gp10_h128_lr4 | 2 | 2.455 | 0.157 | 621.1 | -2.052 |
| azure_functions-5m__lstm-critic-madgrad | n1_gp1_h32_lr1 | 2 | 2.467 | 0.019 | 164.3 | 0.0 |
| azure_functions-5m__lstm-critic-madgrad | n1_gp1_h32_lr4 | 2 | 2.524 | 0.301 | 138.35 | 2.306 |
| azure_functions-5m__lstm-critic-madgrad | n1_gp1_h128_lr0.25 | 2 | 2.619 | 0.46 | 122.3 | 6.137 |
| azure_functions-5m__lstm-critic-madgrad | n1_gp1_h128_lr1 | 2 | 2.665 | 0.463 | 126.5 | 8.011 |
| azure_functions-5m__lstm-critic-madgrad | n1_gp1_h128_lr4 | 2 | 2.459 | 0.097 | 156.3 | -0.319 |
| azure_functions-5m__lstm-critic-madgrad | n1_gp10_h32_lr4 | 2 | 2.628 | 0.183 | 111.85 | 6.51 |
| azure_functions-5m__lstm-critic-madgrad | n1_gp10_h128_lr0.25 | 2 | 2.599 | 0.261 | 121.5 | 5.342 |
| azure_functions-5m__lstm-critic-madgrad | n1_gp10_h128_lr1 | 2 | 2.477 | 0.047 | 207.95 | 0.387 |
| azure_functions-5m__lstm-critic-madgrad | n5_gp1_h32_lr0.25 | 2 | 2.48 | 0.092 | 414.5 | 0.515 |
| azure_functions-5m__lstm-critic-madgrad | n5_gp1_h32_lr1 | 2 | 2.433 | 0.019 | 509.8 | -1.397 |
| azure_functions-5m__lstm-critic-madgrad | n5_gp1_h128_lr0.25 | 2 | 2.418 | 0.021 | 611.65 | -2.001 |
| azure_functions-5m__lstm-critic-madgrad | n5_gp1_h128_lr1 | 2 | 2.438 | 0.004 | 447.5 | -1.178 |
| azure_functions-5m__lstm-critic-madgrad | n5_gp10_h32_lr0.25 | 2 | 2.475 | 0.032 | 347.5 | 0.312 |
| azure_functions-5m__lstm-critic-madgrad | n5_gp10_h32_lr1 | 2 | 2.494 | 0.123 | 182.5 | 1.071 |
| azure_functions-5m__lstm-critic-madgrad | n5_gp10_h32_lr4 | 2 | 2.783 | 0.749 | 136.15 | 12.797 |
| azure_functions-5m__lstm-critic-madgrad | n5_gp10_h128_lr4 | 2 | 3.24 | 0.704 | 91.85 | 31.321 |
