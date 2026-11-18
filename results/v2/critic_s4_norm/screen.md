# Screen: critic_s4_norm

16 of 16 trials finished. Validation MAPE (%) only; the test part is not used. Tuning seeds [0, 100], settings while screening: {'train.patience': 150}. Changes are against the variant `w0.01_spectral` in the same cell.

Stage 4 of the critic screen: spectral normalization and layer normalization in the critic, on the setting carried forward from stage 3 (adversarial weight 0.01, the reported critic settings, since no stage-3 combination beat it). Compare with `w0.01` of critic_s1_weight.

## Variants, best first (average over cells)

| variant | cells | worst_cell_pct | best_cell_pct | mean_change_pct |
| --- | --- | --- | --- | --- |
| w0.01_spectral | 4 | 0.0 | 0.0 | 0.0 |
| w0.01_layer | 4 | 3.667 | -1.369 | 1.241 |

## Every cell

| cell | variant | seeds | val_mape | seed_gap | seconds | vs_reference_pct |
| --- | --- | --- | --- | --- | --- | --- |
| google-10m__transformer-critic-madgrad | w0.01_spectral | 2 | 18.467 | 0.046 | 261.25 | 0.0 |
| google-10m__transformer-critic-madgrad | w0.01_layer | 2 | 18.58 | 0.491 | 368.7 | 0.611 |
| google-10m__lstm-critic-madgrad | w0.01_spectral | 2 | 19.685 | 0.232 | 508.85 | 0.0 |
| google-10m__lstm-critic-madgrad | w0.01_layer | 2 | 20.103 | 0.559 | 336.5 | 2.122 |
| azure_functions-5m__transformer-critic-madgrad | w0.01_spectral | 2 | 2.464 | 0.177 | 530.2 | 0.0 |
| azure_functions-5m__transformer-critic-madgrad | w0.01_layer | 2 | 2.43 | 0.117 | 773.25 | -1.369 |
| azure_functions-5m__lstm-critic-madgrad | w0.01_spectral | 2 | 2.537 | 0.103 | 422.35 | 0.0 |
| azure_functions-5m__lstm-critic-madgrad | w0.01_layer | 2 | 2.63 | 0.404 | 258.15 | 3.667 |
