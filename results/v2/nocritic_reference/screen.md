# Screen: nocritic_reference

8 of 8 trials finished. Validation MAPE (%) only; the test part is not used. Tuning seeds [0, 100], settings while screening: {'train.patience': 150}. Changes are against the variant `base` in the same cell.

The matching no-critic configurations at their own tuned settings, under the same screening protocol (tuning seeds, early stopping), as the level the critic variants are compared with.

## Variants, best first (average over cells)

| variant | cells | worst_cell_pct | best_cell_pct | mean_change_pct |
| --- | --- | --- | --- | --- |
| base | 4 | 0.0 | 0.0 | 0.0 |

## Every cell

| cell | variant | seeds | val_mape | seed_gap | seconds | vs_reference_pct |
| --- | --- | --- | --- | --- | --- | --- |
| google-10m__transformer-nocritic-madgrad | base | 2 | 18.217 | 0.552 | 91.3 | 0.0 |
| google-10m__lstm-nocritic-madgrad | base | 2 | 19.493 | 0.499 | 53.9 | 0.0 |
| azure_functions-5m__transformer-nocritic-madgrad | base | 2 | 2.392 | 0.117 | 171.75 | 0.0 |
| azure_functions-5m__lstm-nocritic-madgrad | base | 2 | 2.491 | 0.187 | 93.55 | 0.0 |
