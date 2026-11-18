# Screen: nocritic_recheck

16 of 16 trials finished. Validation MAPE (%) only; the test part is not used. Tuning seeds [200, 300], settings while screening: {'train.patience': 150}. Changes are against the variant `base` in the same cell.

Confirmation of the no-critic LSTM screen's winner (gradient clipping at 1.0) on tuning seeds it has not seen (200, 300), and on the Adam cells. Nothing is selected here.

## Variants, best first (average over cells)

| variant | cells | worst_cell_pct | best_cell_pct | mean_change_pct |
| --- | --- | --- | --- | --- |
| grad_clip1.0 | 4 | 5.174 | -5.262 | -0.432 |
| base | 4 | 0.0 | 0.0 | 0.0 |

## Every cell

| cell | variant | seeds | val_mape | seed_gap | seconds | vs_reference_pct |
| --- | --- | --- | --- | --- | --- | --- |
| google-10m__lstm-nocritic-madgrad | base | 2 | 19.938 | 0.236 | 44.2 | 0.0 |
| google-10m__lstm-nocritic-madgrad | grad_clip1.0 | 2 | 19.61 | 0.981 | 60.5 | -1.646 |
| google-10m__lstm-nocritic-adam | base | 2 | 20.096 | 1.07 | 34.65 | 0.0 |
| google-10m__lstm-nocritic-adam | grad_clip1.0 | 2 | 20.154 | 0.459 | 27.4 | 0.289 |
| azure_functions-5m__lstm-nocritic-madgrad | base | 2 | 2.384 | 0.016 | 76.35 | 0.0 |
| azure_functions-5m__lstm-nocritic-madgrad | grad_clip1.0 | 2 | 2.508 | 0.043 | 72.2 | 5.174 |
| azure_functions-5m__lstm-nocritic-adam | base | 2 | 2.661 | 0.551 | 64.15 | 0.0 |
| azure_functions-5m__lstm-nocritic-adam | grad_clip1.0 | 2 | 2.521 | 0.092 | 84.95 | -5.262 |
