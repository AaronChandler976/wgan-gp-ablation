# Screen: google_history

224 of 224 trials finished. Validation MAPE (%) only; the test part is not used. Tuning seeds [0, 100], settings while screening: {'train.patience': 150}. Changes are against the variant `tuned` in the same cell.

History length on the two Google series, outside the tuning grid (48, 96). 5 and 472 are the ends of the range LoadDynamics selected for Google (its Table IV); 144 and 288 are one and two days at 10 minutes. Only the history length changes: learning rate and model size stay at each cell's tuned values. `tuned` is the cell's own tuned history (48 or 96) and is the reference.

## Variants, best first (average over cells)

| variant | cells | worst_cell_pct | best_cell_pct | mean_change_pct |
| --- | --- | --- | --- | --- |
| tuned | 16 | 0.0 | 0.0 | 0.0 |
| h48 | 16 | 7.05 | -1.554 | 0.487 |
| h96 | 16 | 10.352 | 0.0 | 2.293 |
| h144 | 16 | 12.449 | -0.589 | 3.718 |
| h288 | 16 | 11.309 | -1.404 | 3.999 |
| h472 | 16 | 18.165 | -0.556 | 6.199 |
| h5 | 16 | 31.676 | 1.517 | 8.145 |

## Every cell

| cell | variant | seeds | val_mape | seed_gap | seconds | vs_reference_pct |
| --- | --- | --- | --- | --- | --- | --- |
| google-10m__lstm-nocritic-adam | tuned | 2 | 19.692 | 0.76 | 10.55 | 0.0 |
| google-10m__lstm-nocritic-adam | h5 | 2 | 20.626 | 0.019 | 11.25 | 4.742 |
| google-10m__lstm-nocritic-adam | h48 | 2 | 19.692 | 0.76 | 13.05 | 0.0 |
| google-10m__lstm-nocritic-adam | h96 | 2 | 19.742 | 0.227 | 22.6 | 0.252 |
| google-10m__lstm-nocritic-adam | h144 | 2 | 20.338 | 0.063 | 29.0 | 3.278 |
| google-10m__lstm-nocritic-adam | h288 | 2 | 20.756 | 0.563 | 38.55 | 5.4 |
| google-10m__lstm-nocritic-adam | h472 | 2 | 20.336 | 0.335 | 29.4 | 3.268 |
| google-10m__lstm-nocritic-madgrad | tuned | 2 | 19.493 | 0.499 | 23.2 | 0.0 |
| google-10m__lstm-nocritic-madgrad | h5 | 2 | 20.186 | 0.065 | 12.7 | 3.556 |
| google-10m__lstm-nocritic-madgrad | h48 | 2 | 19.493 | 0.499 | 16.7 | 0.0 |
| google-10m__lstm-nocritic-madgrad | h96 | 2 | 19.622 | 0.063 | 10.9 | 0.663 |
| google-10m__lstm-nocritic-madgrad | h144 | 2 | 19.92 | 0.426 | 11.75 | 2.19 |
| google-10m__lstm-nocritic-madgrad | h288 | 2 | 19.346 | 0.258 | 23.05 | -0.753 |
| google-10m__lstm-nocritic-madgrad | h472 | 2 | 19.695 | 0.569 | 15.55 | 1.037 |
| google-10m__lstm-critic-adam | tuned | 2 | 20.046 | 0.512 | 104.65 | 0.0 |
| google-10m__lstm-critic-adam | h5 | 2 | 21.167 | 1.058 | 115.2 | 5.592 |
| google-10m__lstm-critic-adam | h48 | 2 | 20.64 | 0.192 | 82.4 | 2.966 |
| google-10m__lstm-critic-adam | h96 | 2 | 20.046 | 0.512 | 112.9 | 0.0 |
| google-10m__lstm-critic-adam | h144 | 2 | 20.84 | 0.005 | 68.6 | 3.961 |
| google-10m__lstm-critic-adam | h288 | 2 | 21.166 | 0.08 | 73.95 | 5.59 |
| google-10m__lstm-critic-adam | h472 | 2 | 19.934 | 0.536 | 107.3 | -0.556 |
| google-10m__lstm-critic-madgrad | tuned | 2 | 19.986 | 1.345 | 202.95 | 0.0 |
| google-10m__lstm-critic-madgrad | h5 | 2 | 21.373 | 0.153 | 79.5 | 6.937 |
| google-10m__lstm-critic-madgrad | h48 | 2 | 21.396 | 0.607 | 125.7 | 7.05 |
| google-10m__lstm-critic-madgrad | h96 | 2 | 19.986 | 1.345 | 272.35 | 0.0 |
| google-10m__lstm-critic-madgrad | h144 | 2 | 19.976 | 0.035 | 185.25 | -0.052 |
| google-10m__lstm-critic-madgrad | h288 | 2 | 20.019 | 0.12 | 379.8 | 0.162 |
| google-10m__lstm-critic-madgrad | h472 | 2 | 20.03 | 0.074 | 310.7 | 0.217 |
| google-10m__transformer-nocritic-adam | tuned | 2 | 18.311 | 0.675 | 42.75 | 0.0 |
| google-10m__transformer-nocritic-adam | h5 | 2 | 20.574 | 0.062 | 47.0 | 12.361 |
| google-10m__transformer-nocritic-adam | h48 | 2 | 18.311 | 0.675 | 37.95 | 0.0 |
| google-10m__transformer-nocritic-adam | h96 | 2 | 18.666 | 0.367 | 121.85 | 1.942 |
| google-10m__transformer-nocritic-adam | h144 | 2 | 18.695 | 0.909 | 140.7 | 2.096 |
| google-10m__transformer-nocritic-adam | h288 | 2 | 19.286 | 0.484 | 330.25 | 5.324 |
| google-10m__transformer-nocritic-adam | h472 | 2 | 19.912 | 0.785 | 341.5 | 8.745 |
| google-10m__transformer-nocritic-madgrad | tuned | 2 | 18.217 | 0.552 | 90.95 | 0.0 |
| google-10m__transformer-nocritic-madgrad | h5 | 2 | 20.377 | 0.071 | 58.45 | 11.858 |
| google-10m__transformer-nocritic-madgrad | h48 | 2 | 18.217 | 0.552 | 70.1 | 0.0 |
| google-10m__transformer-nocritic-madgrad | h96 | 2 | 18.337 | 0.148 | 218.85 | 0.661 |
| google-10m__transformer-nocritic-madgrad | h144 | 2 | 19.778 | 2.606 | 98.35 | 8.573 |
| google-10m__transformer-nocritic-madgrad | h288 | 2 | 18.756 | 1.027 | 653.0 | 2.962 |
| google-10m__transformer-nocritic-madgrad | h472 | 2 | 20.476 | 1.491 | 448.7 | 12.404 |
| google-10m__transformer-critic-adam | tuned | 2 | 19.763 | 2.701 | 146.95 | 0.0 |
| google-10m__transformer-critic-adam | h5 | 2 | 21.319 | 0.172 | 181.0 | 7.872 |
| google-10m__transformer-critic-adam | h48 | 2 | 19.763 | 2.701 | 134.5 | 0.0 |
| google-10m__transformer-critic-adam | h96 | 2 | 20.118 | 0.559 | 310.85 | 1.795 |
| google-10m__transformer-critic-adam | h144 | 2 | 20.199 | 0.185 | 280.75 | 2.205 |
| google-10m__transformer-critic-adam | h288 | 2 | 20.429 | 0.042 | 234.45 | 3.372 |
| google-10m__transformer-critic-adam | h472 | 2 | 20.482 | 0.186 | 648.8 | 3.64 |
| google-10m__transformer-critic-madgrad | tuned | 2 | 18.468 | 0.538 | 184.65 | 0.0 |
| google-10m__transformer-critic-madgrad | h5 | 2 | 21.242 | 1.364 | 99.35 | 15.02 |
| google-10m__transformer-critic-madgrad | h48 | 2 | 18.468 | 0.538 | 274.6 | 0.0 |
| google-10m__transformer-critic-madgrad | h96 | 2 | 19.248 | 0.674 | 103.75 | 4.221 |
| google-10m__transformer-critic-madgrad | h144 | 2 | 18.576 | 0.447 | 205.2 | 0.585 |
| google-10m__transformer-critic-madgrad | h288 | 2 | 19.363 | 0.382 | 477.95 | 4.848 |
| google-10m__transformer-critic-madgrad | h472 | 2 | 19.472 | 0.967 | 670.35 | 5.437 |
| google-30m__lstm-nocritic-adam | tuned | 2 | 17.469 | 0.16 | 3.85 | 0.0 |
| google-30m__lstm-nocritic-adam | h5 | 2 | 17.973 | 0.748 | 1.55 | 2.886 |
| google-30m__lstm-nocritic-adam | h48 | 2 | 17.401 | 0.484 | 3.7 | -0.39 |
| google-30m__lstm-nocritic-adam | h96 | 2 | 17.469 | 0.16 | 1.6 | 0.0 |
| google-30m__lstm-nocritic-adam | h144 | 2 | 17.366 | 0.026 | 3.9 | -0.589 |
| google-30m__lstm-nocritic-adam | h288 | 2 | 17.259 | 0.503 | 4.4 | -1.202 |
| google-30m__lstm-nocritic-adam | h472 | 2 | 17.704 | 0.147 | 1.35 | 1.344 |
| google-30m__lstm-nocritic-madgrad | tuned | 2 | 17.246 | 0.858 | 5.1 | 0.0 |
| google-30m__lstm-nocritic-madgrad | h5 | 2 | 17.508 | 0.123 | 2.7 | 1.517 |
| google-30m__lstm-nocritic-madgrad | h48 | 2 | 17.246 | 0.858 | 4.3 | 0.0 |
| google-30m__lstm-nocritic-madgrad | h96 | 2 | 17.427 | 0.486 | 4.75 | 1.048 |
| google-30m__lstm-nocritic-madgrad | h144 | 2 | 17.512 | 0.11 | 2.15 | 1.54 |
| google-30m__lstm-nocritic-madgrad | h288 | 2 | 17.614 | 0.352 | 4.05 | 2.131 |
| google-30m__lstm-nocritic-madgrad | h472 | 2 | 17.976 | 0.248 | 1.45 | 4.229 |
| google-30m__lstm-critic-adam | tuned | 2 | 17.598 | 0.815 | 34.2 | 0.0 |
| google-30m__lstm-critic-adam | h5 | 2 | 17.914 | 0.016 | 20.95 | 1.795 |
| google-30m__lstm-critic-adam | h48 | 2 | 17.598 | 0.815 | 22.55 | 0.0 |
| google-30m__lstm-critic-adam | h96 | 2 | 17.986 | 0.193 | 23.5 | 2.204 |
| google-30m__lstm-critic-adam | h144 | 2 | 17.922 | 0.443 | 18.3 | 1.839 |
| google-30m__lstm-critic-adam | h288 | 2 | 17.351 | 0.772 | 17.9 | -1.404 |
| google-30m__lstm-critic-adam | h472 | 2 | 18.513 | 1.449 | 10.6 | 5.198 |
| google-30m__lstm-critic-madgrad | tuned | 2 | 17.415 | 0.318 | 31.8 | 0.0 |
| google-30m__lstm-critic-madgrad | h5 | 2 | 22.931 | 0.567 | 26.7 | 31.676 |
| google-30m__lstm-critic-madgrad | h48 | 2 | 17.415 | 0.318 | 49.05 | 0.0 |
| google-30m__lstm-critic-madgrad | h96 | 2 | 17.747 | 0.029 | 22.35 | 1.91 |
| google-30m__lstm-critic-madgrad | h144 | 2 | 17.732 | 0.095 | 37.9 | 1.825 |
| google-30m__lstm-critic-madgrad | h288 | 2 | 17.574 | 0.101 | 24.35 | 0.918 |
| google-30m__lstm-critic-madgrad | h472 | 2 | 18.081 | 0.142 | 23.5 | 3.824 |
| google-30m__transformer-nocritic-adam | tuned | 2 | 16.686 | 1.746 | 8.7 | 0.0 |
| google-30m__transformer-nocritic-adam | h5 | 2 | 17.71 | 0.298 | 5.95 | 6.136 |
| google-30m__transformer-nocritic-adam | h48 | 2 | 16.686 | 1.746 | 10.35 | 0.0 |
| google-30m__transformer-nocritic-adam | h96 | 2 | 18.414 | 0.772 | 8.85 | 10.352 |
| google-30m__transformer-nocritic-adam | h144 | 2 | 18.763 | 0.219 | 18.5 | 12.449 |
| google-30m__transformer-nocritic-adam | h288 | 2 | 18.516 | 1.097 | 25.7 | 10.967 |
| google-30m__transformer-nocritic-adam | h472 | 2 | 19.303 | 0.278 | 28.75 | 15.682 |
| google-30m__transformer-nocritic-madgrad | tuned | 2 | 16.239 | 0.739 | 15.2 | 0.0 |
| google-30m__transformer-nocritic-madgrad | h5 | 2 | 17.681 | 0.119 | 9.45 | 8.881 |
| google-30m__transformer-nocritic-madgrad | h48 | 2 | 16.239 | 0.739 | 17.35 | 0.0 |
| google-30m__transformer-nocritic-madgrad | h96 | 2 | 17.157 | 1.463 | 15.1 | 5.654 |
| google-30m__transformer-nocritic-madgrad | h144 | 2 | 17.241 | 0.257 | 26.9 | 6.174 |
| google-30m__transformer-nocritic-madgrad | h288 | 2 | 17.912 | 0.814 | 50.2 | 10.307 |
| google-30m__transformer-nocritic-madgrad | h472 | 2 | 19.188 | 0.142 | 32.5 | 18.165 |
| google-30m__transformer-critic-adam | tuned | 2 | 16.977 | 1.066 | 66.4 | 0.0 |
| google-30m__transformer-critic-adam | h5 | 2 | 18.188 | 0.232 | 39.25 | 7.137 |
| google-30m__transformer-critic-adam | h48 | 2 | 16.977 | 1.066 | 56.35 | 0.0 |
| google-30m__transformer-critic-adam | h96 | 2 | 18.099 | 1.357 | 51.9 | 6.611 |
| google-30m__transformer-critic-adam | h144 | 2 | 18.505 | 0.137 | 36.05 | 9.002 |
| google-30m__transformer-critic-adam | h288 | 2 | 18.897 | 0.027 | 54.65 | 11.309 |
| google-30m__transformer-critic-adam | h472 | 2 | 19.026 | 0.057 | 64.6 | 12.074 |
| google-30m__transformer-critic-madgrad | tuned | 2 | 17.606 | 1.668 | 75.05 | 0.0 |
| google-30m__transformer-critic-madgrad | h5 | 2 | 18.614 | 0.534 | 65.9 | 5.726 |
| google-30m__transformer-critic-madgrad | h48 | 2 | 17.332 | 0.255 | 66.9 | -1.554 |
| google-30m__transformer-critic-madgrad | h96 | 2 | 17.606 | 1.668 | 91.7 | 0.0 |
| google-30m__transformer-critic-madgrad | h144 | 2 | 18.547 | 0.086 | 32.15 | 5.343 |
| google-30m__transformer-critic-madgrad | h288 | 2 | 18.537 | 0.077 | 95.2 | 5.286 |
| google-30m__transformer-critic-madgrad | h472 | 2 | 18.775 | 0.906 | 143.0 | 6.637 |
