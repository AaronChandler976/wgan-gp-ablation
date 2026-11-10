# Data

| Folder | In git? | Contents |
| --- | --- | --- |
| `raw/` | No | Downloads, as published. Recreate with `python -m cwp.data download`. |
| `processed/` | Yes | One CSV per dataset and interval: `t` (interval index), `job_count`. |
| `splits/` | Yes | Train / validation / test boundaries for each processed series. |

```bash
python -m cwp.data download   --dataset google            # 500 small files, job_events only
python -m cwp.data preprocess --dataset google            # -> processed/google_10m.csv, google_30m.csv
python -m cwp.data download   --dataset azure_functions   # one archive, a few hundred MB
python -m cwp.data preprocess --dataset azure_functions   # -> processed/azure_functions_5m.csv, _10m.csv
python -m cwp.data list
```

Only one person needs to do this. Commit `processed/` and `splits/` so everyone trains on
identical files. The split file stores the row count and a hash of the processed CSV, and
loading fails if they no longer match.

The split is chronological: the first 60% is training, the next 20% validation, the last 20%
test. Values are scaled to (-1, 1) with the training part's minimum and maximum.

`synthetic_seasonal_10m.csv` and `synthetic_spiky_10m.csv` are generated series for tests and
timing (`python -m cwp.data synthetic`). Do not report results on them.

Sources and licenses:

- Google Cluster 2011 trace, CC BY: https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md
- Azure Functions 2019 trace, CC BY: https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md
