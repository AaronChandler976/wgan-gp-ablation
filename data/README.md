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
- Azure VM 2017 trace (`AzurePublicDatasetV1`), CC BY 4.0: https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV1.md
- English Wikipedia hourly page views, CC0: Wikimedia Analytics REST API (`pageviews/aggregate`), https://wikimedia.org/api/rest_v1/

## Datasets added for the v2 experiments

```bash
python -m cwp.data download   --dataset azure_vm_2017     # the VM table only, about 160 MB
python -m cwp.data preprocess --dataset azure_vm_2017     # -> processed/azure_vm_2017_10m.csv, _30m, _60m
python -m cwp.data download   --dataset wikipedia         # six small JSON files
python -m cwp.data preprocess --dataset wikipedia         # -> processed/wikipedia_60m.csv
```

**Azure VM 2017.** 30 days of Azure VM activity. The series is the number of VMs *created* per
interval, from the VM table's "timestamp VM created" column. About 2 million VMs are in the
table; those with creation time 0 were already running when the trace began, so they are not
arrivals and are left out. The paper reports this dataset at 10, 30 and 60 minutes but does not
say how it derived the arrival rate, so its Table 3 values are a rough reference only. Only the
VM table is downloaded; the CPU readings (78.5 GB) are not needed.

**Wikipedia.** Hourly page views of English Wikipedia by human users (all access methods),
1 January to 30 June 2019 (181 days, 4,344 hours). This is a **substitute**: the paper used
the WikiBench request traces at 10 and 30 minutes, and that site is no longer online. The
public page-view data is hourly, so this series has no counterpart in the paper's Table 3. It
is here because it is strongly seasonal. "Jobs" for this series means page views.
