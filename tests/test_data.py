import numpy as np
import pytest

from cwp import data


def test_split_is_chronological_60_20_20():
    s = data.compute_split(1000)
    assert (s.train_end, s.val_end, s.n) == (600, 800, 1000)


def test_windows_never_see_the_target_or_the_future():
    series = np.arange(100, dtype=float)
    X, y, idx = data.make_windows(series, window=5, start=60, end=80)
    assert list(idx) == list(range(60, 80))            # every point of the part is a target
    for row, target, t in zip(X, y, idx):
        assert target == series[t]
        assert row.max() < target                      # history ends just before the target
        assert list(row) == list(series[t - 5:t])


def test_same_targets_for_any_window_size():
    series = np.random.default_rng(0).random(300)
    _, _, a = data.make_windows(series, 8, 240, 300)
    _, _, b = data.make_windows(series, 96, 240, 300)
    assert np.array_equal(a, b)


def test_scaler_uses_training_range_only():
    sc = data.MinMaxScaler.fit(np.array([10.0, 20.0, 30.0]))
    assert sc.transform(10.0) == -1 and sc.transform(30.0) == 1
    assert sc.transform(40.0) > 1                      # later, larger values are not clipped
    assert sc.inverse(sc.transform(17.5)) == pytest.approx(17.5)


def test_aggregate_sums_blocks_and_drops_the_tail():
    assert list(data.aggregate(np.arange(1, 8), 3)) == [6, 15]


def test_synthetic_and_split_files(repo):
    series = data.load_series("synthetic_seasonal", 10)
    assert len(series) == 14 * 144 and series.min() > 0
    s = data.get_split("synthetic_seasonal", 10)
    assert 0 < s.train_end < s.val_end < s.n == len(series)
    assert (repo / "data" / "splits" / "synthetic_seasonal-10m.json").exists()


def test_google_and_azure_preprocessing_on_fake_raw_files(repo):
    import pandas as pd
    raw = repo / "data" / "raw" / "google"
    raw.mkdir(parents=True)
    us = data.GOOGLE_START_US
    events = pd.DataFrame({0: [us - 5, us, us + 59_999_999, us + 60_000_000, us + 60_000_001],
                           1: 0, 2: [1, 2, 3, 4, 5], 3: [0, 0, 0, 0, 1], 4: "u", 5: 1, 6: "j", 7: "l"})
    events.to_csv(raw / "part-00000-of-00500.csv.gz", header=False, index=False, compression="gzip")
    minutes = data.google_minutes(raw)
    assert minutes[0] == 2 and minutes[1] == 1 and minutes.sum() == 3   # SUBMIT events only

    raw = repo / "data" / "raw" / "azure_functions"
    raw.mkdir(parents=True)
    cols = {"HashOwner": ["a", "b"], "HashApp": ["a", "b"], "HashFunction": ["f", "g"], "Trigger": ["http", "timer"]}
    cols.update({str(m): [1, 2] for m in range(1, 1441)})
    for day in (1, 2):
        pd.DataFrame(cols).to_csv(raw / f"invocations_per_function_md.anon.d{day:02d}.csv", index=False)
    minutes = data.azure_minutes(raw)
    assert len(minutes) == 2880 and set(minutes) == {3}
    assert list(data.aggregate(minutes, 5)[:2]) == [15, 15]


def test_azure_vm_2017_counts_vm_creations_and_skips_vms_already_running(tmp_path):
    import gzip
    from cwp import data
    rows = ["a,s,d,0,500,1,1,1,x,1,1",          # creation time 0: already running, not an arrival
            "b,s,d,30,500,1,1,1,x,1,1",         # minute 0
            "c,s,d,59,500,1,1,1,x,1,1",         # minute 0
            "d,s,d,60,500,1,1,1,x,1,1",         # minute 1
            f"e,s,d,{data.AZURE_VM_MINUTES * 60},9,1,1,1,x,1,1"]   # after the 30 days
    with gzip.open(tmp_path / "vmtable.csv.gz", "wt") as f:
        f.write("\n".join(rows) + "\n")
    minutes = data.azure_vm_2017_minutes(tmp_path)
    assert len(minutes) == data.AZURE_VM_MINUTES and minutes.sum() == 3
    assert minutes[0] == 2 and minutes[1] == 1


def test_wikipedia_hours_are_read_in_order_and_gaps_are_an_error(tmp_path):
    import json
    import pytest
    from cwp import data
    def items(hours):
        return {"items": [{"timestamp": f"20190101{h:02d}", "views": 100 + h} for h in hours]}
    (tmp_path / "pageviews_201901.json").write_text(json.dumps(items(range(6))))
    hourly = data.wikipedia_minutes(tmp_path)
    assert hourly.tolist() == [100, 101, 102, 103, 104, 105]
    assert data.aggregate(hourly, 2).tolist() == [201, 205, 209]
    (tmp_path / "pageviews_201901.json").write_text(json.dumps(items([0, 1, 3])))
    with pytest.raises(ValueError):
        data.wikipedia_minutes(tmp_path)
