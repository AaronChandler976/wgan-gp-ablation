import itertools

import numpy as np
import pandas as pd
import pytest

from cwp import analysis
from cwp.evaluate import FACTORS


def design(effects: dict, data_id="d-1m", seeds=(1, 2, 3, 4, 5), noise=0.01, rng=None):
    """Runs of the full 2x2x2 design with known log effects of each part."""
    rng = rng or np.random.default_rng(0)
    rows = []
    for seed in seeds:
        for arch, critic, opt in itertools.product(*FACTORS.values()):
            on = {"arch": arch == "transformer", "critic": critic == "wgan-gp", "optimizer": opt == "madgrad"}
            log = np.log(10.0) + sum(effects.get(k, 0.0) * v for k, v in on.items()) + rng.normal(0, noise)
            rows.append({"data_id": data_id, "arch": arch, "critic": critic, "optimizer": opt,
                         "seed": seed, "mape": float(np.exp(log))})
    return pd.DataFrame(rows)


def test_contrasts_recover_known_effects():
    c = analysis.contrasts(design({"arch": -0.10, "critic": 0.05}))
    mean = c.groupby("effect")["value"].mean()
    assert mean["arch"] == pytest.approx(-0.10, abs=0.01)
    assert mean["critic"] == pytest.approx(0.05, abs=0.01)
    assert mean["optimizer"] == pytest.approx(0.0, abs=0.01)
    assert mean["arch x critic"] == pytest.approx(0.0, abs=0.01)
    assert len(c) == 7 * 5


def test_contrasts_skip_incomplete_and_diverged_seeds():
    df = design({})
    df.loc[(df["seed"] == 1) & (df["arch"] == "lstm"), "mape"] = np.inf
    df = df[~((df["seed"] == 2) & (df["critic"] == "none") & (df["optimizer"] == "adam") & (df["arch"] == "lstm"))]
    assert sorted(analysis.contrasts(df)["seed"].unique()) == [3, 4, 5]


def test_effect_table_holm_is_never_below_raw_p():
    df = pd.concat([design({"arch": -0.10}, "a-1m"), design({"critic": 0.08}, "b-1m")])
    t = analysis.effect_table(analysis.contrasts(df))
    assert len(t) == 14
    assert (t["p_holm_dataset"] >= t["p"] - 1e-12).all()
    main = t[t["kind"] == "main effect"]
    assert (main["p_holm_main"] >= main["p"] - 1e-12).all()
    assert t[t["kind"] == "interaction"]["p_holm_main"].isna().all()
    assert t["p_wilcoxon"].isna().all()            # 5 seeds: not reported
    row = t[(t["data_id"] == "a-1m") & (t["effect"] == "arch")].iloc[0]
    assert row["ci_low_pct"] < row["change_pct"] < row["ci_high_pct"] < 0
    assert row["same_sign_seeds"] == "5/5"


def test_wilcoxon_reported_from_six_seeds():
    t = analysis.effect_table(analysis.contrasts(design({"arch": -0.10}, seeds=range(1, 9))))
    assert t["p_wilcoxon"].notna().all()


def test_pooled_detects_effects_that_differ_between_datasets():
    same = pd.concat([design({"arch": -0.10}, "a-1m"), design({"arch": -0.10}, "b-1m")])
    t = analysis.pooled_table(analysis.contrasts(same)).set_index("effect")
    assert t.loc["arch", "pooled_pct"] == pytest.approx((np.exp(-0.10) - 1) * 100, abs=1.0)
    assert t.loc["arch", "p_pooled"] < 0.001
    assert t.loc["arch", "p_differs"] > 0.01
    differ = pd.concat([design({"arch": -0.10}, "a-1m"), design({"arch": 0.10}, "b-1m")])
    t = analysis.pooled_table(analysis.contrasts(differ)).set_index("effect")
    assert t.loc["arch", "p_differs"] < 0.001
    assert t.loc["arch", "ci_low_pct"] < 0 < t.loc["arch", "ci_high_pct"]
    assert t.loc["arch", "between_dataset_sd_pct"] > 5


def test_headroom_curve_and_target():
    y = np.array([100.0, 100.0, 100.0, 100.0])
    p = np.array([100.0, 95.0, 90.0, 110.0])
    curve = analysis.headroom_curve(y, p)
    first = curve.iloc[0]
    assert first["headroom_pct"] == 0 and first["under_rate"] == 50.0
    assert first["missing_vm_pct"] == pytest.approx(15 / 400 * 100)
    assert first["idle_vm_pct"] == pytest.approx(10 / 400 * 100)
    assert (np.diff(curve["under_rate"]) <= 0).all() and (np.diff(curve["idle_vm_pct"]) >= 0).all()
    h = analysis.headroom_for(curve, 0.0)
    assert h["headroom_pct"] == 11.0               # 90 * 1.11 = 99.9 -> 100 VMs
    assert np.isnan(analysis.headroom_for(curve.iloc[:3], 0.0)["headroom_pct"])


def test_screen_table_applies_the_noise_rule():
    cells = ["c1", "c2", "c3", "c4"]

    def trials(screen, variant, values):
        return [{"screen": screen, "cell": c, "variant": variant, "seed": seed, "val_mape": v + d}
                for c, v in zip(cells, values) for seed, d in ((0, -0.1), (100, 0.1))]
    df = pd.DataFrame(trials("s1", "ref", [10, 10, 10, 10])
                      + trials("s1", "one_cell", [5, 10.1, 10.1, 10.1])      # big gain in one cell only
                      + trials("s2", "broad", [9.7, 9.7, 9.7, 10.1])         # 3 of 4 cells, about -2%
                      + trials("s2", "tiny", [9.95, 9.95, 9.95, 9.95]))      # all cells, under 1%
    t = analysis.screen_table(df, ("s1", "ref")).set_index("variant")
    assert t.index[0] == "one_cell"                                          # best average comes first
    assert not t.loc["one_cell", "beats_reference"] and t.loc["one_cell", "cells_better"] == "1/4"
    assert t.loc["broad", "beats_reference"]
    assert not t.loc["tiny", "beats_reference"]
    assert t.loc["ref", "mean_change_pct"] == pytest.approx(0.0)
    assert t.loc["broad", "c1"] == pytest.approx(9.7)


def test_workload_table_relates_effects_to_properties():
    effects = {"a": -0.10, "b": -0.05, "c": 0.0, "d": 0.05}
    runs = pd.concat([design({"arch": e}, f"{name}-10m") for name, e in effects.items()])
    stats_table = pd.DataFrame({"dataset": list(effects), "interval_min": 10,
                                **{p: [1.0, 2.0, 3.0, 4.0] for p in analysis.WORKLOAD_PROPERTIES}})
    table, corr = analysis.workload_table(analysis.effect_table(analysis.contrasts(runs)), stats_table)
    assert len(table) == 4 and {"arch_effect_pct", "arch_p_holm", "seasonal_strength"} <= set(table.columns)
    assert corr.set_index("property").loc["seasonal_strength", "rho_arch"] == pytest.approx(1.0)


def test_load_runs_joins_sources_and_refuses_overlap(tmp_path, monkeypatch):
    import json

    def write(folder, seed, critic="none"):
        folder.mkdir(parents=True, exist_ok=True)
        name = f"d-1m__lstm-{'critic' if critic != 'none' else 'nocritic'}-adam__s{seed}"
        (folder / f"{name}.json").write_text(json.dumps({
            "run_id": name, "arch": "lstm", "critic": critic, "optimizer": "adam", "seed": seed,
            "metrics": {"test": {"mape": 1.0 + seed}}}))
    monkeypatch.setattr(analysis.config, "path", lambda *parts: tmp_path.joinpath(*parts))
    write(tmp_path / "results", 1)
    write(tmp_path / "results" / "v2" / "more", 6)
    write(tmp_path / "results" / "v2" / "more", 6, critic="wgan-gp")
    df = analysis.load_runs("reported+more")
    assert sorted(df["seed"]) == [1, 6, 6] and set(df["source"]) == {"reported", "more"}
    assert len(analysis.load_runs("reported", nocritic_from="more")) == 2
    rest = analysis.load_runs("more+rest_of_reported")           # `reported` has no other cell
    assert len(rest) == 2 and set(rest["source"]) == {"more"}
    rest = analysis.load_runs("reported+rest_of_more")           # only the cell `reported` lacks
    assert sorted(rest["seed"]) == [1, 6] and rest.set_index("source").loc["more", "critic"] == "wgan-gp"
    write(tmp_path / "results" / "v2" / "more", 1)
    with pytest.raises(SystemExit):
        analysis.load_runs("reported+more")


def test_versus_table_pairs_by_seed_and_counts_diverged_runs():
    reference = design({}, seeds=(1, 2, 3, 4, 5)).assign(source="reported")
    runs = design({}, seeds=(1, 2, 3)).assign(source="exp")
    for df in (reference, runs):
        df["config_id"] = df["arch"] + "-" + df["critic"] + "-" + df["optimizer"]
    runs["mape"] = reference.set_index(["config_id", "seed"]).loc[
        list(zip(runs["config_id"], runs["seed"])), "mape"].to_numpy() * 1.10
    runs.loc[(runs["seed"] == 1) & (runs["config_id"] == "lstm-none-adam"), "mape"] = np.nan
    t = analysis.versus_table(runs, reference).set_index("config_id")
    assert len(t) == 8 and (t["change_pct"].round(6) == 10.0).all()
    assert t.loc["lstm-none-adam", "seeds"] == 2 and t.loc["lstm-none-adam", "not_finite"] == 1
    assert t.loc["transformer-none-adam", "seeds"] == 3 and t.loc["transformer-none-adam", "better_in"] == "0/3"
