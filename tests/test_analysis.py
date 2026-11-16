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
