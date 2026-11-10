import numpy as np
import pytest

from cwp import metrics


def test_mape_is_the_average_percentage_error():
    assert metrics.mape([100, 50], [90, 60]) == pytest.approx(15.0)
    assert metrics.mape([200, 100, 50], [180, 110, 50]) == pytest.approx(20 / 3)


def test_mape_skips_zero_actuals_and_wape_does_not_blow_up():
    assert metrics.mape([0, 100], [5, 90]) == pytest.approx(10.0)
    assert metrics.wape([100, 1], [102, 3]) == pytest.approx(4 / 101 * 100)


def test_mase_below_one_beats_naive():
    train = np.array([10.0, 12, 9, 13, 11])
    scale = metrics.naive_scale(train)
    assert scale == pytest.approx(np.mean([2, 3, 4, 2]))
    assert metrics.mase([10, 12], [10.5, 11.5], scale) < 1


def test_provisioning_rates():
    r = metrics.provisioning([10, 10, 10, 10], [8.2, 10.0, 12.5, 9.1])
    assert r["under_rate"] == 25.0 and r["over_rate"] == 25.0   # ceil: 9, 10, 13, 10


def test_baselines_only_use_the_past():
    series = np.arange(50, dtype=float)
    b = metrics.baseline_predictions(series, 30, 40, season=10)
    assert list(b["naive"]) == list(range(29, 39))
    assert list(b["seasonal_naive"]) == list(range(20, 30))
    assert b["moving_avg_3"][0] == pytest.approx(28.0)


def test_diebold_mariano_prefers_the_better_forecast():
    rng = np.random.default_rng(1)
    y = rng.random(400) * 100
    good, bad = y + rng.normal(0, 1, 400), y + rng.normal(0, 5, 400)
    r = metrics.diebold_mariano(y, good, bad)
    assert r["stat"] < 0 and r["p"] < 0.001


def test_holm_adjustment():
    assert metrics.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
