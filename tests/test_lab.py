"""
Automated checks for the helper package. Run from the project root with:

    python -m pytest tests

These tests use a small artificial price table, so they need no data files
and no internet connection.
"""

import numpy as np
import pandas as pd
import pytest

from tslab import features, metrics, validation


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(1)
    index = pd.bdate_range("2018-01-01", periods=600)
    walk = lambda scale, start: start * np.exp(np.cumsum(scale * rng.standard_normal(len(index))))
    close = walk(0.01, 50)
    frame = pd.DataFrame(
        {
            "Open": close * np.exp(0.002 * rng.standard_normal(len(index))),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1_000_000, 2_000_000, len(index)),
            "SP500": walk(0.008, 2000),
            "TECH": walk(0.009, 60),
            "VIX": walk(0.03, 18),
            "US10Y": 2 + np.cumsum(0.01 * rng.standard_normal(len(index))),
            "USD": walk(0.004, 95),
            "BRENT": walk(0.02, 60),
            "GOLD": walk(0.009, 1200),
        },
        index=index,
    )
    return frame


def test_features_use_only_the_past(market):
    assert features.check_no_lookahead(market, n_checks=6)


def test_leak_detector_catches_a_centred_window(market):
    def leaky(frame):
        out = features.build_features(frame)
        out["centred_mean"] = frame["Close"].rolling(5, center=True).mean()
        return out

    with pytest.raises(AssertionError):
        features.check_no_lookahead(market, feature_function=leaky, n_checks=6)


def test_dataset_target_is_the_next_day(market):
    data = features.make_dataset(market, horizon=1)
    row = data.index[10]
    position = market.index.get_loc(row)
    expected = np.log(market["Close"].iloc[position + 1] / market["Close"].iloc[position])
    assert data.loc[row, "target_return"] == pytest.approx(expected)


def test_walk_forward_never_trains_on_the_future():
    splits = list(validation.walk_forward_splits(500, initial_train=200, test_size=50, gap=4))
    assert len(splits) >= 5
    for train, test in splits:
        assert train.max() < test.min()
        assert test.min() - train.max() > 4  # the gap


def test_walk_forward_test_blocks_do_not_overlap():
    splits = list(validation.walk_forward_splits(500, initial_train=200, test_size=50))
    seen = np.concatenate([test for _, test in splits])
    assert len(seen) == len(set(seen))


def test_zero_forecast_has_zero_r2():
    actual = np.random.default_rng(0).normal(0, 0.01, 100)
    assert metrics.out_of_sample_r2(actual, np.zeros(100)) == pytest.approx(0.0)


def test_diebold_mariano_detects_a_clearly_better_forecast():
    rng = np.random.default_rng(0)
    actual = rng.normal(0, 1, 500)
    good = actual + rng.normal(0, 0.1, 500)
    bad = actual + rng.normal(0, 1.0, 500)
    statistic, p_value = metrics.diebold_mariano(actual, good, bad)
    assert statistic < 0 and p_value < 0.01


def test_walk_forward_scaler_is_fitted_on_training_rows_only():
    """
    The training rows are all zero and the test rows are all 1000. A scaler
    fitted on the training rows alone leaves the test value at 1000 (a zero
    spread is replaced by 1). A scaler that had also seen the test rows
    would shrink it to about 1.4.
    """
    index = pd.bdate_range("2020-01-01", periods=300)
    X = pd.DataFrame({"a": np.r_[np.zeros(200), np.full(100, 1000.0)]}, index=index)
    y = pd.Series(np.zeros(300), index=index)

    class Echo:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return X[:, 0]

    splits = list(validation.walk_forward_splits(300, initial_train=200, test_size=100))
    predictions = validation.walk_forward_predict(Echo, X, y, splits)
    assert predictions.dropna().min() == pytest.approx(1000.0)
