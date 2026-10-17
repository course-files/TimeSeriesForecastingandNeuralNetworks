"""
Honest evaluation of a forecasting model.

An ordinary train_test_split with shuffling is WRONG for time series,
because it trains on the future and tests on the past. Even a single
split is fragile: a model can look good simply because the test period
happened to be calm. Walk-forward validation fixes both problems by
repeating the experiment through time:

    fold 1:  train on days   0 .. 999       test on days 1000 .. 1062
    fold 2:  train on days   0 .. 1062      test on days 1063 .. 1125
    fold 3:  train on days   0 .. 1125      test on days 1126 .. 1188
    ...

Each model is always tested on days it has never seen, and always trained
only on days that came BEFORE the test period, exactly as it would be used
in practice.
"""

import numpy as np
import pandas as pd


def walk_forward_splits(n_samples, initial_train, test_size, gap=0, window=None):
    """
    Generate (train_positions, test_positions) pairs.

    n_samples      number of rows in the modelling table
    initial_train  size of the first training block
    test_size      number of rows in each test block (63 = one quarter)
    gap            rows to skip between train and test. Use horizon - 1.
                   When the target looks `horizon` days ahead, the last
                   few training labels would otherwise overlap the test period.
    window         None for an EXPANDING window (use all history), or an integer
                   for a ROLLING window (use only the last `window` rows).
    """
    train_end = initial_train  # exclusive
    while train_end + gap < n_samples:
        test_start = train_end + gap
        test_end = min(test_start + test_size, n_samples)
        train_start = 0 if window is None else max(0, train_end - window)
        yield np.arange(train_start, train_end), np.arange(test_start, test_end)
        train_end += test_size


def walk_forward_predict(model_factory, X, y, splits, standardize=True):
    """
    Fit a fresh model on every training block and predict the matching test block.

    model_factory  a function with no arguments that returns a NEW untrained
                   model with .fit and .predict (any scikit-learn style model)
    X, y           a DataFrame and a Series with the same index
    splits         a list produced by walk_forward_splits
    standardize    if True, a StandardScaler is fitted on the TRAINING block
                   only and applied to the test block. Fitting the scaler on
                   all the data would leak information about the future
                   mean and spread into the training step.

    Returns a Series of out-of-sample predictions. Rows that were never in a
    test block are NaN.
    """
    from sklearn.preprocessing import StandardScaler

    predictions = pd.Series(np.nan, index=X.index, name="prediction")
    for train_idx, test_idx in splits:
        X_train, X_test = X.iloc[train_idx].values, X.iloc[test_idx].values
        y_train = y.iloc[train_idx].values
        if standardize:
            scaler = StandardScaler().fit(X_train)
            X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
        model = model_factory()
        model.fit(X_train, y_train)
        predictions.iloc[test_idx] = model.predict(X_test)
    return predictions


def describe_splits(splits, index):
    """A small table of the folds, useful for checking that nothing overlaps."""
    rows = []
    for number, (train_idx, test_idx) in enumerate(splits, start=1):
        rows.append(
            {
                "fold": number,
                "train_rows": len(train_idx),
                "train_from": index[train_idx[0]].date(),
                "train_to": index[train_idx[-1]].date(),
                "test_rows": len(test_idx),
                "test_from": index[test_idx[0]].date(),
                "test_to": index[test_idx[-1]].date(),
            }
        )
    return pd.DataFrame(rows).set_index("fold")
