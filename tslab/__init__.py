"""
tslab: the small helper package used by the notebooks in this lab.

Why a package at all?
The notebooks are meant to teach ideas. Anything that is
repeated in several notebooks (loading the data,
building features, walk-forward validation, metrics) lives here so that each
notebook can stay short. Every function has a docstring and is written to be
read (for educational purposes). Open the files and read them; that is an
important part of the lab.

    config.py       where the data lives and which tickers we use
    data.py         downloading, saving and loading the market data
    features.py     turning prices into predictors (without peeking at the future)
    validation.py   walk-forward splits, the leakage check, model evaluation
    metrics.py      error measures, significance test, and a small backtest
    plots.py        a few plotting helpers
    nnutils.py      windowing, scaling, and Keras model builders
"""

__all__ = ["config", "data", "features", "validation", "metrics", "plots"]
