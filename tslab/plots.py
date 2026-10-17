"""A few plotting helpers, so that the notebooks stay readable."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_walk_forward_layout(splits, n_samples, max_folds=12):
    """Draw which rows are used for training (blue) and testing (orange) in each fold."""
    shown = splits[:max_folds]
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(shown) + 1.2))
    for row, (train_idx, test_idx) in enumerate(shown):
        ax.barh(row, len(train_idx), left=train_idx[0], color="tab:blue", height=0.6)
        ax.barh(row, len(test_idx), left=test_idx[0], color="tab:orange", height=0.6)
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels([f"Fold {i + 1}" for i in range(len(shown))])
    ax.invert_yaxis()
    ax.set_xlim(0, n_samples)
    ax.set_xlabel("Row number (time runs to the right)")
    ax.set_title("Walk-forward validation: blue = training rows, orange = test rows")
    plt.tight_layout()
    plt.show()


def plot_price_forecasts(dates, actual, forecasts, title, last_n=None):
    """
    Plot the actual closing price with one or more forecasts on top.
    `forecasts` is a dict of name -> array aligned with `dates`.
    Use last_n to zoom in on the final n days; zooming is how you see
    whether a forecast is merely copying yesterday's price.
    """
    dates = pd.Index(dates)
    actual = np.asarray(actual)
    if last_n:
        dates, actual = dates[-last_n:], actual[-last_n:]
    plt.figure(figsize=(12, 5))
    plt.plot(dates, actual, color="black", linewidth=1.8, label="Actual close")
    for name, values in forecasts.items():
        values = np.asarray(values)
        values = values[-last_n:] if last_n else values
        plt.plot(dates, values, linewidth=1.2, alpha=0.9, label=name)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Closing price")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_learning_curves(history, title="Learning curves", reference=None, reference_label="Validation loss of predicting zero"):
    """Training versus validation loss from a Keras History object (or a dict)."""
    values = history.history if hasattr(history, "history") else history
    plt.figure(figsize=(8, 4.5))
    plt.plot(values["loss"], label="Training loss")
    if "val_loss" in values:
        plt.plot(values["val_loss"], label="Validation loss")
        best = int(np.argmin(values["val_loss"]))
        plt.axvline(best, color="grey", linestyle="--", label=f"Best epoch ({best + 1})")
    if reference is not None:
        plt.axhline(reference, color="red", linestyle=":", label=reference_label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (mean squared error)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_acf_pacf(series, lags=30, title=""):
    """ACF and PACF side by side."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0], zero=False)
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], zero=False, method="ywm")
    axes[0].set_title(f"Autocorrelation (ACF) {title}")
    axes[1].set_title(f"Partial autocorrelation (PACF) {title}")
    plt.tight_layout()
    plt.show()
