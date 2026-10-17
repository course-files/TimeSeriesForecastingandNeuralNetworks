"""
Turning prices into predictors.

THE GOLDEN RULE
A feature in row t may only use information that was known at the close of
day t. Nothing from day t+1 or later. Breaking this rule is called
look-ahead bias (or data leakage). It produces models that look brilliant
in the notebook and lose money in the real world.

How the rows are laid out
    row t   features: built from data up to and including the close of day t
            target  : what happens AFTER day t (the return from t to t+h)

We forecast the RETURN, not the price level, and then convert the forecast
back to a closing price:

    predicted close(t+h) = close(t) * exp(predicted log return)

Notebook 2 explains why.
"""

import numpy as np
import pandas as pd

# Names of the predictor markets, as produced by tslab.data.load_market_data
LEVEL_PREDICTORS = ["SP500", "TECH", "GOLD", "BRENT", "USD"]  # prices: use log returns
RATE_PREDICTOR = "US10Y"                                      # a yield: use differences
FEAR_PREDICTOR = "VIX"                                        # a volatility level


# ---------------------------------------------------------------------------
# Individual indicators. Each one is a few lines so that you can read it.
# ---------------------------------------------------------------------------
def log_return(series, periods=1):
    """The natural-log return over `periods` days: log(P_t / P_{t-periods})."""
    return np.log(series).diff(periods)


def rsi(close, window=14):
    """
    Relative Strength Index, scaled to 0..1.
    Compares the size of recent gains with the size of recent losses.
    Near 1: a strong run of gains ("overbought"). Near 0: a strong run of losses.
    """
    change = close.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    return (1 - 1 / (1 + relative_strength))


def macd_histogram(close, fast=12, slow=26, signal=9):
    """
    MACD histogram, divided by the price so that it is comparable across years.
    The MACD line is the gap between a fast and a slow exponential moving
    average. The histogram is the gap between the MACD line and its own
    smoothed version. It measures the change in momentum.
    """
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return (macd_line - signal_line) / close


def bollinger(close, window=20, num_std=2.0):
    """
    Bollinger Bands. Returns two numbers:
      percent_b : where the price sits between the lower band (0) and upper band (1)
      width     : how wide the bands are, relative to the middle band (a volatility gauge)
    """
    middle = close.rolling(window).mean()
    spread = close.rolling(window).std()
    upper = middle + num_std * spread
    lower = middle - num_std * spread
    percent_b = (close - lower) / (upper - lower)
    width = (upper - lower) / middle
    return percent_b, width


def average_true_range(high, low, close, window=14):
    """
    Average True Range: the typical size of a day's price movement, including
    overnight gaps. Divided by the close so that it is a percentage.
    """
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(window).mean() / close


# ---------------------------------------------------------------------------
# The full feature table
# ---------------------------------------------------------------------------
def build_features(frame):
    """
    Build every predictor from the raw market table.

    `frame` is the output of tslab.data.load_market_data. The result has one
    row per trading day and uses only information up to that day.
    """
    close, open_, high, low, volume = (frame[c] for c in ["Close", "Open", "High", "Low", "Volume"])
    daily_return = log_return(close)
    out = pd.DataFrame(index=frame.index)

    # 1. Returns: what the price has done recently ---------------------------
    for lag in range(1, 6):
        out[f"ret_lag{lag}"] = daily_return.shift(lag - 1)  # lag1 = today's return
    out["ret_5d"] = log_return(close, 5)
    out["ret_10d"] = log_return(close, 10)
    out["ret_21d"] = log_return(close, 21)
    out["ret_63d"] = log_return(close, 63)

    # 2. Volatility: how violently the price has been moving -----------------
    out["vol_5d"] = daily_return.rolling(5).std()
    out["vol_21d"] = daily_return.rolling(21).std()
    out["vol_63d"] = daily_return.rolling(63).std()
    out["vol_ratio"] = out["vol_5d"] / out["vol_21d"]
    out["range_hl"] = np.log(high / low)               # today's high to low range
    out["atr_14"] = average_true_range(high, low, close)

    # 3. Technical indicators -------------------------------------------------
    out["rsi_14"] = rsi(close)
    out["macd_hist"] = macd_histogram(close)
    out["bb_percent_b"], out["bb_width"] = bollinger(close)
    for window in (20, 50, 200):
        out[f"dist_sma_{window}"] = np.log(close / close.rolling(window).mean())
    out["gap_overnight"] = np.log(open_ / close.shift(1))   # yesterday's close to today's open
    out["move_intraday"] = np.log(close / open_)            # today's open to today's close

    # 4. Volume ---------------------------------------------------------------
    out["volume_rel_20d"] = np.log(volume / volume.rolling(20).mean())

    # 5. Other markets (the multiple predictors) -----------------------------
    for name in LEVEL_PREDICTORS:
        out[f"{name}_ret_1d"] = log_return(frame[name], 1)
        out[f"{name}_ret_5d"] = log_return(frame[name], 5)
    out["VIX_level"] = np.log(frame[FEAR_PREDICTOR])
    out["VIX_change_1d"] = log_return(frame[FEAR_PREDICTOR], 1)
    out["US10Y_level"] = frame[RATE_PREDICTOR]
    out["US10Y_change_5d"] = frame[RATE_PREDICTOR].diff(5)
    out["relative_to_SP500_5d"] = out["ret_5d"] - out["SP500_ret_5d"]

    # 6. Calendar -------------------------------------------------------------
    weekday = frame.index.dayofweek
    for day, label in enumerate(["mon", "tue", "wed", "thu", "fri"]):
        out[f"is_{label}"] = (weekday == day).astype(float)
    month = frame.index.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)

    return out


# Features grouped by idea. The groups are used in notebook 4 to ask a simple
# question: which GROUP of predictors actually adds forecasting power?
FEATURE_GROUPS = {
    "returns": ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5",
                "ret_5d", "ret_10d", "ret_21d", "ret_63d"],
    "volatility": ["vol_5d", "vol_21d", "vol_63d", "vol_ratio", "range_hl", "atr_14"],
    "technical": ["rsi_14", "macd_hist", "bb_percent_b", "bb_width",
                  "dist_sma_20", "dist_sma_50", "dist_sma_200",
                  "gap_overnight", "move_intraday"],
    "volume": ["volume_rel_20d"],
    "other_markets": ["SP500_ret_1d", "SP500_ret_5d", "TECH_ret_1d", "TECH_ret_5d",
                      "GOLD_ret_1d", "GOLD_ret_5d", "BRENT_ret_1d", "BRENT_ret_5d",
                      "USD_ret_1d", "USD_ret_5d", "VIX_level", "VIX_change_1d",
                      "US10Y_level", "US10Y_change_5d", "relative_to_SP500_5d"],
    "calendar": ["is_mon", "is_tue", "is_wed", "is_thu", "is_fri", "month_sin", "month_cos"],
}

ALL_FEATURES = [name for group in FEATURE_GROUPS.values() for name in group]


def make_dataset(frame, horizon=1):
    """
    Build the modelling table: features at day t, targets for day t+horizon.

    Returns a DataFrame with
        all feature columns
        close            the closing price on day t (known at prediction time)
        target_return    log(close[t+horizon] / close[t])    <- what we model
        target_close     close[t+horizon]                    <- what we report

    Rows whose features are not yet defined (the first ~200 days, because of
    the 200-day average) are dropped. The LAST `horizon` rows have no target
    yet; they are kept so that you can make a genuine forecast, and
    `target_return` is NaN for them. Use  dataset.dropna(subset=["target_return"])
    for training.
    """
    features = build_features(frame)
    data = features.copy()
    data["close"] = frame["Close"]
    data["target_close"] = frame["Close"].shift(-horizon)
    data["target_return"] = np.log(frame["Close"].shift(-horizon) / frame["Close"])
    data = data.replace([np.inf, -np.inf], np.nan)
    feature_ready = data[ALL_FEATURES].notna().all(axis=1)
    return data[feature_ready]


def check_no_lookahead(frame, feature_function=None, n_checks=5, seed=0):
    """
    The truncation test for look-ahead bias.

    Idea: if a feature really uses only the past, then computing it on a table
    that STOPS at day t must give exactly the same value for day t as
    computing it on the full table. We cut the data at several random dates
    and compare. Any difference means a feature is peeking at the future.

    feature_function  a function that takes the market table and returns a
                      table of features (default: build_features). Pass your own
                      function to test a new feature before you trust it.

    Returns True if every feature passes. Raises AssertionError otherwise.
    """
    feature_function = feature_function or build_features
    rng = np.random.default_rng(seed)
    full = feature_function(frame)
    valid_rows = np.where(full.notna().all(axis=1))[0]
    candidates = valid_rows[valid_rows < len(frame) - 10]
    cut_positions = rng.choice(candidates, size=n_checks, replace=False)
    for position in cut_positions:
        truncated = feature_function(frame.iloc[: position + 1])
        differences = (full.iloc[position] - truncated.iloc[-1]).abs()
        bad = differences[~(differences <= 1e-9)].index.tolist()
        assert not bad, (
            f"Look-ahead bias detected on {frame.index[position].date()}: "
            f"these features changed when the future was removed: {bad}"
        )
    return True
