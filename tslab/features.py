"""
Turning prices into predictors (also called "features").

WHAT THIS FILE DOES
A model cannot learn much from a raw list of prices. So we compute
useful numbers from the prices, such as "how much did the stock move this
week?" or "how jumpy has it been lately?". Each of these numbers is a
FEATURE. The model uses the features to make its forecast.

THE GOLDEN RULE: NO PEEKING AT THE FUTURE
Picture yourself on Monday evening after the market has closed. You can see
every price up to and including Monday's close. You cannot see Tuesday.

So each feature for Monday may use only Monday's data and earlier data.
If a feature accidentally uses Tuesday's price, the model is "cheating".
This mistake is called LOOK-AHEAD BIAS (or DATA LEAKAGE). The model will look
excellent in the notebook and then fail with real money, because in real
life you never get to see tomorrow's price in advance.

HOW EACH ROW OF THE TABLE IS ORGANIZED
Every row is one trading day. Call it "day t".
    features : computed from prices up to the close of day t   (the past)
    target   : what happens AFTER day t                        (the future,
               which is the thing we are trying to predict)

WE PREDICT THE RETURN, NOT THE PRICE
We do not predict "the price will be $152". We predict "the price will go
up by 1.3%" (a return). After that we turn the return back into a price:

    predicted close tomorrow = today's close * exp(predicted log return)

Notebook 2 explains why returns are easier to model than prices.
"""

import numpy as np
import pandas as pd

# The other markets we use as extra predictors. The names match the columns
# produced by tslab.data.load_market_data (see tslab/config.py for details).
LEVEL_PREDICTORS = ["SP500", "TECH", "GOLD", "BRENT", "USD"]  # ordinary prices, so we use their returns
RATE_PREDICTOR = "US10Y"                                      # an interest rate, so we use "change in the rate"
FEAR_PREDICTOR = "VIX"                                        # the "fear index": how nervous the market is


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# Each function below calculates one well-known trading indicator. They are
# kept short on purpose so that you can read them line by line.
#
# Two pandas tools appear many times. It helps to understand them first:
#   .rolling(n)  looks at the last n days only. Example: .rolling(20).mean()
#                is the average of the last 20 days. That average is
#                recalculated each day as the 20-day window moves forward.
#   .ewm(...)    is like .rolling(), but recent days count MORE than older
#                days. It is an "exponentially weighted" average.
# ---------------------------------------------------------------------------
def log_return(series, periods=1):
    """
    How much the price changed over the last `periods` days, as a log return.

    Formula: log(price today / price `periods` days ago)
    Example: the price goes from 100 to 105, so log(105 / 100) = 0.0488.
    That is roughly +4.9%. For small moves a log return is almost the same
    as the percentage change.
    """
    return np.log(series).diff(periods)


def rsi(close, window=14):
    """
    Relative Strength Index (RSI): a number between 0 and 1 that tells you
    whether recent days have mostly been UP days or DOWN days.

        close to 1  -> mostly gains lately   (traders call this "overbought")
        close to 0  -> mostly losses lately  (traders call this "oversold")
        around 0.5  -> gains and losses roughly balanced

    How it works:
      1. Work out each day's price change.
      2. Split the changes into gains (up days) and losses (down days).
      3. Average the gains and average the losses over about 14 days.
      4. Compare the two averages and squash the answer into the range 0 to 1.
    """
    change = close.diff()                    # today's price minus yesterday's
    gains = change.clip(lower=0)             # keep the up moves, set the down moves to 0
    losses = -change.clip(upper=0)           # keep the down moves (as positive numbers), set the up moves to 0
    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)   # avoid dividing by zero
    return (1 - 1 / (1 + relative_strength))                     # convert to the 0..1 scale


def macd_histogram(close, fast=12, slow=26, signal=9):
    """
    Moving Average Convergence Divergence (MACD) histogram: tells you whether
    the price's momentum is speeding up or slowing down.

    How it works:
      1. Compute a FAST average of the price (about the last 12 days) and a
         SLOW average (about the last 26 days).
      2. MACD line = fast average minus slow average.
         It is positive when recent prices are above the longer-term trend.
      3. Signal line = a smoothed version of the MACD line.
      4. Histogram = MACD line minus signal line.
         Positive -> upward momentum is getting stronger.
         Negative -> upward momentum is getting weaker.

    We divide by the price at the end so that the number has the same meaning
    whether the stock costs $20 or $2,000.
    """
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return (macd_line - signal_line) / close


def bollinger(close, window=20, num_std=2.0):
    """
    Bollinger Bands: a "normal range" drawn around the recent average price.

    The middle line is the 20-day average price. The upper and lower bands
    sit 2 standard deviations above and below it. Most of the time the price
    stays between the two bands.

    Returns two numbers:
      percent_b : where today's price sits inside the bands.
                  0 = at the lower band, 0.5 = in the middle, 1 = at the upper band.
                  (It can go below 0 or above 1 if the price breaks out of the bands.)
      width     : how far apart the bands are. Wide bands mean the price has
                  been jumping around a lot. Narrow bands mean it has been calm.
    """
    middle = close.rolling(window).mean()     # 20-day average price
    spread = close.rolling(window).std()      # how much the price varied over those 20 days
    upper = middle + num_std * spread
    lower = middle - num_std * spread
    percent_b = (close - lower) / (upper - lower)
    width = (upper - lower) / middle
    return percent_b, width


def average_true_range(high, low, close, window=14):
    """
    Average True Range (ATR): how big a typical day's price swing is.

    For each day we take the LARGEST of these three numbers:
      - today's high minus today's low
      - the distance from yesterday's close to today's high
      - the distance from yesterday's close to today's low
    The last two capture overnight jumps (for example after news while the
    market was closed). We then average this over 14 days and divide by the
    price, so the answer reads like a percentage. 0.02 means a typical day
    moves about 2%.
    """
    previous_close = close.shift(1)           # yesterday's close, placed on today's row
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)                             # the largest of the three, for each day
    return true_range.rolling(window).mean() / close


# ---------------------------------------------------------------------------
# BUILDING THE FULL FEATURE TABLE
# ---------------------------------------------------------------------------
def build_features(frame):
    """
    Build every feature from the raw market data.

    Input:  `frame`, the table from tslab.data.load_market_data. It has one
            row per trading day with the stock's Open, High, Low, Close and
            Volume, plus a column for each of the other markets.
    Output: a new table with the SAME rows (one per trading day) and one
            column per feature. Each value uses only information available on
            or before that day.

    A NOTE ON .shift()
    .shift(1) moves a column DOWN by one row. After the shift, each row holds
    YESTERDAY's value. Example: close.shift(1) on Tuesday's row gives Monday's
    close. A POSITIVE shift looks at the past, which is safe. A NEGATIVE
    shift, such as .shift(-1), looks at the future, and must NEVER be used to
    build a feature.

    WHY THE FIRST ROWS ARE EMPTY (NaN)
    A 200-day average cannot be calculated until 200 days of data exist. So
    the first rows of the table contain NaN ("not a number"). make_dataset()
    removes them later.
    """
    # Pull out the five basic price columns, so that we can write `close`
    # instead of frame["Close"] everywhere.
    close, open_, high, low, volume = (frame[c] for c in ["Close", "Open", "High", "Low", "Volume"])

    # Each day's return (today's close compared with yesterday's close). Many
    # of the features below start from this.
    daily_return = log_return(close)

    # An empty table with one row per trading day. We add the features to it
    # one column at a time.
    out = pd.DataFrame(index=frame.index)

    # ----------------------------------------------------------------------
    # GROUP 1: RETURNS - "What has the price done recently?"
    # ----------------------------------------------------------------------
    # The returns of the last 5 individual days:
    #   ret_lag1 = today's return            (no shift)
    #   ret_lag2 = yesterday's return        (shifted down 1 row)
    #   ret_lag3 = the day before that       (shifted down 2 rows), and so on.
    for lag in range(1, 6):
        out[f"ret_lag{lag}"] = daily_return.shift(lag - 1)

    # Total return over longer periods. There are about 21 trading days in a
    # month and about 63 in three months.
    out["ret_5d"] = log_return(close, 5)      # the past one week
    out["ret_10d"] = log_return(close, 10)    # the past two weeks
    out["ret_21d"] = log_return(close, 21)    # the past month
    out["ret_63d"] = log_return(close, 63)    # the past three months

    # ----------------------------------------------------------------------
    # GROUP 2: VOLATILITY - "How 'jumpy' has the price been?"
    # ----------------------------------------------------------------------
    # Standard deviation of the daily returns. Big number = big swings
    # (risky, nervous market). Small number = calm, steady market.
    out["vol_5d"] = daily_return.rolling(5).std()      # over the past week
    out["vol_21d"] = daily_return.rolling(21).std()    # over the past month
    out["vol_63d"] = daily_return.rolling(63).std()    # over the past three months

    # Is this week more jumpy than usual? Above 1 = yes, below 1 = calmer than usual.
    out["vol_ratio"] = out["vol_5d"] / out["vol_21d"]

    # How far apart today's highest and lowest prices were.
    out["range_hl"] = np.log(high / low)

    # The typical daily swing over the past 14 days (see average_true_range above).
    out["atr_14"] = average_true_range(high, low, close)

    # ----------------------------------------------------------------------
    # GROUP 3: TECHNICAL INDICATORS - classic signals that traders watch
    # ----------------------------------------------------------------------
    out["rsi_14"] = rsi(close)                                # mostly up days or mostly down days?
    out["macd_hist"] = macd_histogram(close)                  # is momentum speeding up or slowing down?
    out["bb_percent_b"], out["bb_width"] = bollinger(close)   # where is the price inside its "normal range"?

    # How far is today's price from its average over the last 20, 50 and 200
    # days? Positive = above the average, negative = below it.
    # Example: dist_sma_200 = 0.10 means the price is about 10% ABOVE its
    # 200-day average, which suggests a long-term uptrend.
    for window in (20, 50, 200):
        out[f"dist_sma_{window}"] = np.log(close / close.rolling(window).mean())

    # The overnight jump: yesterday's close compared with this morning's open.
    out["gap_overnight"] = np.log(open_ / close.shift(1))

    # The move during today's trading session: this morning's open compared
    # with this evening's close.
    out["move_intraday"] = np.log(close / open_)

    # ----------------------------------------------------------------------
    # GROUP 4: VOLUME - "Were more shares traded than usual today?"
    # ----------------------------------------------------------------------
    # Today's volume compared with the average of the last 20 days.
    # Positive = busier than usual, negative = quieter than usual.
    # Unusually heavy trading often comes with news.
    out["volume_rel_20d"] = np.log(volume / volume.rolling(20).mean())

    # ----------------------------------------------------------------------
    # GROUP 5: OTHER MARKETS - "What is happening in the wider world?"
    # ----------------------------------------------------------------------
    # A stock does not move on its own. When the whole market falls, most
    # stocks fall with it. So we also look at other markets.

    # The 1-day and 5-day returns of the S&P 500, the tech sector, gold, oil
    # (Brent) and the US dollar. This loop creates columns such as
    # SP500_ret_1d, SP500_ret_5d, GOLD_ret_1d, and so on.
    for name in LEVEL_PREDICTORS:
        out[f"{name}_ret_1d"] = log_return(frame[name], 1)
        out[f"{name}_ret_5d"] = log_return(frame[name], 5)

    # The VIX ("fear index"): how nervous investors are right now, and
    # whether they became more nervous today.
    out["VIX_level"] = np.log(frame[FEAR_PREDICTOR])
    out["VIX_change_1d"] = log_return(frame[FEAR_PREDICTOR], 1)

    # The 10-year US interest rate: its current level, and how much it moved
    # over the past week. A rate is already a percentage, so we subtract
    # (.diff) instead of taking a log return.
    out["US10Y_level"] = frame[RATE_PREDICTOR]
    out["US10Y_change_5d"] = frame[RATE_PREDICTOR].diff(5)

    # Did our stock beat the overall market this week?
    # Positive = it did better than the S&P 500, negative = it did worse.
    out["relative_to_SP500_5d"] = out["ret_5d"] - out["SP500_ret_5d"]

    # ----------------------------------------------------------------------
    # GROUP 6: CALENDAR - "What day and what month is it?"
    # ----------------------------------------------------------------------
    # Some people believe stocks behave a little differently on certain
    # weekdays or in certain months. These features let the model test that.
    # The date is always known in advance, so these features never peek.

    # One column per weekday. Each column is 1 on that weekday and 0 on every
    # other day. Example: on a Wednesday, is_wed = 1 and all the others = 0.
    weekday = frame.index.dayofweek            # Monday = 0, Tuesday = 1, ..., Friday = 4
    for day, label in enumerate(["mon", "tue", "wed", "thu", "fri"]):
        out[f"is_{label}"] = (weekday == day).astype(float)

    # The month, encoded as a position on a circle. If we used plain numbers
    # (1 to 12), the model would think December (12) and January (1) are far
    # apart, when really they are neighbours. Placing the months around a
    # circle, like the numbers on a clock face, fixes this. sin and cos are
    # simply the two coordinates of each month's position on that circle.

    # Each month is turned into an angle of month × 30° (a full 360° circle divided by 12).
    # The sin and cos of that angle place the month on a circle.

    # If you plot month_sin on the x-axis and month_cos on the y-axis,
    # the months form a clock face:
    #   December is at 12 o'clock (0, 1).
    #   March is at 3 o'clock (1, 0).
    #   June is at 6 o'clock (0, −1).
    #   September is at 9 o'clock (−1, 0).

    month = frame.index.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)

    return out


# The features sorted into the six groups above. Notebook 4 uses these groups
# to answer a simple question: which TYPE of information actually helps the
# forecast? (For example, does adding the "other_markets" group improve it?)
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

# One flat list of every feature name, taken from all the groups combined.
ALL_FEATURES = [name for group in FEATURE_GROUPS.values() for name in group]


def make_dataset(frame, horizon=1):
    """
    Build the final table that the models learn from.

    Each row pairs "what we knew on day t" (the features) with "what happened
    next" (the target). `horizon` is how many days ahead we forecast.
    horizon=1 means we forecast tomorrow.

    The returned table contains:
        all the feature columns   (from build_features)
        close           the closing price on day t. We already know this
                        price when we make the forecast.
        target_return   the log return from day t to day t+horizon.
                        THIS IS WHAT THE MODEL LEARNS TO PREDICT.
        target_close    the actual closing price on day t+horizon. We use it
                        to report the forecast as a price.

    Two things to know:
      - The first ~200 rows are removed, because some features (such as the
        200-day average) cannot be calculated yet.
      - The LAST `horizon` rows have no target yet, because that future has
        not happened. Their target_return is NaN. They are kept so that we
        can make a real forecast for tomorrow. Before training, remove them using:
            dataset.dropna(subset=["target_return"])
    """
    features = build_features(frame)
    data = features.copy()
    data["close"] = frame["Close"]

    # .shift(-horizon) pulls FUTURE prices back onto today's row.
    # This is the ONE place where looking ahead is correct, because these
    # columns are the answers we want to predict, not inputs to the model.
    data["target_close"] = frame["Close"].shift(-horizon)
    data["target_return"] = np.log(frame["Close"].shift(-horizon) / frame["Close"])

    # Division by zero can produce infinity. Treat it as missing data.
    data = data.replace([np.inf, -np.inf], np.nan)

    # Keep only the rows where every feature has a value.
    feature_ready = data[ALL_FEATURES].notna().all(axis=1)
    return data[feature_ready]


def check_no_lookahead(frame, feature_function=None, n_checks=5, seed=0):
    """
    An automatic test that catches features which secretly peek at the future.

    THE IDEA
    Pick a date, for example 14 June. Compute the features twice:
      (a) using ALL the data, and
      (b) using only the data up to 14 June (everything after it deleted).
    If the features are honest, the values for 14 June must be IDENTICAL in
    (a) and (b). Deleting the future should make no difference. If a value
    changes, that feature was using data from after 14 June, so it is
    "cheating".

    We repeat this at several randomly chosen dates.

    feature_function  the function to test. By default it is build_features.
                      If you invent a new feature, pass your own function here
                      to check if it is honest before you use it.

    Returns True if every feature passes. Otherwise it stops with an error
    that names the features that peeked at the future.
    """
    feature_function = feature_function or build_features
    rng = np.random.default_rng(seed)
    full = feature_function(frame)                       # (a) features from ALL the data

    # Choose random test dates from rows where all features have values.
    # Skip the last 10 rows, so that some future data is always deleted.
    valid_rows = np.where(full.notna().all(axis=1))[0]
    candidates = valid_rows[valid_rows < len(frame) - 10]
    cut_positions = rng.choice(candidates, size=n_checks, replace=False)

    for position in cut_positions:
        # (b) features from the data up to this date only (the future deleted)
        truncated = feature_function(frame.iloc[: position + 1])

        # Compare this date's values in (a) and (b). Any difference larger
        # than a tiny rounding error means a feature peeked at the future.
        differences = (full.iloc[position] - truncated.iloc[-1]).abs()
        bad = differences[~(differences <= 1e-9)].index.tolist()
        assert not bad, (
            f"Look-ahead bias detected on {frame.index[position].date()}: "
            f"these features changed when the future was removed: {bad}"
        )
    return True
