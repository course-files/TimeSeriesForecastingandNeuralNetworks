"""
How to judge a forecast.

Three ideas matter more than any formula in this file:

1. ALWAYS COMPARE WITH A NAIVE FORECAST.
   The "no change" forecast (tomorrow equals today) is very hard to beat for
   a stock price. An RMSE of 2.3 dollars means nothing until you know that
   the naive forecast scores 2.3 dollars as well.

2. ONE NUMBER IS NOT ENOUGH.
   A model can have a small error and still be useless for decisions
   (it never predicts the direction correctly), or the other way around.

3. A DIFFERENCE IS NOT A DISCOVERY UNTIL IT SURVIVES A SIGNIFICANCE TEST.
   Two models with errors of 1.50 and 1.49 are not different. The
   Diebold-Mariano test below asks whether the gap is bigger than luck.
"""

import numpy as np
import pandas as pd
from scipy import stats


def rmse(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.mean(np.abs(actual - predicted)))


def mape(actual, predicted):
    """Mean absolute percentage error (in percent). Only valid for non-zero actuals."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def directional_accuracy(actual_return, predicted_return):
    """
    Share of days on which the model got the SIGN of the move right.

    A forecast of exactly zero (the naive forecast) has no direction, so those
    days are left out. If no day has a direction, the result is NaN, shown in
    the tables as a blank. A coin toss scores about 0.50.
    """
    actual_return, predicted_return = np.asarray(actual_return), np.asarray(predicted_return)
    has_direction = np.abs(predicted_return) > 1e-8
    if not has_direction.any():
        return float("nan")
    return float(np.mean(np.sign(actual_return[has_direction]) == np.sign(predicted_return[has_direction])))


def directional_accuracy_test(actual_return, predicted_return):
    """
    Is the hit rate better than a coin toss? Returns (hit_rate, p_value).
    A two-sided exact binomial test against a 50 percent hit rate.
    """
    actual_return, predicted_return = np.asarray(actual_return), np.asarray(predicted_return)
    has_direction = np.abs(predicted_return) > 1e-8
    actual_return, predicted_return = actual_return[has_direction], predicted_return[has_direction]
    hits = int(np.sum(np.sign(actual_return) == np.sign(predicted_return)))
    result = stats.binomtest(hits, len(actual_return), 0.5)
    return hits / len(actual_return), float(result.pvalue)


def out_of_sample_r2(actual_return, predicted_return):
    """
    Out-of-sample R squared against the 'predict zero return' benchmark.

        R2_oos = 1 - SSE(model) / SSE(zero forecast)

    Positive: the model beats the naive forecast. Zero or negative: it does not.
    For daily stock returns, honest values sit very close to zero (a fraction
    of one percent). A large positive value should make you suspect a leak
    in your pipeline before you celebrate.
    """
    actual_return, predicted_return = np.asarray(actual_return), np.asarray(predicted_return)
    sse_model = np.sum((actual_return - predicted_return) ** 2)
    sse_naive = np.sum(actual_return ** 2)
    return float(1 - sse_model / sse_naive)


def diebold_mariano(actual, forecast_a, forecast_b, horizon=1):
    """
    Diebold-Mariano test for equal forecast accuracy (squared error loss),
    with the Harvey-Leybourne-Newbold small sample correction.

    H0: the two forecasts are equally accurate.
    Returns (statistic, p_value). A NEGATIVE statistic means forecast A has the
    smaller error. A small p-value (below 0.05) means the difference is
    unlikely to be luck.
    """
    actual = np.asarray(actual)
    error_a = actual - np.asarray(forecast_a)
    error_b = actual - np.asarray(forecast_b)
    loss_difference = error_a ** 2 - error_b ** 2
    n = len(loss_difference)
    mean_difference = loss_difference.mean()

    # Long-run variance of the loss differences, with lags up to horizon - 1
    centred = loss_difference - mean_difference
    long_run_variance = np.sum(centred ** 2) / n
    for lag in range(1, horizon):
        weight = 1 - lag / horizon
        long_run_variance += 2 * weight * np.sum(centred[lag:] * centred[:-lag]) / n
    statistic = mean_difference / np.sqrt(long_run_variance / n)

    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic = statistic * correction
    p_value = 2 * (1 - stats.t.cdf(abs(statistic), df=n - 1))
    return float(statistic), float(p_value)


def dm_verdict(statistic, p_value, level=0.05):
    """
    Put the Diebold-Mariano result into words.
    (The statistic compares the model with the naive forecast: negative means the model has the smaller error.)
    """
    if p_value >= level:
        return "No significant difference"
    return "Significantly BETTER than naive" if statistic < 0 else "Significantly WORSE than naive"


def evaluate_forecast(close_today, close_next_actual, predicted_return, name="model"):
    """
    Convert a predicted RETURN into a predicted closing PRICE and score it.

    close_today        closing price on day t (a Series)
    close_next_actual  the actual closing price on day t+1
    predicted_return   the model's predicted log return from t to t+1
    """
    close_today = np.asarray(close_today)
    actual_close = np.asarray(close_next_actual)
    predicted_return = np.asarray(predicted_return)
    actual_return = np.log(actual_close / close_today)
    predicted_close = close_today * np.exp(predicted_return)
    return pd.Series(
        {
            "RMSE ($)": rmse(actual_close, predicted_close),
            "MAE ($)": mae(actual_close, predicted_close),
            "MAPE (%)": mape(actual_close, predicted_close),
            "Directional accuracy": directional_accuracy(actual_return, predicted_return),
            "Out-of-sample R2 (returns)": out_of_sample_r2(actual_return, predicted_return),
        },
        name=name,
    )


def compare_models(dataset, predictions, naive_name="Naive (no change)"):
    """
    Score several models on exactly the same rows.

    dataset      the modelling table (needs close, target_close, target_return)
    predictions  dict of  model name -> Series of predicted returns
                 (NaN where the model made no prediction)
    The naive forecast (predicted return = 0) is added automatically.
    """
    rows = None
    for name, series in predictions.items():
        mask = series.notna() if rows is None else (rows & series.notna())
        rows = mask
    scored = dataset.loc[rows]
    table = []
    table.append(
        evaluate_forecast(scored["close"], scored["target_close"], np.zeros(len(scored)), naive_name)
    )
    for name, series in predictions.items():
        table.append(
            evaluate_forecast(scored["close"], scored["target_close"], series.loc[rows].values, name)
        )
    return pd.DataFrame(table)


def simple_backtest(actual_return, predicted_return, cost_bps=0.0, long_only=False):
    """
    A deliberately simple trading experiment, to show what a forecast is worth.

    Each day we hold +1 unit if the model predicts a rise and -1 unit (a short
    sale) if it predicts a fall; with long_only=True we hold 0 instead of -1.
    Every change of position costs `cost_bps` basis points (1 bp = 0.01 percent)
    of the amount traded. Returns a Series of daily strategy log returns.
    """
    actual_return = pd.Series(np.asarray(actual_return, dtype=float))
    signal = np.sign(np.asarray(predicted_return, dtype=float))
    if long_only:
        signal = np.clip(signal, 0, 1)
    position = pd.Series(signal)
    trades = position.diff().abs().fillna(position.abs())
    return position * actual_return - trades * cost_bps / 10_000


def backtest_summary(daily_returns, periods_per_year=252):
    """Annualised return, volatility and Sharpe ratio of a daily return Series."""
    daily_returns = pd.Series(daily_returns).dropna()
    annual_return = daily_returns.mean() * periods_per_year
    annual_vol = daily_returns.std() * np.sqrt(periods_per_year)
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
    return pd.Series(
        {"Annual return": annual_return, "Annual volatility": annual_vol, "Sharpe ratio": sharpe}
    )
