"""
Getting the market data into a DataFrame.

There are two ways to get data, and it is worth understanding why we have both.

1. THE SNAPSHOT (what students use).
   The lecturer downloads the data once, on one date, and saves it in the
   data/ folder. Every student then loads the same files, so every student
   gets the same numbers. This also protects us from a classroom full of
   laptops asking Yahoo Finance for the same data at the same time, which
   gets the whole campus network blocked.

2. A LIVE DOWNLOAD (what the lecturer uses, and what you use for your own
   stock in the submission).
   Run  python download_data.py  from the project root. It calls Yahoo
   Finance through the yfinance package and writes the snapshot files.
"""

import time
from pathlib import Path

import pandas as pd

from . import config

TARGET_FILE_TEMPLATE = "target_{ticker}.csv"
PREDICTORS_FILE = "predictors.csv"


# ---------------------------------------------------------------------------
# Live download
# ---------------------------------------------------------------------------
def download_prices(ticker, start, end, retries=3, pause=5.0):
    """
    Download daily OHLCV data for one ticker from Yahoo Finance.

    Prices are ADJUSTED for stock splits and dividends (auto_adjust=True).
    Without that adjustment a 4-for-1 split would look like a 75 percent crash.
    A consequence worth remembering: adjusted history is rewritten each time a
    dividend is paid, which is another reason we freeze a snapshot.
    """
    import yfinance as yf  # imported here so that loading a snapshot needs no internet package

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            frame = yf.Ticker(ticker).history(
                start=start, end=end, interval="1d", auto_adjust=True, actions=False
            )
            if frame is not None and len(frame) > 0:
                frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
                frame.index.name = "Date"
                return frame[["Open", "High", "Low", "Close", "Volume"]].sort_index()
            last_error = "Yahoo Finance returned no rows"
        except Exception as error:  # network problem, rate limit, bad symbol
            last_error = str(error)
        print(f"  attempt {attempt} for {ticker} failed ({last_error}); waiting {pause} seconds")
        time.sleep(pause)
    raise RuntimeError(
        f"Could not download {ticker}. Last problem: {last_error}. "
        "If many students are downloading at once Yahoo Finance may be limiting "
        "requests. Use the snapshot in the data/ folder instead."
    )


def build_snapshot(data_dir=None, target=None, predictors=None, start=None, end=None):
    """Download the target stock and all predictors, and save them as CSV files."""
    data_dir = Path(data_dir or config.DATA_DIR)
    target = target or config.TARGET_TICKER
    predictors = predictors or config.PREDICTOR_TICKERS
    start = start or config.SNAPSHOT_START
    end = end or config.SNAPSHOT_END
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading the target stock {target} from {start} to {end} (exclusive)")
    target_frame = download_prices(target, start, end)
    target_path = data_dir / TARGET_FILE_TEMPLATE.format(ticker=target)
    target_frame.to_csv(target_path)
    print(f"  saved {len(target_frame)} rows to {target_path}")

    closes = {}
    for name, ticker in predictors.items():
        print(f"Downloading predictor {name} ({ticker})")
        closes[name] = download_prices(ticker, start, end)["Close"]
    predictor_frame = pd.DataFrame(closes).sort_index()
    predictor_path = data_dir / PREDICTORS_FILE
    predictor_frame.to_csv(predictor_path)
    print(f"  saved {len(predictor_frame)} rows to {predictor_path}")
    return target_frame, predictor_frame


def clean_manual_csv(path, target=None, data_dir=None):
    """
    Turn a price file that you downloaded BY HAND into the format used by this lab.

    Many exchanges and data sites (MarketWatch, for example, for the Nairobi
    Securities Exchange) let you download a CSV of daily prices from a web page.
    Those files are not clean: the newest day is first, dates are written as
    month/day/year, and numbers contain quotation marks and thousands
    separators such as "1,231,534". This function fixes all of that.

    The file must have the columns Date, Open, High, Low, Close and Volume
    (in any order and with any capitalisation). The result is saved as
    data/target_<target>.csv.

    NOTE: hand-downloaded prices are usually NOT adjusted for splits and
    dividends. Check the history of your stock for splits before you trust it.
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    target = target or Path(path).stem.replace(" ", "_")
    raw = pd.read_csv(path, thousands=",", quotechar='"')
    raw.columns = [column.strip().title() for column in raw.columns]
    missing = {"Date", "Open", "High", "Low", "Close", "Volume"} - set(raw.columns)
    if missing:
        raise ValueError(f"The file is missing these columns: {sorted(missing)}. Found: {list(raw.columns)}")

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        raw[column] = pd.to_numeric(raw[column].astype(str).str.replace(",", "", regex=False), errors="coerce")
    clean = raw.dropna(subset=["Date", "Close"]).set_index("Date").sort_index()
    clean = clean[~clean.index.duplicated(keep="last")][["Open", "High", "Low", "Close", "Volume"]]

    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / TARGET_FILE_TEMPLATE.format(ticker=target)
    clean.to_csv(out_path)
    print(f"Cleaned {len(clean)} rows ({clean.index.min().date()} to {clean.index.max().date()}) and saved them to {out_path}")
    return clean


# ---------------------------------------------------------------------------
# Loading the snapshot
# ---------------------------------------------------------------------------
def load_market_data(data_dir=None, target=None, verbose=True):
    """
    Load the snapshot and return ONE DataFrame, one row per trading day of the
    target stock.

    Columns:  Open, High, Low, Close, Volume   (the stock itself)
              SP500, TECH, VIX, US10Y, USD, BRENT, GOLD   (the predictors)

    The predictor markets do not always trade on the same days as the stock.
    We keep the stock's trading calendar and carry the last known predictor
    value forward for at most MAX_FFILL_DAYS days. We never fill backwards,
    because that would copy a FUTURE value into the past.
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    target = target or config.TARGET_TICKER
    target_path = data_dir / TARGET_FILE_TEMPLATE.format(ticker=target)
    predictor_path = data_dir / PREDICTORS_FILE

    if not target_path.exists() or not predictor_path.exists():
        raise FileNotFoundError(
            "The data snapshot was not found in the data/ folder.\n"
            "Option 1: pull the latest version of the repository (the lecturer "
            "publishes the snapshot there).\n"
            "Option 2: run  python download_data.py  from the project root "
            "(needs an internet connection)."
        )

    stock = pd.read_csv(target_path, index_col="Date", parse_dates=True)
    others = pd.read_csv(predictor_path, index_col="Date", parse_dates=True)

    frame = stock.join(others, how="left")
    frame[others.columns] = frame[others.columns].ffill(limit=config.MAX_FFILL_DAYS)
    frame = frame.dropna()
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    if verbose:
        print(
            f"Loaded {len(frame)} trading days of {target}, "
            f"{frame.index.min().date()} to {frame.index.max().date()}, "
            f"with {len(others.columns)} predictor series."
        )
    return frame


def data_quality_report(frame):
    """
    A short list of the problems that most often break a forecasting project.
    Run it on any dataset BEFORE you model it.
    """
    report = {}
    report["rows"] = len(frame)
    report["first_date"] = frame.index.min().date()
    report["last_date"] = frame.index.max().date()
    report["duplicate_dates"] = int(frame.index.duplicated().sum())
    report["missing_values"] = int(frame.isna().sum().sum())
    report["non_positive_prices"] = int((frame[["Open", "High", "Low", "Close"]] <= 0).sum().sum())
    report["days_with_zero_volume"] = int((frame["Volume"] == 0).sum())
    report["high_below_low"] = int((frame["High"] < frame["Low"]).sum())
    gaps = frame.index.to_series().diff().dt.days
    report["longest_gap_in_days"] = int(gaps.max()) if len(gaps) > 1 else 0
    returns = frame["Close"].pct_change().abs()
    report["days_with_move_over_10pct"] = int((returns > 0.10).sum())
    return pd.Series(report)
