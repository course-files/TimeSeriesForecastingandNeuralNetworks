"""
Settings shared by every notebook.

Change TARGET_TICKER if you want to repeat the lab on another stock. For the
submission, your group must use a target stock that is different from the one
used in the notebooks (see lab_submission_instructions.md).
"""

from pathlib import Path

# The project root is the folder that contains this package, wherever the
# notebook happens to be launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# What we forecast: the daily closing price of one stock.
# ---------------------------------------------------------------------------
# TARGET_TICKER = "AAPL"
TARGET_TICKER = "NVDA"

# ---------------------------------------------------------------------------
# Predictors that are NOT the stock itself. Each one is a different market that
# a trader would plausibly watch. The dictionary maps our own short name to the
# ticker symbol used by Yahoo Finance.
#
#   SP500  : the broad US equity market (the stock moves with the market)
#   TECH   : a sector fund for technology companies
#   VIX    : the "fear index", the market's expected volatility
#   US10Y  : the yield on 10-year US government bonds (a proxy for interest rates)
#   USD    : the strength of the US dollar against a basket of currencies
#   BRENT  : the price of Brent crude oil
#   GOLD   : the price of gold, a traditional safe haven
#
# Brent is used instead of WTI crude on purpose: the WTI futures price was
# negative on 20 April 2020, and you cannot take the logarithm of a negative
# number. Checking that a transformation is valid for every row in your data
# is part of data preparation.
# ---------------------------------------------------------------------------
PREDICTOR_TICKERS = {
    "SP500": "^GSPC",
    "TECH": "XLK",
    "VIX": "^VIX",
    "US10Y": "^TNX",
    "USD": "DX-Y.NYB",
    "BRENT": "BZ=F",
    "GOLD": "GC=F",
}

# The snapshot is a fixed period so that every student works on identical
# numbers. END is exclusive (Yahoo Finance convention), so the last trading
# day in the snapshot is 30 September 2026.
SNAPSHOT_START = "2015-01-01"
SNAPSHOT_END = "2026-10-01"

# Predictors are forward filled for at most this many days. This covers days
# when one market was open and another was closed (public holidays).
MAX_FFILL_DAYS = 3

# Random seed used everywhere, so that your results can be compared with your
# classmates' results and with the lecturer's.
SEED = 42

# ---------------------------------------------------------------------------
# Walk-forward validation settings used by the forecasting notebooks. Keeping
# them here means that notebooks 3, 4, 6 and 7 are scored on the same days and
# can therefore be compared fairly.
# ---------------------------------------------------------------------------
INITIAL_TRAIN = 1000   # rows in the first training block (about four years)
TEST_SIZE = 63         # rows in each test block (about one quarter)
