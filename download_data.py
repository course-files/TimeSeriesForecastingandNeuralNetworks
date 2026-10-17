"""
Download the data snapshot used by every notebook.

Run this ONCE from the project root:

    python download_data.py

It needs an internet connection and the yfinance package. It writes two
files into the data/ folder:

    data/target_<TICKER>.csv    the stock we forecast (Open, High, Low, Close, Volume)
    data/predictors.csv         the closing level of each predictor market

To use a different stock, pass its Yahoo Finance symbol:

    python download_data.py --target MSFT
    python download_data.py --target SCOM.NR      (Safaricom; check the data first,
                                                   coverage of the Nairobi exchange on
                                                   Yahoo Finance is patchy)

If your stock is not available on Yahoo Finance (many Nairobi Securities Exchange
shares are not), download its daily prices by hand as a CSV file from another
site, and let this script clean the file. The predictor markets are still
downloaded from Yahoo Finance, unless a predictors.csv file already exists.

    python download_data.py --target SCOM --from-csv "data/Download Data - STOCK_KE_XNAI_SCOM.csv"
"""

import argparse

from tslab import config
from tslab.data import (PREDICTORS_FILE, build_snapshot, clean_manual_csv, data_quality_report,
                        download_prices, load_market_data)


def main():
    parser = argparse.ArgumentParser(description="Download the market data snapshot.")
    parser.add_argument("--target", default=config.TARGET_TICKER, help="Yahoo Finance symbol of the stock to forecast")
    parser.add_argument("--start", default=config.SNAPSHOT_START)
    parser.add_argument("--end", default=config.SNAPSHOT_END, help="exclusive end date, YYYY-MM-DD")
    parser.add_argument("--from-csv", default=None, help="a CSV of daily prices that you downloaded by hand (see above)")
    args = parser.parse_args()

    if args.from_csv:
        clean_manual_csv(args.from_csv, target=args.target)
        if not (config.DATA_DIR / PREDICTORS_FILE).exists():
            import pandas as pd
            print("Downloading the predictor markets from Yahoo Finance...")
            closes = {name: download_prices(symbol, args.start, args.end)["Close"]
                      for name, symbol in config.PREDICTOR_TICKERS.items()}
            pd.DataFrame(closes).sort_index().to_csv(config.DATA_DIR / PREDICTORS_FILE)
    else:
        build_snapshot(target=args.target, start=args.start, end=args.end)

    print("\nChecking what was saved...")
    frame = load_market_data(target=args.target)
    print(data_quality_report(frame).to_string())
    print(
        "\nIf you changed the target symbol, also set TARGET_TICKER in "
        "tslab/config.py so that the notebooks load the right file."
    )


if __name__ == "__main__":
    main()
