# The data folder

| File | What it is |
|:-----|:-----------|
| `target_NVDA.csv` | Daily Open, High, Low, Close and Volume of the stock we forecast. Prices are adjusted for splits and dividends. |
| `predictors.csv` | The daily closing level of the seven predictor markets: S&P 500 (`SP500`), a technology sector fund (`TECH`), the VIX (`VIX`), the 10-year US Treasury yield (`US10Y`), the US dollar index (`USD`), Brent crude oil (`BRENT`) and gold (`GOLD`). |
| `forecast_basic.csv`, `forecast_backtest.csv` | Created when you run Notebook 7. These are the files you import into Power BI. |
| `Download Data - STOCK_KE_XNAI_SCOM.csv` | A small example of a price file downloaded by hand (Safaricom, from MarketWatch). It shows what such a file looks like before it is cleaned. |

## Where the data comes from

The market data was downloaded from Yahoo Finance with the `yfinance` Python package on **17 October 2026**, for the period **2 January 2015 to 30 September 2026**, using `python download_data.py`. The same script made the files in this folder.

The data is provided for teaching and for your own learning only. It remains subject to the terms of the data provider. Do not redistribute it, and do not use it for trading decisions.

## Why a fixed snapshot?

1. **Everyone gets the same numbers.** If your RMSE differs from a classmate's, you can look for a mistake in the code.
2. **Adjusted prices change over time.** Each time a company pays a dividend, the adjusted history is rewritten. Downloading on a different day can give slightly different numbers.
3. **A classroom of laptops should not all ask Yahoo Finance for the same data at the same moment.** Doing so can cause requests to be refused for the whole network.

## Using your own stock (for the submission)

```shell
python download_data.py --target MSFT
```

This downloads the new stock and the seven predictors, and replaces the two files above, so **copy this folder somewhere safe first** if you want to keep the originals. Then set `TARGET_TICKER = "MSFT"` in `tslab/config.py`.

If your stock is not available on Yahoo Finance (for example, many Nairobi Securities Exchange shares), download its daily prices by hand as a CSV file from another site, and let the script clean the file:

```shell
python download_data.py --target SCOM --from-csv "path/to/your/file.csv"
```

The file needs the columns Date, Open, High, Low, Close and Volume.
