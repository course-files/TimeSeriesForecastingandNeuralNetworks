# Lab Submission Instructions

---

## Student Details

**Name of the team on GitHub Classroom:**

**Team Member Contributions:**

**Member 1**

| **Details**                                                                                        | **Comment** |
|:---------------------------------------------------------------------------------------------------|:------------|
| **Student ID:**                                                                                    |             |
| **Name:**                                                                                          |             |
| **What part of the lab did you personally contribute to,** <br>**and what did you learn from it?** |             |

**Member 2**

| **Details**                                                                                        | **Comment** |
|:---------------------------------------------------------------------------------------------------|:------------|
| **Student ID:**                                                                                    |             |
| **Name:**                                                                                          |             |
| **What part of the lab did you personally contribute to,** <br>**and what did you learn from it?** |             |

**Member 3**

| **Details**                                                                                        | **Comment** |
|:---------------------------------------------------------------------------------------------------|:------------|
| **Student ID:**                                                                                    |             |
| **Name:**                                                                                          |             |
| **What part of the lab did you personally contribute to,** <br>**and what did you learn from it?** |             |

**Member 4**

| **Details**                                                                                        | **Comment** |
|:---------------------------------------------------------------------------------------------------|:------------|
| **Student ID:**                                                                                    |             |
| **Name:**                                                                                          |             |
| **What part of the lab did you personally contribute to,** <br>**and what did you learn from it?** |             |

**Member 5**

| **Details**                                                                                        | **Comment** |
|:---------------------------------------------------------------------------------------------------|:------------|
| **Student ID:**                                                                                    |             |
| **Name:**                                                                                          |             |
| **What part of the lab did you personally contribute to,** <br>**and what did you learn from it?** |             |

## Scenario

Your client is a portfolio manager at a financial institution. The manager wants to know what the **closing price of one stock** is likely to do over the next trading day, the next week, the next month and the next quarter (63 business days). The manager has been disappointed before by forecasts that looked excellent on a chart and failed in practice, so the client has set one condition:

> Show me that your forecast beats the simplest possible forecast ("tomorrow will be the same as today") on days that your model never saw, and tell me honestly how uncertain it is.

You will answer that question using **real market data** and **several predictors at once**, with the methods taught in the notebooks.

---

## Dataset

1. Your group must forecast **a stock that is different from NVDA**. Choose a company whose price history is available on Yahoo Finance (for example MSFT, JPM, TSLA, NVDA, KO, or a company listed on another exchange).
2. Download the data with the helper script:

    ```shell
    python download_data.py --target MSFT
    ```

    and then set `TARGET_TICKER = "MSFT"` in `tslab/config.py`, so that the notebooks load your stock.
3. If your stock is not available on Yahoo Finance (many Nairobi Securities Exchange shares are not), download the daily prices by hand from another site and let the script clean the file:

    ```shell
    python download_data.py --target SCOM --from-csv "data/Download Data - STOCK_KE_XNAI_SCOM.csv"
    ```

    Check the quality report that the script prints. Missing days, repeated prices (no trading), and very short histories will limit what you can do, and you must say so in your report. A history shorter than about four years cannot support the walk-forward settings used in the notebooks; reduce `INITIAL_TRAIN` in `tslab/config.py` and explain your choice.
4. The predictors (S&P 500, a technology fund, VIX, 10-year yield, US dollar, Brent crude, gold) are downloaded for you. You may add or replace predictors if you can justify why they would matter for **your** stock (for example, a local index for a local stock). Predictors must be known **before** the day you forecast.
5. State the **date on which you downloaded your data**. Market data providers revise history, so the date matters.

---

## What Every Tier Must Respect

These rules apply to Baseline, Intermediate and Advanced alike. They are the reason this lab exists.

1. **Forecast the return, then convert to a price.** A one-day-ahead prediction of the closing price is the last close multiplied by `exp(predicted log return)`.
2. **Never shuffle time.** Use walk-forward (expanding window) validation, as in Notebook 3. A random train/test split on a time series is a serious error.
3. **Fit every scaler and every model on the training block only.**
4. **Tune on data that is before your test period**, never on the test period.
5. **Compare every model with the naive forecast**, and report whether the difference is statistically significant (Diebold-Mariano test). A model that does not beat the naive forecast is a valid finding. You will not lose marks for an honest negative result. You will lose marks for hiding it.
6. **Run the look-ahead test** (`check_no_lookahead`) on every feature that you add.

---

## Baseline (Required)

1. **Prepare and understand your data**
   - Run the data quality report for your stock, and plot a candlestick chart for a period of your choice.
   - Test the closing price and the log return for stationarity (ADF and KPSS), and state the conclusion.

2. **Forecast with multiple predictors**
   - Build the forecast of the next-day log return using the feature set of Notebook 4 (or an improved one), with **at least one** model from this list: ARMAX, Ridge regression, gradient boosting.
   - Evaluate it with walk-forward validation, and report RMSE, MAE, the out-of-sample R2 and the directional accuracy **next to the naive forecast**.

3. **Produce the 63-day forecast**
   - Produce a forecast of the closing price for the next 63 business days, **with a lower and an upper bound**.
   - Save it as `./data/forecast_basic.csv` (columns: `date`, `forecast`, `lower`, `upper`).

4. **Power BI**
   - Import the CSV file and create a line chart that shows the last year of actual closing prices, the forecast, and the range.

5. **A short written verdict** (about half a page, in this file or in a separate file) that a portfolio manager could understand: does your model beat the naive forecast, how do you know, and how much should the client trust the 63-day forecast?

---

## Intermediate Level (Recommended)

Everything in Baseline, and:

1. **Compare at least three forecasts**
   - The naive forecast, one classical model (ARIMA, ARMAX or exponential smoothing), and one machine learning model (Ridge or gradient boosting) with many predictors.
   - Use the Diebold-Mariano test for each model against the naive forecast.

2. **Feature ablation**
   - Run a feature group ablation (Notebook 4, Step 8) and explain, in your own words, which group of predictors helps and which does not. Remember that a very small R2 difference is not a finding.

3. **Prediction intervals**
   - Estimate a 10% to 90% range with quantile regression, and report the **coverage** (the share of actual outcomes inside the range). Comment on whether the model is overconfident.

4. **Multi-horizon results**
   - Report the skill of your model at horizons of 1, 5, 21 and 63 days (Notebook 7), and explain why it changes with the horizon.

5. **Save and visualise**
   - Save all the forecasts in one CSV file, with a separate column for each method (`./data/forecast_intermediate.csv`).
   - Create a Power BI dashboard with a line chart comparing the methods, a table of the forecast values, and a card or table that shows the error metrics of each method.

---

## Advanced Level (Optional)

Everything in Intermediate, and:

1. **A neural network**
   - Train at least one recurrent or convolutional network (LSTM, GRU or CNN) on windows of your features, using the protocol of Notebook 6: scalers fitted on the training block only, a validation share taken from the **end** of the training block, early stopping, and a fixed random seed.
   - Report the **zero-forecast baseline** loss next to your validation loss, so that a reader can see whether the network learned anything.

2. **Tuning and stability**
   - Tune at least two hyperparameters (for example the window length, the number of units, or the dropout rate) using the validation data **only**, and show the results in a table.
   - Train with at least three different seeds and report the spread of the results. Explain what the spread tells you.

3. **Save and visualise**
   - Save all the forecasts, with a column for each model and each neural network configuration, in a single CSV file (`./data/forecast_advanced.csv`).
   - Create a Power BI dashboard with comparative charts for all the models, a summary table, and a visual that highlights differences between models (for example error metrics, or the cumulative back-test curves).

4. **A back-test with costs**
   - Run the long/short experiment of Notebook 4 (Step 11) for your best model, with transaction costs of 0, 5 and 10 basis points, and compare it with buy and hold. Explain why the result is not a trading recommendation.

5. **Use of an NVIDIA GPU (optional)**
   - If you have access to an NVIDIA GPU, explain what is needed to use it for training (see the note in Notebook 6), and document the effect on training time and on the results.

---

## General Instructions

- Every CSV file must contain a date column for the forecast period.
- Use the notebooks and the `tslab` package as your guide. You may copy and adapt code from them. You must be able to **explain every line** that you submit.
- Use your own words in the written parts. Describe what **your** results show, not what the notebooks said about the NVDA data.
- Keep your results reproducible: set the random seed, state the download date of your data, and make sure the notebooks run from top to bottom after a restart of the kernel.

### Video Demonstration

For all three tiers, submit the link to a short video (not more than 4 minutes) that shows:

1. Your stock and the data quality report.
2. Your main result table, with the naive forecast in it, and what you conclude from it.
3. Your Power BI report or dashboard with the forecast and the range.

| **Key**                | **Value** |
|:-----------------------|:----------|
| **Link to the video:** |           |

---

## How Your Submission Will Be Marked (Out of 100)

| #   | What you need to show                                                                                                                | Marks |
|:----|:-------------------------------------------------------------------------------------------------------------------------------------|------:|
| 1   | **Data**: a justified stock, a data quality check, a stated download date, and a stationarity analysis                               | 10    |
| 2   | **Honest validation**: walk-forward validation, no look-ahead, tuning before the test period, scalers fitted on the training block   | 25    |
| 3   | **Evaluation against the naive forecast**: metrics, the Diebold-Mariano test, and a correct reading of the result                    | 20    |
| 4   | **Modelling quality** for your tier (see the tier descriptions above)                                                                | 15    |
| 5   | **Uncertainty**: a forecast range, and (Intermediate and Advanced) the coverage of the range                                          | 10    |
| 6   | **Power BI** report or dashboard for your tier, readable by a manager                                                                | 10    |
| 7   | **Written verdict and video**: clear, honest, in your own words, within 4 minutes                                                    | 10    |

### Caps (these apply to the whole submission)

- **No naive forecast as a comparison**: your total is capped at 50.
- **Randomly shuffled train/test split, or any scaler or model that was fitted using the test period**: your total is capped at 50.
- **Features that use future information** (look-ahead bias) in the final model: your total is capped at 60.
- **Same stock as the notebooks (NVDA)**: your total is capped at 60.
- **No individual contribution documented** for a team member: minus 5 for each such member.

### Grading Approach

- Baseline = Pass >= 60%
- Intermediate = Merit 75 to 85%
- Advanced = Distinction >= 86%

A model that does **not** beat the naive forecast does not lose marks for that reason. A claim that it does, without the evidence, does.

### Before You Submit, Check

- [ ] The notebooks run from top to bottom after restarting the kernel.
- [ ] The naive forecast appears in every result table.
- [ ] The CSV files for your tier are in the `data` folder and open correctly in Power BI.
- [ ] Your written verdict says plainly whether the model beats the naive forecast.
- [ ] The video is not longer than 4 minutes and the link works without a login.
- [ ] The contribution of every team member is filled in at the top of this file.
