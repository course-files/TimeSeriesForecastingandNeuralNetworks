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

Your client, a portfolio manager at a financial institution, is seeking to
enhance their investment decision-making using data-driven techniques. They
have provided you with a dataset containing historical daily stock prices
for a publicly traded company. The client wants you to apply time series
forecasting methods to predict the company's stock price for the next 63
business days (the next quarter), supporting their short-term trading and
risk management strategies.

---
## Dataset

Use any OHLCV stock data apart from the ones used in the notebooks.

## Baseline (Required)

1. **Forecast Stock Prices**
   - Use any one forecasting technique (Simple Exponential Smoothing, Double Exponential Smoothing (Holt's Method), Triple Exponential Smoothing (Holt-Winters Method), ARIMA, SARIMA, Prophet, or LSTM) as demonstrated in the notebooks.
   - Forecast the next 63 business days of closing prices.

2. **Save Forecasted Data**
   - Save your forecasted results as a CSV file (`./data/forecast_basic.csv`).

3. **Power BI Import**
   - Import your CSV file into Power BI.
   - Create a simple line chart showing the forecasted closing prices.

---

## Intermediate Level (Recommended)

1. **Compare Multiple Forecasting Methods**
   - Apply at least two different forecasting models (e.g., ARIMA and Prophet, or Holt-Winters and LSTM).
   - Forecast the next 63 business days for each method.

2. **Save Forecasted Data**
   - Save all forecast results in a single CSV file with separate columns for each method (`./data/forecast_intermediate.csv`).

3. **Power BI Dashboard**
   - Import your CSV file into Power BI.
   - Create a dashboard with:
     - A line chart comparing forecasts from each method.
     - A table summarizing forecasted values.

---

## Advanced Level (Optional)

1. **Hyperparameter Tuning & Model Comparison**
   - Apply at least three forecasting models, including LSTM with hyperparameter tuning (as shown in the notebooks).
   - Forecast the next 63 business days for each model and each LSTM configuration.

2. **Save Forecasted Data**
   - Save all forecast results in a single CSV file, with columns for each model and LSTM configuration (e.g., `forecast_advanced.csv`).

3. **Power BI Dashboard**
   - Import your CSV file into Power BI.
   - Create a dashboard with:
     - Comparative line charts for all models and configurations.
     - A summary table of forecasted values.
     - Visuals highlighting differences between models (e.g., error metrics, trends).

4. **Use of NVIDIA GPU**
   - If you have access to an NVIDIA GPU (locally not on Colab), demonstrate what must be done to use it for training the LSTM model to improve performance and reduce training time.
   - Document the impact of using the GPU on training time and model performance in your submission.

---

**General Instructions:**
- Ensure your CSV files include a date column for the forecasted periods.
- Use the code examples in the repository for guidance on saving and formatting your forecast data.
- Your Power BI dashboard should clearly present the forecast results for interpretation and decision-making.


## Grading Approach

- Baseline = Pass >= 60%
- Intermediate = Merit 75–85%
- Advanced = Distinction >= 86%
