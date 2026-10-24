# Time-Series Analysis, Time-Series Forecasting, and Neural Networks

| Key              | Value |
|:-----------------|:------|
| **Course Codes** | BBT 4206 and BFS 4102 |
| **Course Names** | BBT 4206: Business Intelligence II (Week 7-9 of 13)<br>BFS 4102: Advanced Business Data Analytics (Week 1-3 of 13) |
| **Semester**     | September to December 2026 |
| **Lecturer**     | Allan Omondi |
| **Contact**      | aomondi@strathmore.edu |
| **Note**         | The lecture contains both theory and practice.<br/>These notebooks form part of the practice.<br/>They are intended for educational purposes only, and nothing in them is investment advice.<br/>Recommended citation: [BibTex](https://raw.githubusercontent.com/course-files/TimeSeriesForecastingandNeuralNetworks/refs/heads/main/RecommendedCitation.bib) |

## Repository Structure

```text
.
├── 1_market_data_and_candlesticks.ipynb
├── 2_stationarity_and_returns.ipynb
├── 3_classical_forecasting_baselines.ipynb
├── 4_multivariate_machine_learning_forecasting.ipynb
├── 5_neural_networks_from_first_principles.ipynb
├── 6_neural_networks_for_stock_forecasting.ipynb
├── 7_multi_horizon_forecast_and_power_bi_export.ipynb
├── 9_fourier_transform_optional.ipynb
├── download_data.py
├── lab_submission_instructions.md
├── tslab/                  helper package used by the notebooks (read it!)
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── validation.py
│   ├── metrics.py
│   ├── plots.py
│   └── nnutils.py
├── tests/                  automated checks for the helper package
├── data/                   the data snapshot (see data/README.md)
├── outputs/                files created when you run the notebooks
├── requirements/
├── admin_instructions/
├── assets/
├── LICENSE
└── RecommendedCitation.bib
```

## Setup Instructions

1. Follow [the project setup instructions](./0_setup_instructions/1_environment_setup.md) to create and activate a virtual environment (`.venv`).
2. Install the packages (the first line is for your own computer, the second is for Google Colab):

    ```shell
    pip install -r requirements/dev.txt -c requirements/constraints.txt
    pip install -r requirements/colab.txt
    ```

3. Confirm that everything works by running the automated checks from the project root:

    ```shell
    python -m pytest tests
    ```

4. Open the notebooks in order. **Select the interpreter from your `.venv`** as the notebook kernel. The first cell of every notebook prints the interpreter it uses; confirm that it points to your `.venv`.

**Python version.** The lab was developed and tested with Python 3.11. Python 3.10 to 3.12 are expected to work. TensorFlow supports a Python release only some months after it appears, so if the installation of `tensorflow` fails, check your Python version first and use 3.11 if in doubt.

**Google Colab.** Open a notebook from the repository in Colab. The first cell clones the repository and installs what is missing. Colab disconnects idle sessions, so download your results regularly.

## Lab Manual

Work through the notebooks in order. Each one starts with what you will learn, and ends with exercises and a summary.

| # | Notebook | What it teaches | Run time |
|:--|:---------|:----------------|:---------|
| 1 | [Market data, candlestick charts, and returns](1_market_data_and_candlesticks.ipynb) | OHLCV data, data quality checks, the predictors and why we chose them, prices versus returns | 1 minute |
| 2 | [Stationarity, autocorrelation, and why we forecast returns](2_stationarity_and_returns.ipynb) | ADF and KPSS tests, differencing, ACF and PACF, spurious regression, volatility clustering and GARCH | 1 minute |
| 3 | [Classical forecasting baselines](3_classical_forecasting_baselines.ipynb) | Naive, drift, moving average, exponential smoothing, ARIMA, ARMAX, walk-forward validation, the Diebold-Mariano test | 2 minutes |
| 4 | [Forecasting with many predictors](4_multivariate_machine_learning_forecasting.ipynb) | Feature engineering, look-ahead bias, Ridge, gradient boosting, tuning without leakage, feature ablation, prediction intervals, a back-test with trading costs | 2 minutes |
| 5 | [Neural networks from first principles](5_neural_networks_from_first_principles.ipynb) | Neurons, loss, gradient descent, backpropagation in NumPy, overfitting, RNN, LSTM, GRU, CNN and attention explained with small examples | 4 to 8 minutes |
| 6 | [Neural networks for forecasting](6_neural_networks_for_stock_forecasting.ipynb) | Windows, scaling without leakage, MLP, LSTM, GRU, CNN and Transformer in Keras, seeds and ensembles, tuning, realistic expectations | 6 to 15 minutes |
| 7 | [Multi-horizon forecasts and Power BI export](7_multi_horizon_forecast_and_power_bi_export.ipynb) | Direct multi-step forecasting, overlapping labels, how skill and uncertainty change with the horizon, the 63-day forecast | 2 minutes |
| 9 | [Fourier transform (optional)](9_fourier_transform_optional.ipynb) | Decomposing a complex seasonal pattern into sine waves | 1 minute |

The run times are for a typical laptop with a CPU. No GPU is needed.

## Lab Submission Instructions

- [Lab Submission Instructions](lab_submission_instructions.md)

## Cleanup Instructions (to be done after submitting the lab)

- [Cleanup Instructions](0_setup_instructions/4_teardown_instructions.md)
