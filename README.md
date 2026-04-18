# Stock Market Trading Expert System

A beginner-friendly AI-based stock trading decision support system built with Python, Flask, scikit-learn, pandas, and Yahoo Finance data.

The project predicts `BUY`, `SELL`, or `HOLD` signals using historical stock data, technical indicators, a simple machine learning model, and rule-based explainability.

## Features

- Fetches historical stock data from Yahoo Finance using `yfinance`
- Computes technical indicators such as `MA_20`, `MA_50`, `RSI`, and `MACD`
- Trains a machine learning classifier for next-day price direction
- Generates `BUY / SELL / HOLD` signals
- Adds explainable reasoning based on RSI, price trend, and MACD
- Runs a simple backtest to compare strategy performance vs buy-and-hold
- Includes a Flask web dashboard with signal output and interactive Plotly charts

## Tech Stack

- Python
- Flask
- scikit-learn
- pandas
- numpy
- yfinance
- Plotly

## Project Structure

```text
stock_trading_expert_system/
├── app/
│   ├── __init__.py
│   └── routes.py
├── data/
├── ml/
│   ├── backtest.py
│   ├── data_fetch.py
│   ├── indicators.py
│   ├── predict.py
│   └── train.py
├── models/
├── templates/
│   └── index.html
├── app.py
├── README.md
└── requirements.txt
```

## How It Works

### 1. Data Collection

Downloads historical OHLCV stock data:

- Open
- High
- Low
- Close
- Volume

Source: Yahoo Finance via `yfinance`

### 2. Feature Engineering

Builds technical indicators:

- 20-day moving average
- 50-day moving average
- 14-day RSI
- MACD
- MACD signal line
- MACD histogram
- Daily return

### 3. Machine Learning

Creates labels for next-day movement:

- `1` if next close is higher than current close
- `0` otherwise

Supports:

- Logistic Regression
- Random Forest

### 4. Prediction Logic

Combines ML output with simple trading rules:

- If RSI > 70, avoid buying
- If RSI < 30, buying is preferred
- If trend and momentum conflict, soften the signal to `HOLD`

### 5. Backtesting

Simulates a basic long-or-cash strategy over historical data and reports:

- total return
- buy-and-hold return
- win rate
- number of trades

### 6. Web Dashboard

The Flask app lets the user:

- enter a stock ticker
- view the latest signal
- see indicator values
- read the explanation
- inspect interactive price, RSI, and MACD charts

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/The8thwarlord/stock_expert_system.git
cd stock_expert_system
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Fetch historical data

```bash
python ml/data_fetch.py AAPL --period 1y
python ml/data_fetch.py TSLA --period 6mo
python ml/data_fetch.py RELIANCE.NS --start 2025-01-01 --end 2025-12-31
```

### Build feature dataset

```bash
python ml/indicators.py AAPL --period 1y
python ml/indicators.py AAPL --period 1y --normalize
```

### Train model

```bash
python ml/train.py AAPL --period 1y --model logistic_regression --normalize
python ml/train.py AAPL --period 1y --model random_forest --normalize
```

Saved artifacts are stored in `models/`.

### Predict signal

```bash
python ml/predict.py AAPL --model logistic_regression --period 1y
```

Example output:

```text
Ticker: AAPL
Date: 2026-04-17
Current Price: 270.23
Probability Up: 0.4499
Signal: HOLD
```

### Run backtest

```bash
python ml/backtest.py AAPL --model logistic_regression --period 1y
```

### Start Flask app

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Example Workflow

```bash
python ml/data_fetch.py AAPL --period 1y
python ml/indicators.py AAPL --period 1y --normalize
python ml/train.py AAPL --period 1y --model logistic_regression --normalize
python ml/predict.py AAPL --model logistic_regression --period 1y
python ml/backtest.py AAPL --model logistic_regression --period 1y
python app.py
```

## Outputs

Generated files include:

- `data/<TICKER>_historical.csv`
- `data/<TICKER>_features.csv`
- `data/<TICKER>_backtest.csv`
- `models/<TICKER>_<MODEL>.joblib`
- `models/<TICKER>_<MODEL>_metadata.json`

## Current Limitations

- Uses simple technical indicators and baseline ML models
- Uses only historical price-based features
- Does not include transaction costs, slippage, or position sizing
- Backtesting logic is intentionally simple for learning purposes
- Predictions should be treated as decision support, not financial advice

## Future Improvements

- Add more indicators such as Bollinger Bands, ATR, and Stochastic Oscillator
- Compare multiple models automatically
- Add model performance charts and confusion matrix views
- Include portfolio-level backtesting
- Add user-selectable models from the web UI
- Save and compare results for multiple tickers
- Deploy the Flask app online

## Disclaimer

This project is for educational and research purposes only. It is not financial advice, and it should not be used as the sole basis for making investment decisions.

## Author

Built as an AI + software engineering stock market expert system project in Python.
