import argparse
from pathlib import Path

import pandas as pd

try:
    from ml.predict import generate_signal, load_model_artifacts
    from ml.indicators import build_feature_dataset_for_ticker
except ModuleNotFoundError:
    from predict import generate_signal, load_model_artifacts
    from indicators import build_feature_dataset_for_ticker


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def get_prediction_for_row(model, row: pd.Series, feature_columns: list[str]) -> dict:
    feature_frame = pd.DataFrame([row[feature_columns]], columns=feature_columns)
    predicted_class = int(model.predict(feature_frame)[0])

    probability_up = None
    if hasattr(model, "predict_proba"):
        probability_up = float(model.predict_proba(feature_frame)[0][1])

    return {
        "predicted_class": predicted_class,
        "probability_up": probability_up,
    }


def run_backtest(
    ticker: str,
    model_name: str = "logistic_regression",
    period: str = "1y",
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    initial_cash: float = 10000.0,
) -> tuple[pd.DataFrame, dict]:
    model, metadata = load_model_artifacts(ticker, model_name)
    feature_columns = metadata["feature_columns"]
    normalize = any(column.endswith("_Norm") for column in feature_columns)

    feature_df, _ = build_feature_dataset_for_ticker(
        ticker=ticker,
        period=period,
        interval=interval,
        start=start,
        end=end,
        normalize=normalize,
    )

    backtest_df = feature_df.copy()
    backtest_df["Next_Close"] = backtest_df["Close"].shift(-1)
    backtest_df = backtest_df.dropna(subset=["Next_Close"]).copy()

    signals = []
    positions = []
    strategy_returns = []
    market_returns = []
    probabilities = []
    explanations = []

    current_position = 0

    for _, row in backtest_df.iterrows():
        ml_prediction = get_prediction_for_row(model, row, feature_columns)
        signal, reasons = generate_signal(row, ml_prediction)

        if signal == "BUY":
            current_position = 1
        elif signal == "SELL":
            current_position = 0

        next_day_return = (float(row["Next_Close"]) - float(row["Close"])) / float(row["Close"])
        strategy_return = current_position * next_day_return

        signals.append(signal)
        positions.append(current_position)
        strategy_returns.append(strategy_return)
        market_returns.append(next_day_return)
        probabilities.append(ml_prediction["probability_up"])
        explanations.append("; ".join(reasons))

    backtest_df["Signal"] = signals
    backtest_df["Position"] = positions
    backtest_df["Probability_Up"] = probabilities
    backtest_df["Market_Return"] = market_returns
    backtest_df["Strategy_Return"] = strategy_returns
    backtest_df["Market_Equity"] = initial_cash * (1 + backtest_df["Market_Return"]).cumprod()
    backtest_df["Strategy_Equity"] = initial_cash * (1 + backtest_df["Strategy_Return"]).cumprod()
    backtest_df["Explanation"] = explanations

    total_return = (backtest_df["Strategy_Equity"].iloc[-1] / initial_cash) - 1
    buy_and_hold_return = (backtest_df["Market_Equity"].iloc[-1] / initial_cash) - 1
    trade_count = int((backtest_df["Position"].diff().fillna(backtest_df["Position"]) != 0).sum())
    win_rate = float((backtest_df["Strategy_Return"] > 0).mean())

    summary = {
        "ticker": ticker.strip().upper(),
        "model_name": model_name,
        "rows_tested": int(len(backtest_df)),
        "initial_cash": initial_cash,
        "final_strategy_equity": float(backtest_df["Strategy_Equity"].iloc[-1]),
        "final_market_equity": float(backtest_df["Market_Equity"].iloc[-1]),
        "strategy_total_return": float(total_return),
        "buy_and_hold_return": float(buy_and_hold_return),
        "trade_count": trade_count,
        "win_rate": win_rate,
    }

    return backtest_df, summary


def save_backtest_results(df: pd.DataFrame, ticker: str) -> Path:
    output_path = DATA_DIR / f"{ticker.strip().upper()}_backtest.csv"
    df.to_csv(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the stock trading expert system.")
    parser.add_argument("ticker", help="Stock ticker symbol, for example AAPL or RELIANCE.NS")
    parser.add_argument(
        "--model",
        default="logistic_regression",
        help="Model name used during training, default: logistic_regression",
    )
    parser.add_argument("--period", default="1y", help="Yahoo Finance period, default: 1y")
    parser.add_argument("--interval", default="1d", help="Yahoo Finance interval, default: 1d")
    parser.add_argument("--start", help="Optional start date in YYYY-MM-DD format")
    parser.add_argument("--end", help="Optional end date in YYYY-MM-DD format")
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=10000.0,
        help="Initial portfolio cash, default: 10000",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results_df, summary = run_backtest(
        ticker=args.ticker,
        model_name=args.model,
        period=args.period,
        interval=args.interval,
        start=args.start,
        end=args.end,
        initial_cash=args.initial_cash,
    )
    saved_path = save_backtest_results(results_df, args.ticker)

    print(f"Ticker: {summary['ticker']}")
    print(f"Rows Tested: {summary['rows_tested']}")
    print(f"Trades: {summary['trade_count']}")
    print(f"Win Rate: {summary['win_rate']:.4f}")
    print(f"Strategy Return: {summary['strategy_total_return']:.4%}")
    print(f"Buy and Hold Return: {summary['buy_and_hold_return']:.4%}")
    print(f"Final Strategy Equity: {summary['final_strategy_equity']:.2f}")
    print(f"Final Market Equity: {summary['final_market_equity']:.2f}")
    print(f"Saved backtest to: {saved_path}")
