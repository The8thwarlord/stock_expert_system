import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

try:
    from ml.indicators import build_feature_dataset_for_ticker
except ModuleNotFoundError:
    from indicators import build_feature_dataset_for_ticker


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def resolve_model_paths(ticker: str, model_name: str) -> tuple[Path, Path]:
    """
    Resolve the trained model and metadata paths for a ticker.
    """
    ticker_name = ticker.strip().upper()
    model_path = MODELS_DIR / f"{ticker_name}_{model_name}.joblib"
    metadata_path = MODELS_DIR / f"{ticker_name}_{model_name}_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train the model first using ml/train.py."
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}. Train the model first using ml/train.py."
        )

    return model_path, metadata_path


def load_model_artifacts(ticker: str, model_name: str) -> tuple[object, dict]:
    """
    Load the trained model and its metadata.
    """
    model_path, metadata_path = resolve_model_paths(ticker, model_name)
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text())
    return model, metadata


def get_latest_feature_row(
    ticker: str,
    model_feature_columns: list[str],
    period: str = "1y",
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Build the latest feature row matching the saved model columns.
    """
    normalize = any(column.endswith("_Norm") for column in model_feature_columns)
    feature_df, _ = build_feature_dataset_for_ticker(
        ticker=ticker,
        period=period,
        interval=interval,
        start=start,
        end=end,
        normalize=normalize,
    )

    missing_columns = [column for column in model_feature_columns if column not in feature_df.columns]
    if missing_columns:
        raise ValueError(f"Missing model feature columns in latest dataset: {missing_columns}")

    latest_row = feature_df.iloc[-1].copy()
    return latest_row, feature_df


def predict_price_movement(model, latest_row: pd.Series, feature_columns: list[str]) -> dict:
    """
    Run the ML model on the latest data row.
    """
    latest_features = pd.DataFrame([latest_row[feature_columns]], columns=feature_columns)
    predicted_class = int(model.predict(latest_features)[0])

    probability_up = None
    if hasattr(model, "predict_proba"):
        class_probabilities = model.predict_proba(latest_features)[0]
        probability_up = float(class_probabilities[1])

    return {
        "predicted_class": predicted_class,
        "probability_up": probability_up,
    }


def generate_signal(latest_row: pd.Series, ml_prediction: dict) -> tuple[str, list[str]]:
    """
    Combine ML output with simple rule-based trading logic.
    """
    reasons: list[str] = []
    probability_up = ml_prediction["probability_up"]
    predicted_class = ml_prediction["predicted_class"]

    rsi_value = float(latest_row["RSI_14"])
    close_price = float(latest_row["Close"])
    ma_20 = float(latest_row["MA_20"])
    macd_value = float(latest_row["MACD"])
    macd_signal = float(latest_row["MACD_Signal"])

    trend_up = close_price > ma_20
    momentum_up = macd_value > macd_signal

    if probability_up is not None:
        if probability_up >= 0.55:
            signal = "BUY"
            reasons.append(f"model probability of an upward move is {probability_up:.2f}")
        elif probability_up <= 0.45:
            signal = "SELL"
            reasons.append(f"model probability of an upward move is only {probability_up:.2f}")
        else:
            signal = "HOLD"
            reasons.append(f"model confidence is neutral at {probability_up:.2f}")
    else:
        signal = "BUY" if predicted_class == 1 else "SELL"
        reasons.append(f"model predicted {'upward' if predicted_class == 1 else 'downward'} movement")

    if trend_up:
        reasons.append("price is above the 20-day moving average")
    else:
        reasons.append("price is below the 20-day moving average")

    if momentum_up:
        reasons.append("MACD is above the signal line")
    else:
        reasons.append("MACD is below the signal line")

    if rsi_value > 70:
        reasons.append(f"RSI is high at {rsi_value:.2f}, so buying is risky")
        if signal == "BUY":
            signal = "HOLD"
    elif rsi_value < 30:
        reasons.append(f"RSI is low at {rsi_value:.2f}, which supports a buy setup")
        if signal == "SELL":
            signal = "HOLD"
        elif signal == "HOLD" and (trend_up or momentum_up):
            signal = "BUY"
    else:
        reasons.append(f"RSI is neutral at {rsi_value:.2f}")

    if signal == "BUY" and not trend_up and not momentum_up:
        signal = "HOLD"
        reasons.append("trend confirmation is weak, so buy was softened to hold")

    if signal == "SELL" and trend_up and momentum_up:
        signal = "HOLD"
        reasons.append("trend is still constructive, so sell was softened to hold")

    return signal, reasons


def build_prediction_explanation(signal: str, reasons: list[str]) -> str:
    """
    Build a short natural-language explanation.
    """
    if not reasons:
        return f"{signal} because the model signaled {signal.lower()}."

    return f"{signal} because " + "; ".join(reasons) + "."


def predict_for_ticker(
    ticker: str,
    model_name: str = "logistic_regression",
    period: str = "1y",
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """
    Load a trained model and produce a trading signal for the latest row.
    """
    model, metadata = load_model_artifacts(ticker, model_name)
    feature_columns = metadata["feature_columns"]

    latest_row, feature_df = get_latest_feature_row(
        ticker=ticker,
        model_feature_columns=feature_columns,
        period=period,
        interval=interval,
        start=start,
        end=end,
    )

    ml_prediction = predict_price_movement(model, latest_row, feature_columns)
    signal, reasons = generate_signal(latest_row, ml_prediction)
    explanation = build_prediction_explanation(signal, reasons)

    return {
        "ticker": ticker.strip().upper(),
        "date": str(feature_df.index[-1].date()),
        "current_price": float(latest_row["Close"]),
        "signal": signal,
        "probability_up": ml_prediction["probability_up"],
        "rsi": float(latest_row["RSI_14"]),
        "ma_20": float(latest_row["MA_20"]),
        "macd": float(latest_row["MACD"]),
        "macd_signal": float(latest_row["MACD_Signal"]),
        "explanation": explanation,
        "reasons": reasons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict BUY, SELL, or HOLD using a trained stock model."
    )
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = predict_for_ticker(
        ticker=args.ticker,
        model_name=args.model,
        period=args.period,
        interval=args.interval,
        start=args.start,
        end=args.end,
    )

    print(f"Ticker: {result['ticker']}")
    print(f"Date: {result['date']}")
    print(f"Current Price: {result['current_price']:.2f}")
    if result["probability_up"] is not None:
        print(f"Probability Up: {result['probability_up']:.4f}")
    print(f"Signal: {result['signal']}")
    print(f"Explanation: {result['explanation']}")
