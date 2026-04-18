import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

try:
    from ml.indicators import FEATURE_COLUMNS, build_feature_dataset_for_ticker
except ModuleNotFoundError:
    from indicators import FEATURE_COLUMNS, build_feature_dataset_for_ticker


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SUPPORTED_MODELS = {"logistic_regression", "random_forest"}


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the binary target column based on next-day close movement.

    Target definition:
    1 = next day's closing price is higher than today's close
    0 = next day's closing price is lower than or equal to today's close
    """
    labeled_df = df.copy()
    labeled_df["Next_Close"] = labeled_df["Close"].shift(-1)
    labeled_df["Target"] = (labeled_df["Next_Close"] > labeled_df["Close"]).astype(int)
    labeled_df = labeled_df.dropna(subset=["Next_Close"])
    return labeled_df


def get_feature_columns(df: pd.DataFrame, use_normalized: bool = False) -> list[str]:
    """
    Select which feature columns will be used for model training.
    """
    if use_normalized:
        normalized_columns = [f"{column}_Norm" for column in FEATURE_COLUMNS]
        available_normalized_columns = [
            column for column in normalized_columns if column in df.columns
        ]
        if len(available_normalized_columns) == len(normalized_columns):
            return available_normalized_columns

    return FEATURE_COLUMNS.copy()


def split_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a chronological train/test split to reduce time-series leakage.
    """
    if df.empty:
        raise ValueError("The dataset is empty after preprocessing.")

    if len(df) < 10:
        raise ValueError("Need at least 10 rows to train a basic model.")

    split_index = max(int(len(df) * (1 - test_size)), 1)
    if split_index >= len(df):
        split_index = len(df) - 1

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    x_train = train_df[feature_columns]
    y_train = train_df["Target"]
    x_test = test_df[feature_columns]
    y_test = test_df["Target"]
    return x_train, x_test, y_train, y_test


def build_model(model_name: str):
    """
    Create the requested scikit-learn model.
    """
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=42)

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=4,
            random_state=42,
        )

    raise ValueError(f"Unsupported model '{model_name}'. Choose from {SUPPORTED_MODELS}.")


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the trained model on the holdout set.
    """
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy,
        "classification_report": report,
        "predictions": predictions.tolist(),
    }


def save_training_artifacts(
    model,
    ticker: str,
    model_name: str,
    feature_columns: list[str],
    metrics: dict,
) -> tuple[Path, Path]:
    """
    Save the trained model and a small metadata file for later prediction use.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ticker_name = ticker.strip().upper()
    model_path = MODELS_DIR / f"{ticker_name}_{model_name}.joblib"
    metadata_path = MODELS_DIR / f"{ticker_name}_{model_name}_metadata.json"

    joblib.dump(model, model_path)

    metadata = {
        "ticker": ticker_name,
        "model_name": model_name,
        "feature_columns": feature_columns,
        "accuracy": metrics["accuracy"],
        "classification_report": metrics["classification_report"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return model_path, metadata_path


def train_model_for_ticker(
    ticker: str,
    model_name: str = "logistic_regression",
    period: str = "1y",
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    normalize: bool = False,
    test_size: float = 0.2,
) -> dict:
    """
    Build features, train the model, evaluate it, and save the artifacts.
    """
    feature_df, feature_path = build_feature_dataset_for_ticker(
        ticker=ticker,
        period=period,
        interval=interval,
        start=start,
        end=end,
        normalize=normalize,
    )

    labeled_df = create_labels(feature_df)
    feature_columns = get_feature_columns(labeled_df, use_normalized=normalize)

    model_ready_df = labeled_df.dropna(subset=feature_columns + ["Target"]).copy()
    x_train, x_test, y_train, y_test = split_dataset(
        model_ready_df,
        feature_columns=feature_columns,
        test_size=test_size,
    )

    model = build_model(model_name)
    model.fit(x_train, y_train)

    metrics = evaluate_model(model, x_test, y_test)
    model_path, metadata_path = save_training_artifacts(
        model=model,
        ticker=ticker,
        model_name=model_name,
        feature_columns=feature_columns,
        metrics=metrics,
    )

    return {
        "ticker": ticker.strip().upper(),
        "feature_path": feature_path,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "feature_columns": feature_columns,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "accuracy": metrics["accuracy"],
        "classification_report": metrics["classification_report"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a stock movement classifier from technical indicators."
    )
    parser.add_argument("ticker", help="Stock ticker symbol, for example AAPL or RELIANCE.NS")
    parser.add_argument(
        "--model",
        default="logistic_regression",
        choices=sorted(SUPPORTED_MODELS),
        help="Model type to train",
    )
    parser.add_argument("--period", default="1y", help="Yahoo Finance period, default: 1y")
    parser.add_argument("--interval", default="1d", help="Yahoo Finance interval, default: 1d")
    parser.add_argument("--start", help="Optional start date in YYYY-MM-DD format")
    parser.add_argument("--end", help="Optional end date in YYYY-MM-DD format")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Train using normalized feature columns when available",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for testing, default: 0.2",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train_model_for_ticker(
        ticker=args.ticker,
        model_name=args.model,
        period=args.period,
        interval=args.interval,
        start=args.start,
        end=args.end,
        normalize=args.normalize,
        test_size=args.test_size,
    )

    print(f"Trained {args.model} model for {result['ticker']}")
    print(f"Training rows: {result['train_rows']}")
    print(f"Testing rows: {result['test_rows']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Feature dataset: {result['feature_path']}")
    print(f"Saved model to: {result['model_path']}")
    print(f"Saved metadata to: {result['metadata_path']}")
