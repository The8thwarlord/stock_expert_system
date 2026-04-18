import argparse
from pathlib import Path

import pandas as pd

try:
    from ml.data_fetch import DATA_DIR, fetch_and_save_stock_data
except ModuleNotFoundError:
    from data_fetch import DATA_DIR, fetch_and_save_stock_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURED_DATA_DIR = DATA_DIR
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
FEATURE_COLUMNS = [
    "MA_20",
    "MA_50",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "Daily_Return",
]


def moving_average(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate a simple moving average for a price series.
    """
    return series.rolling(window=window, min_periods=window).mean()


def relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the RSI momentum indicator.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD, signal line, and histogram.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "MACD": macd_line,
            "MACD_Signal": signal_line,
            "MACD_Hist": histogram,
        },
        index=close.index,
    )


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep expected price columns, sort by date, and remove incomplete rows.
    """
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.sort_index()
    cleaned_df = cleaned_df.dropna(subset=PRICE_COLUMNS)
    return cleaned_df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the first set of technical indicators used by the expert system.
    """
    featured_df = clean_price_data(df)
    featured_df["MA_20"] = moving_average(featured_df["Close"], window=20)
    featured_df["MA_50"] = moving_average(featured_df["Close"], window=50)
    featured_df["RSI_14"] = relative_strength_index(featured_df["Close"], window=14)

    macd_df = macd(featured_df["Close"])
    featured_df = pd.concat([featured_df, macd_df], axis=1)

    # Daily return is a simple, useful feature for the first ML version.
    featured_df["Daily_Return"] = featured_df["Close"].pct_change()
    return featured_df


def normalize_feature_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Apply simple min-max scaling to selected columns.

    This keeps the implementation beginner-friendly while preparing the data
    for future ML experiments.
    """
    normalized_df = df.copy()
    for column in columns:
        if column not in normalized_df.columns:
            continue

        column_min = normalized_df[column].min()
        column_max = normalized_df[column].max()
        if pd.isna(column_min) or pd.isna(column_max) or column_min == column_max:
            normalized_df[f"{column}_Norm"] = 0.0
            continue

        normalized_df[f"{column}_Norm"] = (
            normalized_df[column] - column_min
        ) / (column_max - column_min)

    return normalized_df


def prepare_feature_dataset(df: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
    """
    Build a cleaned feature dataset from raw OHLCV stock data.
    """
    featured_df = add_technical_indicators(df)

    # Drop rows where rolling indicators are still warming up.
    featured_df = featured_df.dropna(subset=FEATURE_COLUMNS)

    if normalize:
        featured_df = normalize_feature_columns(featured_df, FEATURE_COLUMNS)

    return featured_df


def save_feature_dataset(
    df: pd.DataFrame,
    ticker: str,
    output_dir: str | Path = FEATURED_DATA_DIR,
) -> Path:
    """
    Save the enriched dataset with indicators to CSV.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"{ticker.strip().upper()}_features.csv"
    df.to_csv(file_path)
    return file_path


def build_feature_dataset_for_ticker(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    normalize: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """
    Fetch stock data, calculate indicators, and save the enriched dataset.
    """
    raw_df, _ = fetch_and_save_stock_data(
        ticker=ticker,
        period=period,
        interval=interval,
        start=start,
        end=end,
    )
    feature_df = prepare_feature_dataset(raw_df, normalize=normalize)
    saved_path = save_feature_dataset(feature_df, ticker=ticker)
    return feature_df, saved_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a feature dataset with technical indicators."
    )
    parser.add_argument("ticker", help="Stock ticker symbol, for example AAPL or RELIANCE.NS")
    parser.add_argument("--period", default="1y", help="Yahoo Finance period, default: 1y")
    parser.add_argument("--interval", default="1d", help="Yahoo Finance interval, default: 1d")
    parser.add_argument("--start", help="Optional start date in YYYY-MM-DD format")
    parser.add_argument("--end", help="Optional end date in YYYY-MM-DD format")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Add normalized versions of the feature columns",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    features_df, file_path = build_feature_dataset_for_ticker(
        ticker=args.ticker,
        period=args.period,
        interval=args.interval,
        start=args.start,
        end=args.end,
        normalize=args.normalize,
    )

    print(f"Built feature dataset with {len(features_df)} rows for {args.ticker.upper()}")
    print(features_df.tail())
    print(f"Saved feature data to: {file_path}")
