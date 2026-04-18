import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def fetch_stock_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol, for example "AAPL" or "RELIANCE.NS".
        period: Predefined Yahoo Finance history window such as "1y" or "6mo".
        interval: Data granularity such as "1d" or "1h".
        start: Optional start date in YYYY-MM-DD format.
        end: Optional end date in YYYY-MM-DD format.

    Returns:
        Cleaned pandas DataFrame indexed by Date.
    """
    cleaned_ticker = ticker.strip().upper()
    if not cleaned_ticker:
        raise ValueError("Ticker symbol cannot be empty.")

    download_kwargs = {
        "tickers": cleaned_ticker,
        "interval": interval,
        "progress": False,
        "auto_adjust": False,
    }

    # Use either a preset period or an explicit date range.
    if start or end:
        download_kwargs["start"] = start
        download_kwargs["end"] = end
    else:
        download_kwargs["period"] = period

    df = yf.download(**download_kwargs)

    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{cleaned_ticker}'. Check the symbol and date range."
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Yahoo Finance response for '{cleaned_ticker}' is missing columns: {missing_columns}"
        )

    df = df.loc[:, REQUIRED_COLUMNS].copy()
    df.columns.name = None
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return df


def save_stock_data(
    df: pd.DataFrame,
    ticker: str,
    output_dir: str | Path = DATA_DIR,
) -> Path:
    """
    Save historical stock data to the project's data directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"{ticker.strip().upper()}_historical.csv"
    csv_path = output_path / filename
    df.to_csv(csv_path)
    return csv_path


def fetch_and_save_stock_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    output_dir: str | Path = DATA_DIR,
) -> tuple[pd.DataFrame, Path]:
    """
    Convenience helper that downloads stock data and saves it as CSV.
    """
    stock_df = fetch_stock_data(
        ticker=ticker,
        period=period,
        interval=interval,
        start=start,
        end=end,
    )
    saved_path = save_stock_data(stock_df, ticker=ticker, output_dir=output_dir)
    return stock_df, saved_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download historical stock data from Yahoo Finance."
    )
    parser.add_argument("ticker", help="Stock ticker symbol, for example AAPL or RELIANCE.NS")
    parser.add_argument("--period", default="1y", help="Yahoo Finance period, default: 1y")
    parser.add_argument("--interval", default="1d", help="Yahoo Finance interval, default: 1d")
    parser.add_argument("--start", help="Optional start date in YYYY-MM-DD format")
    parser.add_argument("--end", help="Optional end date in YYYY-MM-DD format")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    downloaded_df, file_path = fetch_and_save_stock_data(
        ticker=args.ticker,
        period=args.period,
        interval=args.interval,
        start=args.start,
        end=args.end,
    )

    print(f"Downloaded {len(downloaded_df)} rows for {args.ticker.upper()}")
    print(downloaded_df.tail())
    print(f"Saved historical data to: {file_path}")
