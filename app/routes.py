from __future__ import annotations

from flask import Blueprint, render_template, request
import plotly.graph_objects as go

try:
    from ml.indicators import build_feature_dataset_for_ticker
    from ml.predict import predict_for_ticker
except ModuleNotFoundError:
    from indicators import build_feature_dataset_for_ticker
    from predict import predict_for_ticker


main_bp = Blueprint("main", __name__)


def build_price_chart(feature_df, ticker: str) -> str:
    recent_df = feature_df.tail(90)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=recent_df.index,
            y=recent_df["Close"],
            mode="lines",
            name="Close",
            line={"color": "#0f766e", "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=recent_df.index,
            y=recent_df["MA_20"],
            mode="lines",
            name="MA 20",
            line={"color": "#f59e0b", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=recent_df.index,
            y=recent_df["MA_50"],
            mode="lines",
            name="MA 50",
            line={"color": "#2563eb", "width": 2},
        )
    )

    figure.update_layout(
        title=f"{ticker.upper()} Price and Moving Averages",
        template="plotly_white",
        height=420,
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis_title="Date",
        yaxis_title="Price",
    )
    return figure.to_html(full_html=False, include_plotlyjs="cdn")


@main_bp.route("/", methods=["GET", "POST"])
def index():
    ticker = "AAPL"
    result = None
    chart_html = None
    error = None

    if request.method == "POST":
        ticker = request.form.get("ticker", "AAPL").strip().upper() or "AAPL"
        try:
            result = predict_for_ticker(
                ticker=ticker,
                model_name="logistic_regression",
                period="1y",
            )
            feature_df, _ = build_feature_dataset_for_ticker(
                ticker=ticker,
                period="1y",
                normalize=True,
            )
            chart_html = build_price_chart(feature_df, ticker)
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        ticker=ticker,
        result=result,
        chart_html=chart_html,
        error=error,
    )
