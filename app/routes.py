from __future__ import annotations

from flask import Blueprint, render_template, request
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from ml.indicators import build_feature_dataset_for_ticker
    from ml.predict import predict_for_ticker
except ModuleNotFoundError:
    from indicators import build_feature_dataset_for_ticker
    from predict import predict_for_ticker


main_bp = Blueprint("main", __name__)


def build_market_dashboard(feature_df, ticker: str) -> str:
    recent_df = feature_df.tail(90)

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.56, 0.20, 0.24],
        subplot_titles=(
            f"{ticker.upper()} Price and Trend",
            "RSI 14",
            "MACD",
        ),
    )

    figure.add_trace(
        go.Scatter(
            x=recent_df.index,
            y=recent_df["Close"],
            mode="lines",
            name="Close",
            line={"color": "#0f766e", "width": 3},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=recent_df.index,
            y=recent_df["MA_20"],
            mode="lines",
            name="MA 20",
            line={"color": "#f59e0b", "width": 2},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=recent_df.index,
            y=recent_df["MA_50"],
            mode="lines",
            name="MA 50",
            line={"color": "#2563eb", "width": 2},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=recent_df.index,
            y=recent_df["RSI_14"],
            mode="lines",
            name="RSI 14",
            line={"color": "#7c3aed", "width": 2.5},
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=recent_df.index,
            y=recent_df["MACD"],
            mode="lines",
            name="MACD",
            line={"color": "#111827", "width": 2},
        ),
        row=3,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=recent_df.index,
            y=recent_df["MACD_Signal"],
            mode="lines",
            name="MACD Signal",
            line={"color": "#dc2626", "width": 2},
        ),
        row=3,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=recent_df.index,
            y=recent_df["MACD_Hist"],
            name="MACD Hist",
            marker={"color": "#93c5fd"},
            opacity=0.5,
        ),
        row=3,
        col=1,
    )

    figure.update_layout(
        template="plotly_white",
        height=760,
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        paper_bgcolor="#fffdf8",
        plot_bgcolor="#fffdf8",
    )
    figure.add_hline(y=70, line_dash="dot", line_color="#dc2626", row=2, col=1)
    figure.add_hline(y=30, line_dash="dot", line_color="#059669", row=2, col=1)
    figure.update_yaxes(title_text="Price", row=1, col=1)
    figure.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    figure.update_yaxes(title_text="MACD", row=3, col=1)
    figure.update_xaxes(title_text="Date", row=3, col=1)
    return figure.to_html(full_html=False, include_plotlyjs="cdn")


def build_highlights(result: dict) -> list[dict]:
    probability_value = result["probability_up"]
    probability_label = "N/A" if probability_value is None else f"{probability_value:.2f}"

    return [
        {"label": "Current Price", "value": f"{result['current_price']:.2f}"},
        {"label": "Probability Up", "value": probability_label},
        {"label": "RSI 14", "value": f"{result['rsi']:.2f}"},
        {"label": "MA 20", "value": f"{result['ma_20']:.2f}"},
        {"label": "MACD", "value": f"{result['macd']:.4f}"},
        {"label": "MACD Signal", "value": f"{result['macd_signal']:.4f}"},
    ]


@main_bp.route("/", methods=["GET", "POST"])
def index():
    ticker = "AAPL"
    result = None
    chart_html = None
    highlights = []
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
            chart_html = build_market_dashboard(feature_df, ticker)
            highlights = build_highlights(result)
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        ticker=ticker,
        result=result,
        chart_html=chart_html,
        highlights=highlights,
        error=error,
    )
