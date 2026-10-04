# Run with: marimo run notebooks/molab_live.py
# Environment: CHECKPOINT=checkpoints/best.pt DATA_ROOT=data/processed marimo run ...

import marimo

__generated_with = "0.11.0"
app = marimo.App(width="full", app_title="BTC 25-Minute Forecast")


@app.cell
def _():
    import asyncio
    import os
    from collections import deque
    from datetime import datetime, timedelta, timezone

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    from btc_predictor.live.bybit import BybitSecondBars
    from btc_predictor.live.predictor import LivePredictor

    return (
        BybitSecondBars,
        LivePredictor,
        asyncio,
        datetime,
        deque,
        go,
        mo,
        np,
        os,
        timedelta,
        timezone,
    )


@app.cell
def _(go):
    # FigureWidget is created exactly once. Its traces are mutated in-place; no new
    # chart is appended on each tick.
    figure = go.FigureWidget()
    figure.add_scatter(name="BTCUSDT actual", mode="lines", line={"color": "#3861FB", "width": 2})
    figure.add_scatter(
        name="25m median forecast", mode="lines", line={"color": "#F7931A", "width": 2}
    )
    figure.add_scatter(name="90% high", mode="lines", line={"width": 0}, showlegend=False)
    figure.add_scatter(
        name="80% interval",
        mode="lines",
        line={"width": 0},
        fill="tonexty",
        fillcolor="rgba(56,97,251,0.16)",
        showlegend=True,
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D1421",
        plot_bgcolor="#0D1421",
        font={"family": "Inter, Arial, sans-serif", "color": "#A6B0C3"},
        margin={"l": 55, "r": 25, "t": 65, "b": 45},
        height=650,
        title={"text": "Bitcoin  •  BTC/USDT", "font": {"size": 23, "color": "#F8FAFD"}},
        xaxis={"gridcolor": "#222D3D", "rangeslider": {"visible": False}},
        yaxis={"gridcolor": "#222D3D", "side": "right", "tickprefix": "$", "fixedrange": False},
        legend={"orientation": "h", "y": 1.08, "x": 0.52},
        hovermode="x unified",
    )
    return (figure,)


@app.cell
def _(LivePredictor, os):
    checkpoint = os.environ.get("CHECKPOINT", "checkpoints/best.pt")
    data_root = os.environ.get("DATA_ROOT", "data/processed")
    predictor = LivePredictor(checkpoint, os.environ.get("DEVICE", "cuda"))
    bootstrap_count = predictor.bootstrap_from_parquet(data_root)
    return bootstrap_count, predictor


@app.cell
def _(BybitSecondBars, asyncio, datetime, deque, figure, predictor, timedelta, timezone):
    actual_time = deque(maxlen=3600)
    actual_price = deque(maxlen=3600)
    status = {"message": "Connecting to Bybit…", "updates": 0}

    async def update_forever():
        feed = BybitSecondBars()
        async for bar in feed.stream():
            predictor.append(bar.timestamp_ms, bar.raw_values())
            now = datetime.fromtimestamp(bar.timestamp_ms / 1000, tz=timezone.utc)
            actual_time.append(now)
            actual_price.append(bar.close)
            with figure.batch_update():
                figure.data[0].x = tuple(actual_time)
                figure.data[0].y = tuple(actual_price)
                if predictor.warm:
                    result = await asyncio.to_thread(predictor.predict)
                    paths = result["paths"]
                    future = tuple(now + timedelta(seconds=i + 1) for i in range(len(paths)))
                    figure.data[1].x = future
                    figure.data[1].y = paths[:, 1]
                    # High is drawn first; low fills upward to it.
                    figure.data[2].x = future
                    figure.data[2].y = paths[:, 2]
                    figure.data[3].x = future
                    figure.data[3].y = paths[:, 0]
                    figure.layout.title.text = (
                        f"Bitcoin  •  BTC/USDT  ${bar.close:,.2f}  •  25m ${paths[-1, 1]:,.2f}"
                    )
                    status["message"] = "LIVE • prediction updated in place"
                else:
                    status["message"] = (
                        f"Warming up {len(predictor.history):,}/{predictor.history_size:,}"
                    )
                status["updates"] += 1

    task = asyncio.create_task(update_forever())
    return actual_price, actual_time, status, task


@app.cell
def _(bootstrap_count, figure, mo):
    mo.vstack(
        [
            mo.md(
                f"## Live model dashboard\n**Warm-start bars:** {bootstrap_count:,}  ·  **Feed:** Bybit public spot trades  ·  **Mode:** forecast only"
            ),
            figure,
            mo.md(
                "> The chart is a single persistent Plotly `FigureWidget`; trace arrays are updated with `batch_update()`. This is not an execution interface and does not place orders."
            ),
        ]
    )


if __name__ == "__main__":
    app.run()
