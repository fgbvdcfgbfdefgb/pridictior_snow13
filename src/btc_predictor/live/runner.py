from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from .bybit import BybitSecondBars
from .predictor import LivePredictor

LOG = logging.getLogger(__name__)


async def run(args) -> None:
    predictor = LivePredictor(args.checkpoint, args.device)
    if args.bootstrap:
        count = predictor.bootstrap_from_parquet(args.bootstrap)
        LOG.info("bootstrapped %s historical bars", count)
    feed = BybitSecondBars(args.symbol, args.url)
    async for bar in feed.stream():
        predictor.append(bar.timestamp_ms, bar.raw_values())
        if not predictor.warm:
            print(
                json.dumps({"timestamp_ms": bar.timestamp_ms, "warmup": len(predictor.history)}),
                flush=True,
            )
            continue
        forecast = predictor.predict()
        paths = forecast.pop("paths")
        # CLI emits compact checkpoints; the dashboard consumes the entire in-memory path.
        indices = [0, 59, 299, 899, 1499]
        forecast["forecast"] = {
            str(i + 1): [round(float(x), 2) for x in paths[i]] for i in indices if i < len(paths)
        }
        print(json.dumps(forecast), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bybit live one-second 25-minute predictor")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--bootstrap", type=Path, default=Path("data/processed"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--url", default="wss://stream.bybit.com/v5/public/spot")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
