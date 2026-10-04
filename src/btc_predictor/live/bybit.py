from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import websockets
from websockets.exceptions import WebSocketException

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LiveBar:
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trade_count: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float

    def raw_values(self):
        return (
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            self.quote_volume,
            self.trade_count,
            self.taker_buy_base_volume,
            self.taker_buy_quote_volume,
        )


class _BarBuilder:
    def __init__(self, second: int, price: float):
        self.second = second
        self.open = self.high = self.low = self.close = price
        self.volume = self.quote = self.buy_base = self.buy_quote = 0.0
        self.count = 0

    def add(self, price: float, quantity: float, taker_side: str) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        quote = price * quantity
        self.volume += quantity
        self.quote += quote
        self.count += 1
        if taker_side.lower() == "buy":
            self.buy_base += quantity
            self.buy_quote += quote

    def finish(self) -> LiveBar:
        return LiveBar(
            self.second * 1000,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            self.quote,
            self.count,
            self.buy_base,
            self.buy_quote,
        )


class BybitSecondBars:
    """Reconnectable Bybit V5 public-trade stream aggregated into UTC 1s bars."""

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        url: str = "wss://stream.bybit.com/v5/public/spot",
        reconnect_max_seconds: float = 30.0,
    ):
        self.symbol = symbol
        self.url = url
        self.reconnect_max_seconds = reconnect_max_seconds

    async def stream(self) -> AsyncIterator[LiveBar]:
        backoff = 1.0
        last_close: float | None = None
        while True:
            builder: _BarBuilder | None = None
            try:
                async with websockets.connect(
                    self.url, ping_interval=None, close_timeout=5, max_queue=4096
                ) as ws:
                    await ws.send(
                        json.dumps({"op": "subscribe", "args": [f"publicTrade.{self.symbol}"]})
                    )
                    backoff = 1.0
                    last_ping = time.monotonic()
                    while True:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            message = json.loads(raw)
                        except TimeoutError:
                            now_second = int(time.time())
                            if builder is not None and now_second > builder.second:
                                bar = builder.finish()
                                last_close = bar.close
                                yield bar
                                for missing in range(builder.second + 1, now_second):
                                    yield LiveBar(
                                        missing * 1000,
                                        last_close,
                                        last_close,
                                        last_close,
                                        last_close,
                                        0.0,
                                        0.0,
                                        0,
                                        0.0,
                                        0.0,
                                    )
                                builder = _BarBuilder(now_second, last_close)
                            if time.monotonic() - last_ping >= 20:
                                await ws.send(json.dumps({"op": "ping"}))
                                last_ping = time.monotonic()
                            continue
                        if not str(message.get("topic", "")).startswith("publicTrade."):
                            continue
                        for trade in message.get("data", []):
                            timestamp_ms = int(trade.get("T", message.get("ts", 0)))
                            second = timestamp_ms // 1000
                            price, quantity = float(trade["p"]), float(trade["v"])
                            if builder is None:
                                builder = _BarBuilder(second, price)
                            elif second > builder.second:
                                bar = builder.finish()
                                last_close = bar.close
                                yield bar
                                for missing in range(builder.second + 1, second):
                                    yield LiveBar(
                                        missing * 1000,
                                        last_close,
                                        last_close,
                                        last_close,
                                        last_close,
                                        0.0,
                                        0.0,
                                        0,
                                        0.0,
                                        0.0,
                                    )
                                builder = _BarBuilder(second, price)
                            elif second < builder.second:
                                continue  # late trade; do not mutate a finalized bar
                            builder.add(price, quantity, str(trade.get("S", "")))
            except asyncio.CancelledError:
                raise
            except (
                OSError,
                TimeoutError,
                WebSocketException,
                json.JSONDecodeError,
                KeyError,
                ValueError,
            ) as exc:
                LOG.warning("Bybit stream disconnected (%s); reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(self.reconnect_max_seconds, backoff * 2.0)
