# Model card: BTCUSDT 25-minute path forecaster

## Intended use

Research and paper-trading forecasts of the next 1,500 one-second BTCUSDT prices. A forecast is refreshed every second and includes 10%, 50%, and 90% paths.

## Not intended for

Autonomous live trading, guaranteed returns, manipulation, or use without venue/latency/cost risk controls. The repository intentionally contains no order endpoint.

## Inputs

Twelve hours of Binance-style one-second OHLCV/trade-activity fields plus 96 deterministic CPU analyser features. Live input is Bybit spot public trades aggregated to the same schema.

## Outputs

Relative log-price paths converted to price using the current venue anchor. The model does not output position size, leverage, stop loss or an order.

## Training/evaluation

Update-count-based continual optimization; labels mature after 1,500 seconds. Validation and testing are contiguous chronological blocks. Candidate selection combines endpoint/path errors, direction, interval calibration and adjacent-forecast instability.

## Material limitations

Regime shift, exchange outages, spread/slippage, basis differences, stablecoin depegs, timestamp gaps, latency, survivorship bias, quantile miscalibration and extreme-event underrepresentation. Highly overlapping one-second samples reduce effective sample size. Backtests do not establish future profitability.

## Release requirements

Populate after training: data manifest hash, exact cutoff, code commit, candidate config, hardware, update count, holdout metrics, paper-trading duration, latency distribution and approver.
