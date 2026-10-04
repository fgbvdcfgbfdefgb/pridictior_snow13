# Architecture and time semantics

## Forecast contract

At issue second `t`, the model receives only observations with timestamps `<= t`:

- raw one-second bars from `t-43,199` through `t`;
- one 96-value Market Analyser vector computed from that same closed interval.

It emits three quantile paths for `t+1` through `t+1,500`. The price path is represented as log returns from price at `t`; live inference anchors those returns to the current venue price.

## Delayed reward

A prediction issued at `t` is not eligible for a complete-path reward until market time `t+1,500`. `ContinualReplayBuffer` retains exactly `history+horizon+1` rows. When its newest row is `t+1,500`, it releases the history ending at `t` and the target beginning at `t+1`. This is test-covered.

The high-throughput DDP trainer draws windows from immutable chronological train partitions but uses the same boundary. It is update-count based; there is no epoch as a training-control concept.

## Shared CPU analyser

There is one analyser definition, never one learned analyser per GPU candidate. It computes 10 measurements over each of nine horizons (5 seconds to 12 hours) plus six candlestick/activity measurements:

- log return, realized volatility, range and close z-score;
- base/quote-volume z-scores and trade intensity;
- taker imbalance, trend slope and max drawdown;
- body/wicks and log activity.

This gives 96 deterministic, clipped, finite values. Model candidates also receive stationary raw channels: four relative log prices, normalized base/quote activity, normalized trade count and two taker imbalances.

## Predictor candidates

### Multiscale Patch Transformer

- Long branch: 12 hours sampled every 60 seconds, patched by six samples.
- Medium branch: latest 3 hours sampled every 10 seconds, patched by ten samples.
- Recent branch: latest 30 minutes at native resolution, patched by 15 samples.
- A feature token and classification token are jointly encoded.

### Dilated TCN

A causal residual convolutional stack operates on a 12-hour multiscale sequence. It is useful when Transformer latency or memory is unacceptable.

### GRU-attention

A stacked GRU encodes the same multiscale sequence; the analyser vector becomes an attention query over the recurrent states.

All heads predict sparse knots for each quantile and linearly interpolate to 1,500 seconds. This architectural low-pass prior reduces jagged paths without freezing legitimate market moves.

## Stability objective

The loss combines:

1. endpoint Huber loss;
2. full-path Huber loss;
3. pinball quantile loss;
4. first-difference and second-difference path losses;
5. adjacent-issue temporal consistency in absolute log-price space;
6. direction loss;
7. quantile-crossing penalty.

Stability is measured, not assumed: the walk-forward evaluator aligns overlapping forecasts issued one second apart and reports the mean discrepancy in basis points.

## Multi-GPU execution

Final training uses data-parallel DDP. Every rank gets a distinct worker/file shard, computes gradients on a full model replica, and synchronizes updates. The fast tournament instead puts an independent architecture on each GPU. These modes answer different questions and are not mixed in final model comparison.

## Model selection

Only chronological validation metrics choose the winner. The weighted score is declared in configuration. A final untouched chronological test and forward paper-trading period are required after selection. All model checkpoints and evaluation metadata are retained for audit.
