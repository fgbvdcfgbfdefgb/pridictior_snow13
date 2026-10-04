# Bitcoin 25-Minute Price Predictor

A leakage-safe research pipeline that forecasts the complete **BTCUSDT price path for the next 1,500 seconds (25 minutes)** and refreshes that forecast every second.

> **Status:** implementation, tests, a real-source sample, reproducible historical-data ingestion, offline Snowflake entry point, and MoLab live dashboard are included. Full 2020–present shards and trained checkpoints are large Git LFS artifacts; generate/push them using the documented commands. No pretrained model is represented as production-ready.

## Design at a glance

| Part | Runtime | Role |
|---|---:|---|
| Binance archive downloader | CPU/network, before offline training | Acquires canonical spot BTCUSDT 1-second klines from 2020-01-01 through the last finalized UTC day |
| Historical market simulator | CPU | Replays rows in strict timestamp order without exposing future labels |
| Market Analyser | CPU | Produces one deterministic 96-signal vector shared by every candidate |
| Price Predictor | GPU | Consumes 12 hours of multiscale 1-second history plus analyser signals and outputs 1,500 prices at 10/50/90% quantiles |
| Continual reward ledger | CPU/GPU | Releases a forecast's label only after all 1,500 future seconds have occurred |
| Walk-forward evaluator | GPU | Compares every second's forecast with stored market data and measures accuracy, calibration, direction and forecast-update stability |
| Bybit runner | CPU + GPU | Aggregates public spot trades to one-second bars and refreshes inference every second |
| Marimo dashboard | MoLab | Mutates one persistent Plotly `FigureWidget`, styled in a CoinMarketCap-like dark theme |

## Key properties

- **No epoch loop.** Training is bounded by optimizer update count. Offline data is an endless sharded stream, and chronological replay uses a real 1,500-second delayed reward.
- **No label leakage.** Inputs stop at issue time `t`; targets are `t+1 … t+1500`; walk-forward splits are chronological.
- **Stable forecasts.** Sparse forecast knots are linearly interpolated, and the objective penalizes path error, first/second differences, adjacent-issue inconsistency, quantile crossing, and endpoint error.
- **Multi-GPU.** Final candidates use PyTorch DDP via `torchrun`. A fast tournament can run one different architecture per GPU. All checkpoints are retained.
- **Hardware-aware.** `btc-hardware` detects CUDA devices, per-device VRAM, BF16, CPU and RAM before resolving model width, depth, microbatch, accumulation, and loader workers. Four 24 GB A10s select the 768-wide/12-layer profile.
- **Offline after staging.** Snowflake code makes no data-network calls. Dataset shards, repository code, and an optional wheelhouse can be staged before egress is disabled.

## Repository layout

```text
configs/                 default and candidate configurations
data/                    schema, sample, manifests, Git LFS partitions
notebooks/molab_live.py  live in-place Marimo dashboard
scripts/                 acquisition, DDP training, evaluation, offline packaging
snowflake/               offline Snowflake GPU entry point
src/btc_predictor/
  data/                   downloader, validator, chronological simulator, windows
  features/               one shared CPU Market Analyser
  models/                 Patch Transformer, dilated TCN, GRU-attention
  training/               DDP trainer, delayed reward, losses, evaluation, selection
  live/                   Bybit stream aggregation and checkpoint inference
tests/                    leakage, delay, feature, model and selection tests
```

## 1. Installation

Python 3.11 or 3.12 is recommended.

```bash
git clone https://github.com/fgbvdcfgbfdefgb/pridictior_snow13.git
cd pridictior_snow13
git lfs install
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,live,dev]'
pytest
```

CUDA PyTorch wheels can differ by host image. If Snowflake or MoLab already supplies a CUDA-matched `torch`, install the base and extras without replacing that build.

## 2. Historical dataset

The canonical source is Binance's public market-data archive. The 1-second kline schema has OHLC, base/quote volume, trade count, and taker-buy volume. Binance states that spot timestamps from 2025-01-01 onward are microseconds; ingestion normalizes the complete series to milliseconds.

At 2026-10-04, finalized 2020-01-01 through 2026-10-03 coverage spans **213,235,200 wall-clock seconds** before source gaps. Exact source gaps are determined rather than assumed; each absent wall-clock second is represented by a marked flat-price, zero-activity row so replay remains one-second regular.

```bash
# Downloads completed months plus daily archives for the current partial month,
# converts one archive at a time to Zstandard Parquet, deletes temporary ZIPs,
# and records SHA-256 provenance.
make data

# Or choose an immutable cutoff explicitly.
python -m btc_predictor.data.binance_download \
  --start 2020-01-01 --end 2026-10-03 \
  --output data/processed \
  --manifest data/manifests/binance_1s.jsonl

python -m btc_predictor.data.validate data/processed
```

### Git LFS

`.gitattributes` tracks monthly/daily Parquet shards and checkpoints with LFS. Each shard remains below GitHub Free's documented 2 GB per-file LFS limit, but **storage and bandwidth are account-billed quotas**. Check available quota before the full push.

```bash
git lfs install
git add data/processed data/manifests/binance_1s.jsonl
git commit -m "data: add verified BTCUSDT 1s partitions"
git push
```

Do not put data blobs in ordinary Git. The small `data/sample/` CSV is only a test fixture. The current UTC day cannot be a finalized archive; live observations remain a separate feed until the day closes.

## 3. Detect and size the compute

```bash
btc-hardware --output runs/hardware.json
```

For four A10 24 GB GPUs, 48 vCPUs and 100 GB RAM, the intended final profile uses DDP across all four devices, BF16 when available, TF32 matrix operations, gradient accumulation, pinned loader memory, and a shared CPU feature definition. DDP keeps a full model replica on every GPU; VRAM is not treated as one 96 GB address space.

## 4. Train all candidates

### Fast parallel tournament

One architecture per GPU, useful for early rejection:

```bash
TOURNAMENT_UPDATES=20000 bash scripts/train_candidates_parallel.sh
```

### Comparable final DDP training

Every architecture receives all visible GPUs and equal update budget, sequentially:

```bash
GPUS=4 bash scripts/train_all.sh
```

Candidate models:

1. **Multiscale Patch Transformer** — primary model. Encodes 30 minutes at 1-second resolution, 3 hours at 10-second resolution, and 12 hours at 60-second resolution.
2. **Dilated TCN** — convolutional alternative with causal receptive fields.
3. **GRU + attention** — recurrent alternative over the same multiscale sequence.

Checkpoints, resolved configuration, detected hardware and JSONL metrics are stored under `runs/<candidate>/`. No checkpoint is deleted when a winner is selected.

## 5. Evaluate and select

```bash
EVAL_SECONDS=3600 bash scripts/evaluate_all.sh
```

The holdout evaluator walks one issue second at a time. It compares each complete 25-minute predicted path to stored non-simulated Binance observations and measures:

- endpoint MAE and RMSE in basis points;
- full-path MAE;
- direction error;
- 10–90% interval calibration;
- **instability**: the absolute difference between overlapping absolute-price paths issued one second apart.

`btc-select` computes a declared weighted score from the 2025 validation period and copies the best checkpoint to `checkpoints/best.pt`; all source checkpoints remain in place. Then run `make evaluate-test` once against the untouched 2026+ test period, followed by forward paper trading.

## 6. Snowflake offline training

Build an optional wheelhouse while package downloads are allowed:

```bash
make offline-bundle
```

Stage `project.tar.gz`, `wheels/`, the Git LFS dataset, and checksums. In the Snowflake GPU runtime:

```bash
python -m pip install --no-index --find-links wheels -r requirements/train.txt
python -m pip install --no-index --find-links wheels -e .
python snowflake/train_offline.py \
  --data-root /mnt/data/processed \
  --output-root /mnt/checkpoints/runs \
  --candidate all
```

See [`snowflake/README.md`](snowflake/README.md). The entry point fails closed when no CUDA GPU or Parquet data is visible.

## 7. Bybit live runner

Public market-data endpoints do not require an API key. Training uses stationary relative prices and normalized activity channels, then anchors the predicted log-return path to the current Bybit spot price. This reduces—but does not eliminate—the Binance/Bybit venue mismatch.

```bash
python -m btc_predictor.live.runner \
  --checkpoint checkpoints/best.pt \
  --bootstrap data/processed \
  --device cuda
```

The bootstrap uses the latest local 12 hours only as model warm-up context. Treat the initial live period as cold-start/basis adaptation and do not trade it.

## 8. MoLab Marimo dashboard

```bash
python -m pip install -r requirements/molab.txt
CHECKPOINT=checkpoints/best.pt DATA_ROOT=data/processed \
  marimo run notebooks/molab_live.py
```

The notebook creates one `go.FigureWidget` and mutates its four traces inside `batch_update()` every second. It does not create a new graph per update. The display includes actual Bybit BTCUSDT, the median 25-minute path, and an 80% interval.

## Operational caveats

- A predictor cannot guarantee profit or precision. Backtest slippage, fees, latency, outages, regime shifts, venue basis and liquidation risk before any capital is exposed.
- The live module is deliberately **forecast/paper-trading only**. It contains no order-placement API.
- Never commit credentials. The historical Binance and public Bybit feeds used here require no secret. Rotate any token pasted into a chat or shell after use.
- A 1-second dataset has highly overlapping examples. Report performance by contiguous time blocks, not random row splits.
- Keep timestamps in UTC and keep training/inference feature code identical.

## Data/API references

- Binance public data: <https://github.com/binance/binance-public-data>
- Binance archive: <https://data.binance.vision/>
- Bybit V5 WebSocket connections: <https://bybit-exchange.github.io/docs/v5/ws/connect>
- Bybit public trades: <https://bybit-exchange.github.io/docs/v5/websocket/public/trade>
- GitHub LFS limits: <https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage>

## License

MIT. Source market data remains subject to its provider's terms.
