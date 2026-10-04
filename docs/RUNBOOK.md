# Training and live operations runbook

## Preflight

1. `git status --short` is clean.
2. `git lfs pull` has materialized dataset objects.
3. `python -m btc_predictor.data.validate data/processed` passes.
4. `btc-hardware --output runs/hardware.json` shows all expected GPUs.
5. System clocks use UTC/NTP.
6. Available disk can hold checkpoints and loader spill.

## Short smoke test

Override `max_updates` from the command line:

```bash
torchrun --standalone --nproc-per-node=4 -m btc_predictor.training.train \
  --config configs/default.yaml \
  --model-config configs/models/patch_transformer.yaml \
  --output runs/smoke --max-updates 20
```

Confirm loss is finite, every rank remains alive, a checkpoint is written, and GPU utilization is balanced.

## Full run

```bash
GPUS=4 bash scripts/train_all.sh
EVAL_SECONDS=3600 bash scripts/evaluate_all.sh
```

Archive the resolved config, hardware report, metrics, evaluator output and every candidate checkpoint. Never select from test-period results.

## Failure recovery

Resume an interrupted candidate:

```bash
torchrun --standalone --nproc-per-node=4 -m btc_predictor.training.train \
  --config configs/default.yaml \
  --model-config configs/models/patch_transformer.yaml \
  --output runs/patch_transformer \
  --resume runs/patch_transformer/latest.pt
```

Checkpoints are written to a temporary path and atomically renamed.

## Live deployment gates

- checkpoint and configuration hashes approved;
- holdout and paper-trading metrics approved;
- 12-hour history warm;
- Binance-to-Bybit basis/cold-start period observed;
- feed staleness, reconnect and prediction latency monitored;
- output sanity limits and kill switch implemented in the downstream paper trader;
- no order permissions in the predictor process.

The dashboard is observability, not an execution surface.
