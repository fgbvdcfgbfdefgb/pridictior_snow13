#!/usr/bin/env bash
set -euo pipefail
END="${END:-$(date -u -d yesterday +%F)}"
python -m btc_predictor.data.binance_download \
  --start "${START:-2020-01-01}" --end "$END" \
  --output data/processed --manifest data/manifests/binance_1s.jsonl
python -m btc_predictor.data.validate data/processed
printf '\nDataset is ready. To version the shards:\n'
printf '  git lfs install && git add data/processed data/manifests && git commit\n'
