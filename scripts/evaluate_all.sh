#!/usr/bin/env bash
set -euo pipefail
for MODEL in patch_transformer dilated_tcn gru_attention; do
  python -m btc_predictor.training.evaluate \
    --checkpoint "runs/${MODEL}/latest.pt" \
    --data data/processed \
    --output "runs/${MODEL}/evaluation.json" \
    --samples "${EVAL_SECONDS:-3600}"
done
python -m btc_predictor.training.select_model runs --output checkpoints/best.pt
