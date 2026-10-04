#!/usr/bin/env bash
set -euo pipefail
# Run only after validation-based model selection. This does not re-select a model.
python -m btc_predictor.training.evaluate \
  --checkpoint checkpoints/best.pt \
  --data data/processed \
  --output runs/final-test.json \
  --split test \
  --samples "${EVAL_SECONDS:-3600}"
