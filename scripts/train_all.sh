#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-$(python - <<'PY'
try:
 import torch
 print(max(1, torch.cuda.device_count()))
except Exception:
 print(1)
PY
)}"

# Each architecture uses all GPUs with DDP, one after another, so comparisons
# receive equal compute. Checkpoints for every candidate are retained.
for MODEL in patch_transformer dilated_tcn gru_attention; do
  echo "==> training ${MODEL} on ${GPUS} rank(s)"
  torchrun --standalone --nproc-per-node="${GPUS}" -m btc_predictor.training.train \
    --config configs/default.yaml \
    --model-config "configs/models/${MODEL}.yaml" \
    --output "runs/${MODEL}"
done

python -m btc_predictor.training.select_model runs --output checkpoints/best.pt
