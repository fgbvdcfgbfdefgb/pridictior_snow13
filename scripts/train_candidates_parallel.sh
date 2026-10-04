#!/usr/bin/env bash
set -euo pipefail
# Fast architecture tournament: one independent candidate per GPU.
# Use train_all.sh afterward for full DDP training of the winner/all candidates.
MODELS=(patch_transformer dilated_tcn gru_attention)
for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  CUDA_VISIBLE_DEVICES="$i" python -m btc_predictor.training.train \
    --config configs/default.yaml \
    --model-config "configs/models/${model}.yaml" \
    --output "runs/tournament-${model}" \
    --max-updates "${TOURNAMENT_UPDATES:-20000}" &
done
wait
