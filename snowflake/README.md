# Snowflake offline run

The runtime performs no HTTP calls. Stage the repository, Git LFS Parquet shards, and (if the Snowflake environment cannot install from its Python package channel) the wheelhouse made by `scripts/package_offline.sh`.

```bash
# Install while the approved package channel is available
python -m pip install -r requirements/train.txt
python -m pip install -e .

# Then disable egress and verify hardware/data
btc-hardware --output runs/hardware.json
python -m btc_predictor.data.validate /mnt/data/processed

# All three candidates, each trained with DDP over all four A10 GPUs
python snowflake/train_offline.py \
  --data-root /mnt/data/processed \
  --output-root /mnt/checkpoints/runs \
  --candidate all
```

For a short architecture tournament (one candidate per GPU), use `scripts/train_candidates_parallel.sh`; for the final comparable run, use the DDP entry point above. The process is update-count based, not epoch based. Every candidate's `latest.pt`, resolved configuration, hardware report, and metrics log is retained.

Expected resources: four NVIDIA A10 24 GB GPUs, 48 vCPUs and 100 GB RAM. The auto profile uses BF16 when supported, a 768-wide/12-layer transformer, four samples per GPU, gradient accumulation, TF32 matrix multiplication, pinned memory and up to 32 loader workers. Lower-memory hardware is scaled down automatically.
