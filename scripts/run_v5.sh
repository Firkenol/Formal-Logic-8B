#!/bin/bash
# scripts/run_v5.sh: build v5 data, train on all GPUs, then run the demo.
# Run from the repo root inside thrm_env:
#   nohup bash scripts/run_v5.sh > v5_pipeline.log 2>&1 &
#   tail -f v5_pipeline.log
set -e

pip install -q requests pytest numpy
NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "[1/3] building training data $(date)"
# source files are cached in .cache_v5_files, so reruns are quick
python scripts/build_training_data_v5.py --max-examples 3000

echo "[2/3] training on $NUM_GPUS GPUs $(date)"
torchrun --nproc_per_node "$NUM_GPUS" scripts/train_thrm_v5.py \
    --grad-accum $(( NUM_GPUS >= 16 ? 1 : 16 / NUM_GPUS ))

echo "[3/3] demo with the new adapter $(date)"
python scripts/thrm_demo.py --adapter ./thrm_v5 --attempts 5

echo "DONE $(date)"
