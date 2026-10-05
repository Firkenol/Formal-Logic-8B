#!/bin/bash
# scripts/run_v5.sh: build v5 data, train, then run the demo on the new adapter.
# Run from the repo root inside thrm_env:
#   nohup bash scripts/run_v5.sh > v5_pipeline.log 2>&1 &
#   tail -f v5_pipeline.log
set -e

pip install -q requests pytest numpy

echo "[1/3] building training data $(date)"
python scripts/build_training_data_v5.py --max-examples 3000

echo "[2/3] training $(date)"
python scripts/train_thrm_v5.py

echo "[3/3] demo with the new adapter $(date)"
python scripts/thrm_demo.py --adapter ./thrm_v5 --attempts 5

echo "DONE $(date)"
