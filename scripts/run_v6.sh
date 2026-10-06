#!/bin/bash
# scripts/run_v6.sh: Qwen2.5-Coder-7B + THRM pipeline, end to end.
#   1. build ~10k training examples (held-out repos, code in the prompt)
#   2. QLoRA fine-tune on all GPUs
#   3. clean eval: base vs. adapter on SWE-bench Lite, official Docker harness
#   4. demo on demo_repo
#
# Run from the repo root inside thrm_env, detached from SSH:
#   setsid nohup bash scripts/run_v6.sh > v6_pipeline.log 2>&1 < /dev/null &
#   tail -f v6_pipeline.log
set -e

BASE=Qwen/Qwen2.5-Coder-7B
NUM_GPUS=$(nvidia-smi -L | wc -l)
pip install -q requests pytest numpy
python -c "import swebench" 2>/dev/null || pip install -q swebench

if [ -s data_v6/train.jsonl ] && [ -s data_v6/val.jsonl ]; then
    echo "[1/4] data_v6 already built, skipping (delete data_v6/ to rebuild)"
else
    echo "[1/4] building training data $(date)"
    python scripts/build_training_data_v5.py --base-model "$BASE" \
        --out-dir data_v6 --max-examples 10000 --workers 32
fi

if [ -s thrm_v6/adapter_config.json ]; then
    echo "[2/4] thrm_v6 already trained, skipping (delete thrm_v6/ to retrain)"
else
    echo "[2/4] training on $NUM_GPUS GPUs $(date)"
    torchrun --nproc_per_node "$NUM_GPUS" scripts/train_thrm_v5.py \
        --base-model "$BASE" --data-dir data_v6 --output-dir ./thrm_v6 \
        --eval-steps 100 \
        --grad-accum $(( NUM_GPUS >= 16 ? 1 : 16 / NUM_GPUS ))
fi

echo "[3/4] clean evaluation $(date)"
python scripts/eval_v6.py prepare --base-model "$BASE" --n 100
pids=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$i python scripts/eval_v6.py generate --base-model "$BASE" \
        --adapter ./thrm_v6 --shard "$i" --num-shards "$NUM_GPUS" &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
python scripts/eval_v6.py score

echo "[4/4] demo $(date)"
python scripts/thrm_demo.py --adapter ./thrm_v6 --attempts 5

echo "DONE $(date)"
