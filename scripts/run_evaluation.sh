#!/bin/bash

# Define the paths
PREDICTIONS_PATH="../predictions.jsonl"
RUN_ID="thrm_3b_qlora_eval"
LOG_DIR="../eval_logs"

# Ensure the log directory exists
mkdir -p $LOG_DIR

# Run the official SWE-bench evaluation harness
python3 -m swebench.harness.run_evaluation \
    --predictions_path $PREDICTIONS_PATH \
    --run_id $RUN_ID \
    --max_workers 4 \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --split test \
    --log_dir $LOG_DIR