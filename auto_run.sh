#!/bin/bash

# clear vram and stale processes
pkill -9 -f vllm
pkill -9 -f python
sudo fuser -v /dev/nvidia* -k -9
sudo rm -rf /dev/shm/*

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=TRITON
export VLLM_USE_TRITON_FLASH_ATTN=1

# launch teacher model
nohup /home/azureuser/Formal-Logic-8B/thrm_env/bin/vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --enforce-eager \
  --gpu-memory-utilization 0.90 \
  --distributed-executor-backend mp \
  --host 0.0.0.0 \
  --port 8000 > server_log.txt 2>&1 &

echo "warming up vllm server..."

while ! nc -z localhost 8000; do   
  sleep 30
done

echo "starting phase 1 data generation"

# point to the scripts folder
python scripts/generate_dataset.py > generation_progress.log 2>&1

echo "phase 1 done. clearing vram for phase 2"

# kill vllm 
pkill -9 -f vllm
sleep 15

echo "starting phase 2 qlora training"

# point to the scripts folder
python scripts/train_thrm_3b_qlora.py > training_progress.log 2>&1

echo "PIPELINE COMPLETE."