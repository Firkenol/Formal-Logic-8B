#!/bin/bash

echo "========================================"
echo "STARTING THRM V3 PIPELINE (SLEDGEHAMMER MODE)"
echo "========================================"

# THE FIX: Activate your virtual environment inside the background script
source ~/Formal-Logic-8B/thrm_env/bin/activate

# THE FIX: Only kill python processes owned by you, not the system
echo "Purging old processes..."
pkill -9 -u azureuser -f python || true
sleep 5

echo "[1/2] Phase 2: QLoRA V3 Training (~45 mins)..."
python -u scripts/train_qlora_v3.py > logs/train_v3.log 2>&1

echo "Purging VRAM post-training..."
pkill -9 -u azureuser -f python || true
sleep 5

echo "[2/2] Phase 4: SWE-Bench Orchestrator (12-24 hours)..."
python -u scripts/orchestrator.py > logs/orchestrator.log 2>&1

echo "PIPELINE COMPLETE."