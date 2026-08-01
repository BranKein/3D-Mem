#!/bin/bash
# GOAT-Bench val_unseen evaluation with qwen2.5vl:7b via ollama.
cd /home/yhkim/3D-Mem

export VLM_PROVIDER=ollama
export OLLAMA_MODEL=qwen2.5vl:7b
export OLLAMA_TIMEOUT=1200

LOG=/home/yhkim/3D-Mem/goatbench_qwen7b.log
echo "=== started $(date -Is) | model=$OLLAMA_MODEL ===" | tee "$LOG"

~/anaconda3/envs/3dmem/bin/python -u run_goatbench_evaluation.py \
    -cf cfg/eval_goatbench.yaml 2>&1 | tee -a "$LOG"

echo "=== finished $(date -Is) | exit=$? ===" | tee -a "$LOG"
