#!/bin/bash
# Full encoder grid for escalation forecasting (sequential, GPU 0).
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
for task in level jump; do
  for model in microsoft/deberta-v3-base answerdotai/ModernBERT-base allenai/longformer-base-4096; do
    safe=$(echo "$model" | tr '/' '_')
    echo "=== $task / $model ==="
    python train_forecaster.py --task "$task" --models "$model" --gpu 0 \
      2>&1 | tee "logs/${safe}_${task}.log"
  done
done
python evaluate_forecasts.py --pred results/*/predictions.jsonl --out evaluation_summary.json
