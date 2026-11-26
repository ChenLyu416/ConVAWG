#!/bin/bash
# Zero-shot CGA transfer grid: every ConVAWG level forecaster x both corpora,
# scoring derailment as P(next level >= 3).
# Inference only; select a GPU with CUDA_VISIBLE_DEVICES if one is free.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs

[ -f data/wiki/test.jsonl ] || python build_dataset.py

for ckpt in ../escalation_forecasting/results/*_level/checkpoint-*; do
  [ -d "$ckpt" ] || continue
  model=$(basename "$(dirname "$ckpt")")
  step=$(basename "$ckpt")
  for corpus in wiki cmv; do
    echo "=== $model/$step / $corpus ==="
    python evaluate_cga.py --checkpoint "$ckpt" --data_dir "data/$corpus" \
      --score_classes 3 4 \
      --output_dir "results/${model}_${step}__${corpus}" \
      2>&1 | tee "logs/${model}_${step}__${corpus}.log"
  done
done
