#!/usr/bin/env bash
# ============================================================
#  Eight-class toxic behaviour classification (paper Table 3)
#
#  1. build_dataset.py            binary TOXIC / NON-TOXIC data and the
#                                 scenario-level 70/15/15 split (data/)
#  2. build_multiclass_dataset.py eight-class data on the same split
#                                 (data_multiclass/)
#  3. train_multiclass.py         BERT-base, RoBERTa-base, DeBERTa-v3-base
#                                 under S1 (message only) and S2 (with context)
#
#  Usage: bash run_experiments.sh [GPU_ID]
#  The binary task can be trained with train_classifier.py; the zero-shot
#  LLM baselines for both tasks are run with run_llm_baseline.py.
# ============================================================
set -euo pipefail

GPU="${1:-0}"
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

if [[ ! -f data/split_metadata.json ]]; then
    python build_dataset.py --output_dir data --train_ratio 0.70 --dev_ratio 0.15 --seed 42
fi
if [[ ! -f data_multiclass/setting1/train.jsonl ]]; then
    python build_multiclass_dataset.py --output_dir data_multiclass --split_meta data/split_metadata.json
fi

python train_multiclass.py --data_dir data_multiclass --output_dir results_multiclass \
    --epochs 3 --lr 2e-5 --seed 42 --gpu "$GPU" 2>&1 | tee logs/multiclass.log
python train_multiclass.py --output_dir results_multiclass --aggregate_only
