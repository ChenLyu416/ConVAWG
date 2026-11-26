#!/usr/bin/env bash
# ============================================================
#  Run all baseline models for persona-relation classification
#  Usage: bash run_all_baselines.sh [GPU_ID]
#  Example: bash run_all_baselines.sh 0
# ============================================================

set -euo pipefail

GPU="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/configs/relation_clf.yaml"

echo "======================================================"
echo "  Persona-Relation Classification — Baseline Suite"
echo "  GPU: $GPU | Config: $CONFIG"
echo "======================================================"

# ── Step 1: Build dataset ─────────────────────────────────
echo ""
echo ">>> [1/4] Building dataset …"
python "$SCRIPT_DIR/build_dataset.py" --config "$CONFIG"
echo "Dataset build complete."

# ── Step 2: BERT ──────────────────────────────────────────
echo ""
echo ">>> [2/4] Training bert-base-uncased …"
python "$SCRIPT_DIR/train_classifier.py" \
  --config "$CONFIG" \
  --model_name bert-base-uncased \
  --gpu "$GPU"

# ── Step 3: RoBERTa ───────────────────────────────────────
echo ""
echo ">>> [3/4] Training roberta-base …"
python "$SCRIPT_DIR/train_classifier.py" \
  --config "$CONFIG" \
  --model_name roberta-base \
  --gpu "$GPU"

# ── Step 4: Longformer (scenario-level) ───────────────────
echo ""
echo ">>> [4/4] Training allenai/longformer-base-4096 (scenario-level) …"
python "$SCRIPT_DIR/train_classifier.py" \
  --config "$CONFIG" \
  --model_name "allenai/longformer-base-4096" \
  --gpu "$GPU"

# ── Comparison table ──────────────────────────────────────
echo ""
echo ">>> Final comparison across all models …"
python "$SCRIPT_DIR/train_classifier.py" \
  --config "$CONFIG" \
  --compare

echo ""
echo "======================================================"
echo "  All done. Results in:"
echo "  $SCRIPT_DIR/results/"
echo "======================================================"
