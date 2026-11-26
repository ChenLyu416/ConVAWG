#!/bin/bash
# Transfer/augmentation grid of the paper (sequential, GPU 0).
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
python train_transfer.py --matrix --gpu 0 2>&1 | tee logs/matrix.log
