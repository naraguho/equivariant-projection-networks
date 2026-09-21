#!/usr/bin/env bash

# Train both manuscript models on the included real-data samples.
# Usage:
#   bash run_training.sh
#   EPOCHS=20 bash run_training.sh

set -euo pipefail

# Always work relative to this script, regardless of the caller's directory.
repository_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$repository_dir"

epochs="${EPOCHS:-10}"
python_command="${PYTHON:-python3}"

if [[ ! -d .venv ]]; then
    "$python_command" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e ./manuscript

cd manuscript

echo "Training the Holstein model for $epochs epoch(s)..."
python training/train_holstein.py \
    --data-dir data/sample \
    --output outputs/holstein \
    --epochs "$epochs"
echo "Holstein validation PNG: manuscript/outputs/holstein/validation_ed_vs_ml.png"

echo "Training the FK model for $epochs epoch(s)..."
python training/train_fk.py \
    --data data/sample/fk_real_sample.csv.gz \
    --output outputs/fk \
    --epochs "$epochs"
echo "FK validation PNG: manuscript/outputs/fk/validation_ed_vs_ml.png"

echo "Training complete."
echo "Holstein results: manuscript/outputs/holstein/"
echo "FK results:        manuscript/outputs/fk/"
