#!/usr/bin/env bash

# Train both pedagogical models on tiny real-data subsets.
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
python -m pip install --quiet -r requirements.txt

cd demonstration

echo "Training the Holstein model for $epochs epoch(s)..."
python train_holstein.py --epochs "$epochs"
echo "Holstein validation PNG: demonstration/output/holstein_validation_y_equals_x.png"

echo "Training the FK model for $epochs epoch(s)..."
python train_fk.py --epochs "$epochs"
echo "FK validation PNG: demonstration/output/fk_validation_y_equals_x.png"

echo "Training complete."
echo "Results: demonstration/output/"
