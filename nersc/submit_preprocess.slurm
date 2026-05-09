#!/bin/bash
#SBATCH --job-name=battery-aar-preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Template only: set partition/QOS/account/container details for your allocation.

set -euo pipefail

cd "${BATTERY_AAR_REPO:-$PWD}"
mkdir -p logs

python scripts/preprocess_chueh_toyota.py \
  --raw-dir "${BATTERY_AAR_RAW_DIR:-data/raw/chueh_toyota_fast_charge}" \
  --out "${BATTERY_AAR_PROCESSED_DIR:-data/processed/chueh_toyota_fast_charge}" \
  --max-cells-per-batch "${BATTERY_AAR_MAX_CELLS_PER_BATCH:-4}" \
  --first-n-cycles 100 \
  --seed 42
