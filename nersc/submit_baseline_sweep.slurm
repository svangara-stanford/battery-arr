#!/bin/bash
#SBATCH --job-name=battery-aar-baselines
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Template only: set partition/QOS/account/container details for your allocation.

set -euo pipefail

cd "${BATTERY_AAR_REPO:-$PWD}"
mkdir -p logs reports runs

python scripts/run_real_data_baseline_matrix.py \
  --raw-dir "${BATTERY_AAR_RAW_DIR:-data/raw/chueh_toyota_fast_charge}" \
  --processed-dir "${BATTERY_AAR_PROCESSED_DIR:-data/processed/chueh_toyota_fast_charge}" \
  --runs-dir "${BATTERY_AAR_RUNS_DIR:-runs/chueh_toyota_phase1}" \
  --reports-dir "${BATTERY_AAR_REPORTS_DIR:-reports}" \
  --max-cells-per-batch "${BATTERY_AAR_MAX_CELLS_PER_BATCH:-4}" \
  --max-cycle 100 \
  --seed 42
