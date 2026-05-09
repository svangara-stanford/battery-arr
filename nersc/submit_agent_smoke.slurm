#!/bin/bash
#SBATCH --job-name=battery-aar-agent-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Template only: set partition/QOS/account/container details for your allocation.

set -euo pipefail

cd "${BATTERY_AAR_REPO:-$PWD}"
mkdir -p logs runs/agent_smoke

python agent_starter/run_candidate.py \
  --processed-dir "${BATTERY_AAR_PROCESSED_DIR:-data/demo}" \
  --config "${BATTERY_AAR_CANDIDATE_CONFIG:-agent_starter/candidate_configs/ridge_capacity_slope.yaml}" \
  --out "${BATTERY_AAR_AGENT_OUT:-runs/agent_smoke/ridge_capacity_slope}"
