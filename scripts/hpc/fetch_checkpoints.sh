#!/usr/bin/env bash
# Run on YOUR machine: copy trained weights from the HPC into checkpoints/ (they are not in git).
#   CWP_HPC_REPO=/home/<id>/wgan-gp-ablation bash scripts/hpc/fetch_checkpoints.sh
# CWP_HPC_HOST is the ssh name of the HPC (default: hpc, as in ~/.ssh/config).
set -eo pipefail
cd "$(dirname "$0")/../.."
HOST="${CWP_HPC_HOST:-hpc}"
REPO="${CWP_HPC_REPO:?set CWP_HPC_REPO to the repo folder on the HPC}"
rsync -av --include='*.pt' --exclude='*' "$HOST:$REPO/checkpoints/" checkpoints/
