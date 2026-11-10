#!/usr/bin/env bash
#SBATCH --job-name=cwp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/hpc/%x_%A_%a.log
# One task of a job array = one training run. submit.sh passes the partition, the GPU request,
# the time limit, the array range and the file that lists one train command per line.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/hpc/_env.sh
activate_env

LIST="$1"
CMD="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")"
echo "node: $(hostname)   task ${SLURM_ARRAY_TASK_ID}: $CMD"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "no GPU visible"
$CMD
