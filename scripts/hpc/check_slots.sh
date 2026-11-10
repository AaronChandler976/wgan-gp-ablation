#!/usr/bin/env bash
# Show free GPU nodes, your queued jobs and when a new GPU job would start.
#   bash scripts/hpc/check_slots.sh            # quick look
#   bash scripts/hpc/check_slots.sh --gpu      # also run nvidia-smi on a GPU node (uses a slot briefly)
set -eo pipefail
cd "$(dirname "$0")/../.."
source scripts/hpc/_env.sh

echo "== Partition '$PARTITION'"
sinfo -p "$PARTITION" -o "%.12P %.6a %.12l %.6D %.10t %N"
echo
echo "== Idle nodes"
sinfo -p "$PARTITION" -t idle -h -o "%D idle: %N" | grep . || echo "none idle right now"
echo
echo "== Your jobs"
squeue -u "$USER" -o "%.12i %.24j %.10T %.10M %.10l %R"
echo
echo "== When would one GPU job ($TIME) start?"
sbatch --test-only -p "$PARTITION" --gres="$GRES" --time="$TIME" --wrap "hostname" 2>&1 || true

if [ "${1:-}" = "--gpu" ]; then
  echo
  echo "== GPU on a node of '$PARTITION'"
  srun -p "$PARTITION" --gres="$GRES" --time=00:02:00 \
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
