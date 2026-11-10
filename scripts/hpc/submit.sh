#!/usr/bin/env bash
# Claim the next unclaimed runs and submit them as one Slurm job array.
#   bash scripts/hpc/submit.sh -n 6                    # next 6 runs
#   bash scripts/hpc/submit.sh -n 10 --seeds "1 2"     # only seeds 1 and 2
#   bash scripts/hpc/submit.sh -n 5 --dataset google
# Settings (partition, time limit, ...) are in scripts/hpc/_env.sh.
set -eo pipefail
cd "$(dirname "$0")/../.."
source scripts/hpc/_env.sh

N=4; SEEDS=""; DATASET=""
while [ $# -gt 0 ]; do
  case "$1" in
    -n) N="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

activate_env
ARGS=(-n "$N")
# shellcheck disable=SC2206
[ -n "$SEEDS" ] && ARGS+=(--seeds $SEEDS)
[ -n "$DATASET" ] && ARGS+=(--dataset "$DATASET")

mkdir -p logs/hpc
LIST="logs/hpc/batch_$(date +%Y%m%d_%H%M%S).txt"
IDS="$(python -m cwp.runs next "${ARGS[@]}" --ids 2>/dev/null || true)"
if [ -z "$IDS" ]; then
  echo "Nothing left to do for this filter. See: python -m cwp.runs status"
  exit 0
fi
python -m cwp.runs next "${ARGS[@]}" > "$LIST"
COUNT="$(wc -l < "$LIST" | tr -d ' ')"

# shellcheck disable=SC2086
python -m cwp.runs claim $IDS | grep -v '^Run `python'
JOB="$(sbatch --parsable -p "$PARTITION" --gres="$GRES" --time="$TIME" \
       --array="1-${COUNT}%${CONCURRENT}" scripts/hpc/job.sbatch "$LIST")"

echo
echo "Submitted $COUNT run(s) as job array $JOB (at most $CONCURRENT at a time, $TIME each)."
echo "  watch:    squeue -u $USER"
echo "  logs:     logs/hpc/cwp_${JOB}_<task>.log"
echo "  cancel:   scancel $JOB    (then: python -m cwp.runs release <run_id> ...)"
echo "  when done: bash scripts/hpc/sync_results.sh"
