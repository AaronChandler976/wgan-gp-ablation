#!/usr/bin/env bash
# Run on the HPC when jobs have finished: show the status, then commit and push the new
# results and claims so the rest of the team sees them. Needs git push access from the HPC
# (an SSH key or a token for GitHub).
set -eo pipefail
cd "$(dirname "$0")/../.."
source scripts/hpc/_env.sh
activate_env

python -m cwp.runs status | sed -n '1,2p'
running="$(squeue -u "$USER" -h -n cwp 2>/dev/null | wc -l | tr -d ' ')"
[ "$running" != "0" ] && echo "Note: $running of your cwp job(s) are still queued or running."
python -m cwp.runs sync
