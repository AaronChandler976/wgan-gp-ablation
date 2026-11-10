# Sourced by the other scripts in this folder: cluster settings and conda activation.
# Override any of these with environment variables, for example:
#   CWP_PARTITION=gpu CWP_TIME=08:00:00 bash scripts/hpc/submit.sh -n 6

PARTITION="${CWP_PARTITION:-gpu}"        # Slurm partition with the GPU nodes
GRES="${CWP_GRES:-gpu}"                  # GPU request, passed to --gres
TIME="${CWP_TIME:-04:00:00}"             # time limit per run (the gpu partition allows up to 48 h)
CONCURRENT="${CWP_CONCURRENT:-4}"        # how many runs of one batch may run at the same time
ENV_NAME="${CWP_ENV:-cwp}"               # conda environment

activate_env() {
  [ -n "${CWP_NO_CONDA:-}" ] && return 0          # use the current Python as it is
  local base
  base="$(conda info --base 2>/dev/null || true)"
  if [ -z "$base" ] && [ -d "$HOME/miniconda3" ]; then base="$HOME/miniconda3"; fi
  if [ -z "$base" ]; then
    echo "conda not found. Run first: bash scripts/hpc/setup_env.sh" >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source "$base/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
}
