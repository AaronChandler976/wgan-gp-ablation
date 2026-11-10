#!/usr/bin/env bash
# One-time setup on the HPC. Run from the repo root on the login node:
#   bash scripts/hpc/setup_env.sh
# Installs Miniconda in your home folder if conda is missing, creates the conda environment,
# and installs PyTorch and this package.
set -eo pipefail
cd "$(dirname "$0")/../.."
source scripts/hpc/_env.sh

# Older GPUs (the P100 nodes) need an older CUDA build. For newer nodes you can use, for example:
#   CWP_TORCH=torch CWP_TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/hpc/setup_env.sh
TORCH_SPEC="${CWP_TORCH:-torch==2.5.1}"
TORCH_INDEX="${CWP_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

if ! command -v conda >/dev/null 2>&1 && [ ! -d "$HOME/miniconda3" ]; then
  echo "Installing Miniconda into ~/miniconda3 ..."
  installer="$(mktemp --suffix=.sh)"
  curl -fsSL -o "$installer" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$installer" -b -p "$HOME/miniconda3"
  "$HOME/miniconda3/bin/conda" init bash >/dev/null
fi

base="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
source "$base/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda environment '$ENV_NAME' already exists"
else
  conda env create -n "$ENV_NAME" -f environment.yml
fi
conda activate "$ENV_NAME"
pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX"
pip install -e .

python -c "import torch; print('PyTorch', torch.__version__, '| GPU visible on this node:', torch.cuda.is_available())"
echo "Login nodes have no GPU, so 'False' above is normal."
echo "Next: bash scripts/hpc/check_slots.sh"
