#!/bin/bash
#SBATCH --job-name=va4mds_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/diag_%j.out
#SBATCH --error=logs/diag_%j.err

set -euo pipefail

module load python/3.11.3

# Create venv once (first run)
if [ ! -d "$HOME/venvs/va4mds_v3" ]; then
  python -m venv "$HOME/venvs/va4mds_v3"
  source "$HOME/venvs/va4mds_v3/bin/activate"
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source "$HOME/venvs/va4mds_v3/bin/activate"
fi

mkdir -p logs

# Edit these paths if needed
python scripts/run_diagnostics.py \
  --renders_dir data/renders \
  --heatmaps_dir data/heatmaps \
  --instability lpips \
  --out_dir results
