#!/bin/bash
#SBATCH --job-name=va4mds_heatmap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/heatmap_%j.out
#SBATCH --error=logs/heatmap_%j.err

set -euo pipefail

module load python/3.11.3

if [ ! -d "$HOME/venvs/va4mds_v3" ]; then
  python -m venv "$HOME/venvs/va4mds_v3"
  source "$HOME/venvs/va4mds_v3/bin/activate"
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source "$HOME/venvs/va4mds_v3/bin/activate"
fi

mkdir -p logs

python scripts/compute_heatmap.py \
  --pred_dir data/pred \
  --obs_dir data/obs \
  --out_dir data/heatmaps
