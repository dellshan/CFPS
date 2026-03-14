#!/bin/bash
#SBATCH --job-name=ns_gsplat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module purge
module load cuda/11.8
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# conda 正确激活方式（batch 里必须 source conda.sh）
source /home/yshan/miniconda/etc/profile.d/conda.sh
conda activate cfps   # 或者 conda activate ns

# 先确认 CLI 存在
which ns-train
python -c "import torch, gsplat, nerfstudio; print(torch.__version__, torch.version.cuda, gsplat.__version__)"

# 正式训练（把路径换成你的）
ns-train splatfacto \
  --data /path/to/ns_data/my_scene \
  --max-num-iterations 30000
