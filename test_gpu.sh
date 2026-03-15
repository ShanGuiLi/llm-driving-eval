#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1  # 申请1块GPU
#SBATCH --mem=4G
#SBATCH --time=00:02:00
#SBATCH --job-name=test_gpu

# 初始化conda（脚本里必须加）
eval "$(conda shell.bash hook)"
# 激活环境
conda activate llm-driving-eval

# 验证GPU和PyTorch
python -c "
import torch
print('PyTorch版本:', torch.__version__)
print('CUDA版本:', torch.version.cuda)
print('GPU是否可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU型号:', torch.cuda.get_device_name(0))
    print('GPU数量:', torch.cuda.device_count())
    print('当前GPU索引:', torch.cuda.current_device())
"