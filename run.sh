#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --job-name=llm-driving-eval
#SBATCH --output=/home/msds/ming007/llm-driving-eval/eval_%j.out
#SBATCH --error=/home/msds/ming007/llm-driving-eval/eval_%j.err

set -e

PROJECT_ROOT=/home/msds/ming007/llm-driving-eval
CONDA_ENV_PATH=/home/msds/ming007/.conda/envs/llm-driving-eval

module purge
module load anaconda

eval "$(conda shell.bash hook)"
conda activate llm-driving-eval

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

# Make sure the conda env libraries take precedence over module libraries
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"

python -m src.llm_eval.qwen25_local_eval
# python -m src.training.train_qwen25_lora
# python -m src.llm_eval.qwen25_lora_eval
# python -m src.llm_eval.qwen35_api_eval