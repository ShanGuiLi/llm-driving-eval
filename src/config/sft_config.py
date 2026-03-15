# src/config/sft_config.py
# LoRA配置

BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TRAIN_FILE = "data/sft/train.jsonl"
VAL_FILE = "data/sft/val.jsonl"
OUTPUT_DIR = "models/qwen25_lora_adapter"

MAX_LENGTH = 2048
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

USE_8BIT = True