# 训练配置文件

BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TRAIN_FILE = "data/sft/train.jsonl"
VAL_FILE = "data/sft/val.jsonl"
OUTPUT_DIR = "models/qwen25_lora_ckpt"

MAX_LENGTH = 512

LEARNING_RATE = 2e-4
NUM_EPOCHS = 1

PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

USE_4BIT = True

# 仅用于流程验证
MAX_STEPS = 5
LOGGING_STEPS = 1
SAVE_STEPS = 5
EVAL_STEPS = 5