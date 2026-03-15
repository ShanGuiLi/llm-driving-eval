from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForImageTextToText
)

# 自动定位到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# models目录
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# 模型保存路径
text_save_dir = MODELS_DIR / "qwen25_text"
vl_save_dir = MODELS_DIR / "qwen25_vl"

text_model_name = "Qwen/Qwen2.5-7B-Instruct"
vl_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

print("Project root:", PROJECT_ROOT)
print("Downloading models to:", MODELS_DIR)

# =========================
# 下载 Qwen2.5 文本模型
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    text_model_name,
    trust_remote_code=True
)

text_model = AutoModelForCausalLM.from_pretrained(
    text_model_name,
    trust_remote_code=True
)

text_save_dir.mkdir(exist_ok=True)

tokenizer.save_pretrained(text_save_dir)
text_model.save_pretrained(text_save_dir)

print("Text model saved to:", text_save_dir)


# =========================
# 下载 Qwen2.5 VL 模型
# =========================

processor = AutoProcessor.from_pretrained(
    vl_model_name,
    trust_remote_code=True
)

vl_model = AutoModelForImageTextToText.from_pretrained(
    vl_model_name,
    trust_remote_code=True
)

vl_save_dir.mkdir(exist_ok=True)

processor.save_pretrained(vl_save_dir)
vl_model.save_pretrained(vl_save_dir)

print("VL model saved to:", vl_save_dir)

print("All models downloaded successfully.")