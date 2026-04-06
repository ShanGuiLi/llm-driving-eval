import os
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(ENV_PATH)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.5-35b-a3b")

RAW_VIDEO_DIR = os.path.join(PROJECT_ROOT, "data", "raw_videos")
EVAL_RESULT_DIR = os.path.join(PROJECT_ROOT, "data", "eval_results", "qwen35")
PROMPT_DIR = os.path.join(PROJECT_ROOT, "src/config")

for d in [RAW_VIDEO_DIR, EVAL_RESULT_DIR, PROMPT_DIR]:
    os.makedirs(d, exist_ok=True)