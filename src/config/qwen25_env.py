import os
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(dotenv_path=ENV_PATH)

LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b-instruct")
LOCAL_VLM_MODEL = os.getenv("LOCAL_VLM_MODEL", "qwen2.5vl:7b")

RAW_VIDEO_DIR = os.path.join(PROJECT_ROOT, "data", "raw_videos")
ANNOTATION_DIR = os.path.join(PROJECT_ROOT, "data", "annotations")
EVAL_RESULT_DIR = os.path.join(PROJECT_ROOT, "data", "eval_results")

for dir_path in [RAW_VIDEO_DIR, ANNOTATION_DIR, EVAL_RESULT_DIR]:
    os.makedirs(dir_path, exist_ok=True)