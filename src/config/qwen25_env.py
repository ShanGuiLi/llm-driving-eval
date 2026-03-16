import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

#qwen2_5_7b_instruct
LOCAL_VLM_MODEL = os.path.join(PROJECT_ROOT, "models", "qwen25_vl")
#qwen2_5_vl_3b_instruct
LOCAL_LLM_MODEL = os.path.join(PROJECT_ROOT, "models", "qwen25_text")

EVAL_RESULT_DIR = os.path.join(PROJECT_ROOT, "data", "eval_results")