import base64
import json
import os
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.config.qwen35_env import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    QWEN_MODEL,
    PROJECT_ROOT,
)

from src.config.vlm_prompt import VIDEO_DESCRIPTION_PROMPT
from src.config.multi_prompt import MULTI_MODAL_EVAL_PROMPT

# =========================
# 工作模式宏控制
# =========================
MODE_DESCRIBE = "describe"
MODE_SCORE = "score"

# 当前运行模式
CURRENT_MODE = MODE_DESCRIBE

# 数据集目录
RAW_VIDEO_DIR = os.path.join(PROJECT_ROOT, "data", "raw_videos")

# 输出目录
DESC_RESULT_DIR = os.path.join(PROJECT_ROOT, "data", "video_descriptions", "qwen35_vl")
EVAL_RESULT_DIR = os.path.join(PROJECT_ROOT, "data", "eval_results", "qwen35")

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class QwenVideoEvaluator:
    def __init__(self, mode: str = MODE_DESCRIBE) -> None:
        if mode not in {MODE_DESCRIBE, MODE_SCORE}:
            raise ValueError(f"无效的模式: {mode}")

        if not DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY 未配置，请检查环境变量。")

        self.mode = mode
        self.client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
        )
        self.model = QWEN_MODEL

        self.desc_dir = DESC_RESULT_DIR
        self.result_dir = EVAL_RESULT_DIR

        os.makedirs(self.desc_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

    # =========================
    # 基础工具函数
    # =========================
    @staticmethod
    def _extract_json_text(raw_text: str) -> str:
        """从模型输出中提取 JSON 块"""
        raw_text = raw_text.strip()

        if "```json" in raw_text:
            raw_text = raw_text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```", 1)[1].split("```", 1)[0].strip()

        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return raw_text
        return raw_text[start:end + 1]

    @staticmethod
    def _video_to_data_url(video_path: str) -> str:
        """本地视频转 Base64"""
        with open(video_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:video/mp4;base64,{encoded}"

    def _build_video_payload(self, video_path: str, public_url: Optional[str]) -> str:
        if public_url:
            return public_url

        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if file_size_mb > 20:
            raise ValueError(
                f"视频文件过大 ({file_size_mb:.2f}MB)，当前实现使用 base64 直传，请改为公网 URL 或先压缩。"
            )
        return self._video_to_data_url(video_path)

    def _request_qwen_api(self, video_url: str, prompt_text: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": video_url},
                        },
                        {
                            "type": "text",
                            "text": prompt_text,
                        },
                    ],
                }
            ],
            temperature=0,
            extra_body={"enable_thinking": False},
        )
        return completion.choices[0].message.content.strip()

    @staticmethod
    def _collect_videos(dataset_dir: str) -> List[str]:
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"找不到源数据集目录: {dataset_dir}")

        video_paths: List[str] = []
        for root, _, files in os.walk(dataset_dir):
            for file_name in files:
                ext = os.path.splitext(file_name)[1].lower()
                if ext in SUPPORTED_VIDEO_EXTENSIONS:
                    video_paths.append(os.path.join(root, file_name))

        video_paths.sort()
        return video_paths

    @staticmethod
    def _build_output_path(
        video_path: str,
        dataset_dir: str,
        output_dir: str,
        suffix: str,
    ) -> str:
        """
        按照源数据集相对目录结构保存结果，避免不同子目录下同名视频互相覆盖
        例如:
        raw_videos/a/1.mp4 -> output_dir/a/1_desc.json
        raw_videos/b/1.mp4 -> output_dir/b/1_desc.json
        """
        rel_path = os.path.relpath(video_path, dataset_dir)
        rel_stem = os.path.splitext(rel_path)[0]
        save_path = os.path.join(output_dir, f"{rel_stem}{suffix}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path

    @staticmethod
    def _safe_prompt_substitute(prompt_obj: Any, video_id: str) -> str:
        if hasattr(prompt_obj, "safe_substitute"):
            return prompt_obj.safe_substitute(video_id=video_id)
        if isinstance(prompt_obj, str):
            return prompt_obj.replace("${video_id}", video_id)
        raise TypeError("prompt 必须是 string 或 Template 对象")

    # =========================
    # 双模式核心逻辑
    # =========================
    def run_describe(
        self,
        video_path: str,
        dataset_dir: str,
        public_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        [描述模式]：生成结构化、可读性高的 JSON 文件
        这里保留源文件的视频描述逻辑
        """
        video_stem = Path(video_path).stem
        video_url = self._build_video_payload(video_path, public_url)

        prompt_text = self._safe_prompt_substitute(VIDEO_DESCRIPTION_PROMPT, video_stem)

        print(f"[DESCRIBE] 正在生成视频描述: {video_path}")
        raw_desc = self._request_qwen_api(video_url, prompt_text)

        try:
            json_str = self._extract_json_text(raw_desc)
            parsed_content = json.loads(json_str)
        except Exception as e:
            print(f"警告: 描述结果 JSON 解析失败，保存原始文本。错误: {e}")
            parsed_content = {"raw_content": raw_desc, "parse_error": str(e)}

        if isinstance(parsed_content, dict):
            parsed_content.pop("video_id", None)
            result = {
                "video_id": video_stem,
                "video_path": video_path,
                "mode": "video_description",
                **parsed_content,
            }
        else:
            result = {
                "video_id": video_stem,
                "video_path": video_path,
                "mode": "video_description",
                "description": parsed_content,
            }

        json_path = self._build_output_path(
            video_path=video_path,
            dataset_dir=dataset_dir,
            output_dir=self.desc_dir,
            suffix="_desc.json",
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        print(f"描述已保存至 JSON: {json_path}")
        return result

    def run_score(
        self,
        video_path: str,
        dataset_dir: str,
        public_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        [评分模式]：直接用 Qwen3.5 多模态能力进行安全评测
        prompt 来自 src/config/multi_prompt.py
        """
        video_stem = Path(video_path).stem
        video_url = self._build_video_payload(video_path, public_url)

        prompt_text = self._safe_prompt_substitute(MULTI_MODAL_EVAL_PROMPT, video_stem)

        print(f"[SCORE] 正在进行多模态安全评分: {video_path}")
        raw_output = self._request_qwen_api(video_url, prompt_text)

        try:
            json_str = self._extract_json_text(raw_output)
            score_data = json.loads(json_str)
            parse_success = True
        except Exception as e:
            print(f"警告: 评分结果 JSON 解析失败，记录原始文本。错误: {e}")
            score_data = {
                "raw_content": raw_output,
                "parse_error": str(e),
            }
            parse_success = False

        result = {
            "video_id": video_stem,
            "video_path": video_path,
            "mode": "safety_evaluation",
            "model": self.model,
            "prompt_source": "src/config/multi_prompt.py",
            "parse_success": parse_success,
            "eval_result": score_data,
        }

        save_path = self._build_output_path(
            video_path=video_path,
            dataset_dir=dataset_dir,
            output_dir=self.result_dir,
            suffix="_score.json",
        )

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        print(f"评分已保存至 JSON: {save_path}")
        return result

    def run(
        self,
        video_path: str,
        dataset_dir: str,
        public_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.mode == MODE_DESCRIBE:
            return self.run_describe(video_path, dataset_dir, public_url)
        elif self.mode == MODE_SCORE:
            return self.run_score(video_path, dataset_dir, public_url)
        else:
            raise ValueError(f"不支持的模式: {self.mode}")

    def run_on_dataset(
        self,
        dataset_dir: str = RAW_VIDEO_DIR,
    ) -> List[Dict[str, Any]]:
        video_paths = self._collect_videos(dataset_dir)
        if not video_paths:
            print(f"[WARN] 数据集目录下未找到视频: {dataset_dir}")
            return []

        print(f"[INFO] 当前模式: {self.mode}")
        print(f"[INFO] 共发现 {len(video_paths)} 个视频，开始处理")

        all_results: List[Dict[str, Any]] = []
        success_count = 0
        fail_count = 0

        for idx, video_path in enumerate(video_paths, start=1):
            print(f"\n[INFO] 处理进度: {idx}/{len(video_paths)}")
            try:
                result = self.run(video_path=video_path, dataset_dir=dataset_dir)
                all_results.append(result)

                if self.mode == MODE_SCORE:
                    if result.get("parse_success", False):
                        success_count += 1
                    else:
                        fail_count += 1
                else:
                    success_count += 1

            except Exception as e:
                fail_count += 1
                error_result = {
                    "video_id": Path(video_path).stem,
                    "video_path": video_path,
                    "mode": self.mode,
                    "error": str(e),
                }
                all_results.append(error_result)
                print(f"[ERROR] 处理失败: {video_path} -> {e}")

        summary = {
            "mode": self.mode,
            "dataset_dir": dataset_dir,
            "total_videos": len(video_paths),
            "success_count": success_count,
            "fail_count": fail_count,
            "model": self.model,
        }

        if self.mode == MODE_DESCRIBE:
            summary_path = os.path.join(self.desc_dir, "_describe_summary.json")
        else:
            summary_path = os.path.join(self.result_dir, "_score_summary.json")

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)

        print("\n[INFO] 全部处理完成")
        print(f"[INFO] 成功数量: {success_count}")
        print(f"[INFO] 失败数量: {fail_count}")
        print(f"[INFO] 汇总文件: {summary_path}")

        return all_results


def main():
    dataset_dir = RAW_VIDEO_DIR

    if not os.path.exists(dataset_dir):
        print(f"错误: 找不到源数据集目录 -> {dataset_dir}")
        return

    print(f"=== Qwen Evaluator 启动 | 当前模式: {CURRENT_MODE} ===")
    evaluator = QwenVideoEvaluator(mode=CURRENT_MODE)
    evaluator.run_on_dataset(dataset_dir)


if __name__ == "__main__":
    main()