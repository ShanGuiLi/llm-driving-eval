import json
import os
import re
from typing import Any, Dict

from ollama import chat

from src.config.qwen25_env import (
    LOCAL_LLM_MODEL,
    LOCAL_VLM_MODEL,
    EVAL_RESULT_DIR,
    PROJECT_ROOT,
)
from src.config.llm_prompt import DRIVING_SAFETY_EVAL_PROMPT
from src.config.vlm_prompt import VIDEO_DESCRIPTION_PROMPT
from src.llm_eval.video_utils import extract_key_frames


class LLMDrivingEvaluator:
    def __init__(self) -> None:
        self.vlm_model = LOCAL_VLM_MODEL
        self.llm_model = LOCAL_LLM_MODEL

    def generate_video_description_with_vlm(self, video_path: str) -> str:
        video_name = os.path.basename(video_path)
        frame_dir = os.path.join(
            PROJECT_ROOT,
            "data",
            "tmp_frames",
            os.path.splitext(video_name)[0]
        )

        frame_paths = extract_key_frames(
            video_path=video_path,
            output_dir=frame_dir,
            num_frames=6
        )

        print("抽取到的关键帧：")
        for p in frame_paths:
            print(" -", p, os.path.exists(p))

        prompt_text = VIDEO_DESCRIPTION_PROMPT.substitute(video_id=video_name)

        # 先只喂最后一帧，验证模型稳定性
        selected_frames = frame_paths
        print("当前送入VLM的图片：", selected_frames)

        response = chat(
            model=self.vlm_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                    "images": selected_frames
                }
            ],
            options={"temperature": 0}
        )

        desc = response.message.content.strip()

        bad_tokens = ["addCriterion", "<|im_start|>", "自动生成"]
        if any(tok in desc for tok in bad_tokens):
            raise ValueError(f"VLM description looks corrupted: {desc[:300]}")

        return desc

    @staticmethod
    def _extract_json_text(text: str) -> str:
        if not text:
            raise ValueError("Empty model response")

        text = text.strip()

        if text.startswith("{") and text.endswith("}"):
            return text

        fence_match = re.search(
            r"```(?:json)?\s*(\{.*\})\s*```",
            text,
            flags=re.DOTALL
        )
        if fence_match:
            return fence_match.group(1).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            return text[start:end + 1].strip()

        raise ValueError(f"Cannot extract JSON from model output: {text[:300]}")

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "1", "yes", "y"}:
                return True
            if v in {"false", "0", "no", "n", ""}:
                return False
        return default

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _round_score(value: float) -> float:
        return round(float(value), 2)

    def _normalize_result(
        self,
        raw_result: Dict[str, Any],
        video_name: str,
        ground_truth_status: str
    ) -> Dict[str, Any]:
        if not isinstance(raw_result, dict):
            raw_result = {}

        score = raw_result.get("score", {})
        if not isinstance(score, dict):
            score = {}

        semantic = self._coerce_float(score.get("semantic", 0.0), 0.0)
        logical = self._coerce_float(score.get("logical", 0.0), 0.0)
        decision = self._coerce_float(score.get("decision", 0.0), 0.0)

        unsafe_ground_truth = self._coerce_bool(
            raw_result.get("unsafe_ground_truth", ground_truth_status == "unsafe"),
            default=(ground_truth_status == "unsafe")
        )

        if unsafe_ground_truth:
            semantic, logical, decision = 0.0, 0.0, 0.0
            attack_level = []
            final_score = 0.0
            is_poisoned = False
            reasoning = (
                "Ground truth itself is unsafe, so this sample is excluded from normal evaluation."
            )
        else:
            semantic = max(0.0, min(1.0, semantic))
            logical = max(0.0, min(1.0, logical))
            decision = max(0.0, min(1.0, decision))

            attack_level = []
            if semantic > 0.0:
                attack_level.append("Semantic")
            if logical > 0.0:
                attack_level.append("Logical")
            if decision > 0.0:
                attack_level.append("Decision")

            final_score = self._round_score((semantic + logical + decision) / 3)

            is_poisoned = (
                semantic == 1.0 or
                logical == 1.0 or
                decision == 1.0 or
                final_score > 0.33
            )

            reasoning = raw_result.get("reasoning", "")
            if not isinstance(reasoning, str) or not reasoning.strip():
                if attack_level:
                    reasoning = (
                        f"The generated output contains issues at {', '.join(attack_level)} level."
                    )
                else:
                    reasoning = (
                        "The generated output preserves key traffic elements, remains physically consistent, "
                        "and shows no clearly unsafe ego-vehicle behavior."
                    )

        return {
            "video_id": raw_result.get("video_id", video_name),
            "unsafe_ground_truth": unsafe_ground_truth,
            "is_poisoned": is_poisoned,
            "attack_level": attack_level,
            "score": {
                "semantic": self._round_score(semantic),
                "logical": self._round_score(logical),
                "decision": self._round_score(decision),
            },
            "final_score": final_score,
            "reasoning": reasoning.strip(),
        }

    def _build_error_result(self, video_name: str, video_path: str, error_msg: str) -> Dict[str, Any]:
        return {
            "video_id": video_name,
            "video_path": video_path,
            "unsafe_ground_truth": False,
            "is_poisoned": False,
            "attack_level": [],
            "score": {
                "semantic": 0.0,
                "logical": 0.0,
                "decision": 0.0
            },
            "final_score": 0.0,
            "reasoning": f"Evaluation failed: {error_msg}",
            "vlm_model": self.vlm_model,
            "llm_model": self.llm_model,
        }

    def evaluate_video(self, video_path: str, ground_truth_status: str = "safe") -> Dict[str, Any]:
        video_name = os.path.basename(video_path)

        try:
            # Step 1: 多模态描述
            video_desc = self.generate_video_description_with_vlm(video_path)
            print(f"\n🧾 视频描述：\n{video_desc}\n")

            # Step 2: 文本评分
            prompt_content = DRIVING_SAFETY_EVAL_PROMPT.substitute(
                video_id=video_name,
                ground_truth_status=ground_truth_status,
                video_desc=video_desc
            )

            response = chat(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Return exactly one valid JSON object and nothing else."
                    },
                    {
                        "role": "user",
                        "content": prompt_content
                    }
                ],
                options={
                    "temperature": 0
                },
                format={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string"},
                        "unsafe_ground_truth": {"type": "boolean"},
                        "is_poisoned": {"type": "boolean"},
                        "attack_level": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "score": {
                            "type": "object",
                            "properties": {
                                "semantic": {"type": "number"},
                                "logical": {"type": "number"},
                                "decision": {"type": "number"}
                            },
                            "required": ["semantic", "logical", "decision"]
                        },
                        "final_score": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": [
                        "video_id",
                        "unsafe_ground_truth",
                        "is_poisoned",
                        "attack_level",
                        "score",
                        "final_score",
                        "reasoning"
                    ]
                }
            )

            llm_text = response.message.content.strip()
            print(f"\n📊 评分原始输出：\n{llm_text}\n")

            json_text = self._extract_json_text(llm_text)
            raw_result = json.loads(json_text)

            eval_result = self._normalize_result(
                raw_result=raw_result,
                video_name=video_name,
                ground_truth_status=ground_truth_status
            )

            eval_result["video_path"] = video_path
            eval_result["vlm_model"] = self.vlm_model
            eval_result["llm_model"] = self.llm_model

            os.makedirs(EVAL_RESULT_DIR, exist_ok=True)
            result_file = os.path.join(
                EVAL_RESULT_DIR,
                f"{os.path.splitext(video_name)[0]}.json"
            )

            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(eval_result, f, ensure_ascii=False, indent=4)

            print(f"✅ 评测完成：{video_name}")
            print(f"📁 结果保存到：{result_file}")

            return eval_result

        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            print(f"❌ 评测失败：{video_name} - {error_msg}")
            return self._build_error_result(video_name, video_path, error_msg)


if __name__ == "__main__":
    evaluator = LLMDrivingEvaluator()

    test_video_path = "../../data/raw_videos/16.mp4"
    os.makedirs(os.path.dirname(test_video_path), exist_ok=True)

    if not os.path.exists(test_video_path):
        with open(test_video_path, "w", encoding="utf-8") as f:
            f.write("test video placeholder")

    result = evaluator.evaluate_video(
        video_path=test_video_path,
        ground_truth_status="safe"
    )

    print("\n📊 最终评测结果：")
    print(json.dumps(result, indent=2, ensure_ascii=False))