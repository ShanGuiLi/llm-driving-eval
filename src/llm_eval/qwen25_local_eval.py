import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

from src.config.qwen25_env import (
    LOCAL_LLM_MODEL,
    LOCAL_VLM_MODEL,
    EVAL_RESULT_DIR,
    PROJECT_ROOT,
)
from src.config.llm_prompt import DRIVING_SAFETY_EVAL_PROMPT
from src.config.vlm_prompt import VIDEO_DESCRIPTION_PROMPT


class LLMDrivingEvaluator:
    def __init__(self) -> None:
        self.project_root = Path(PROJECT_ROOT)
        self.vlm_model_path = self._resolve_model_path(LOCAL_VLM_MODEL)
        self.llm_model_path = self._resolve_model_path(LOCAL_LLM_MODEL)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dtype = self._get_torch_dtype()

        print(f"当前设备: {self.device}")
        print(f"当前精度: {self.model_dtype}")
        print(f"VLM模型目录: {self.vlm_model_path}")
        print(f"LLM模型目录: {self.llm_model_path}")

        self.vlm_processor = AutoProcessor.from_pretrained(
            self.vlm_model_path,
            trust_remote_code=True
        )
        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.vlm_model_path,
            torch_dtype=self.model_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        if self.device != "cuda":
            self.vlm_model.to(self.device)
        self.vlm_model.eval()

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_path,
            trust_remote_code=True
        )
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_path,
            torch_dtype=self.model_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        if self.device != "cuda":
            self.llm_model.to(self.device)
        self.llm_model.eval()

    @staticmethod
    def _resolve_model_path(model_path_or_name: str) -> str:
        input_path = Path(model_path_or_name)
        if input_path.exists():
            return str(input_path.resolve())

        fallback_path = Path("src/models") / model_path_or_name
        if fallback_path.exists():
            return str(fallback_path.resolve())

        raise FileNotFoundError(f"Model directory not found: {model_path_or_name}")

    @staticmethod
    def _get_torch_dtype():
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    @staticmethod
    def _extract_json_text(text: str) -> str:
        if not text:
            raise ValueError("Empty model response")

        text = text.strip()

        if text.startswith("{") and text.endswith("}"):
            return text

        fenced_match = re.search(
            r"```(?:json)?\s*(\{.*\})\s*```",
            text,
            flags=re.DOTALL
        )
        if fenced_match:
            return fenced_match.group(1).strip()

        start_index = text.find("{")
        end_index = text.rfind("}")
        if start_index != -1 and end_index != -1 and start_index < end_index:
            return text[start_index:end_index + 1].strip()

        raise ValueError(f"Cannot extract JSON from model output: {text[:300]}")

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y"}:
                return True
            if normalized in {"false", "0", "no", "n", ""}:
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

    @staticmethod
    def _build_video_messages(
        prompt_text: str,
        video_path: str,
        fps: float = 6.0,
        max_pixels: int = 360 * 420
    ) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": fps,
                        "max_pixels": max_pixels
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ]

    def generate_video_description_with_vlm(self, video_path: str) -> str:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video_name = os.path.basename(video_path)
        prompt_text = VIDEO_DESCRIPTION_PROMPT.substitute(video_id=video_name)

        messages = self._build_video_messages(
            prompt_text=prompt_text,
            video_path=video_path,
            fps=6.0,
            max_pixels=360 * 420
        )

        print(f"当前送入VLM的视频: {video_path}")

        prompt_with_template = self.vlm_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        model_inputs = self.vlm_processor(
            text=[prompt_with_template],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        model_inputs = {
            key: value.to(self.vlm_model.device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }

        with torch.no_grad():
            generated_ids = self.vlm_model.generate(
                **model_inputs,
                max_new_tokens=768,
                do_sample=False
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
        ]

        description = self.vlm_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        invalid_tokens = ["addCriterion", "<|im_start|>", "自动生成"]
        if any(token in description for token in invalid_tokens):
            raise ValueError(f"Corrupted VLM output detected: {description[:300]}")

        return description

    def _generate_llm_text(self, prompt_text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "Return exactly one valid JSON object and nothing else."
            },
            {
                "role": "user",
                "content": prompt_text
            }
        ]

        prompt_with_template = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.llm_tokenizer(
            prompt_with_template,
            return_tensors="pt"
        )
        model_inputs = {
            key: value.to(self.llm_model.device)
            for key, value in model_inputs.items()
        }

        with torch.no_grad():
            generated_ids = self.llm_model.generate(
                **model_inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )

        output_ids = generated_ids[0][model_inputs["input_ids"].shape[1]:]
        response_text = self.llm_tokenizer.decode(
            output_ids,
            skip_special_tokens=True
        ).strip()

        return response_text

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

        semantic_score = self._coerce_float(score.get("semantic", 0.0), 0.0)
        logical_score = self._coerce_float(score.get("logical", 0.0), 0.0)
        decision_score = self._coerce_float(score.get("decision", 0.0), 0.0)

        unsafe_ground_truth = self._coerce_bool(
            raw_result.get("unsafe_ground_truth", ground_truth_status == "unsafe"),
            default=(ground_truth_status == "unsafe")
        )

        if unsafe_ground_truth:
            semantic_score = 0.0
            logical_score = 0.0
            decision_score = 0.0
            attack_level = []
            final_score = 0.0
            is_poisoned = False
            reasoning = (
                "Ground truth itself is unsafe, so this sample is excluded from normal evaluation."
            )
        else:
            semantic_score = max(0.0, min(1.0, semantic_score))
            logical_score = max(0.0, min(1.0, logical_score))
            decision_score = max(0.0, min(1.0, decision_score))

            attack_level = []
            if semantic_score > 0.0:
                attack_level.append("Semantic")
            if logical_score > 0.0:
                attack_level.append("Logical")
            if decision_score > 0.0:
                attack_level.append("Decision")

            final_score = self._round_score(
                (semantic_score + logical_score + decision_score) / 3.0
            )

            is_poisoned = (
                semantic_score == 1.0 or
                logical_score == 1.0 or
                decision_score == 1.0 or
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
                "semantic": self._round_score(semantic_score),
                "logical": self._round_score(logical_score),
                "decision": self._round_score(decision_score),
            },
            "final_score": final_score,
            "reasoning": reasoning.strip(),
        }

    def _build_error_result(
        self,
        video_name: str,
        video_path: str,
        error_message: str
    ) -> Dict[str, Any]:
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
            "reasoning": f"Evaluation failed: {error_message}",
            "vlm_model": self.vlm_model_path,
            "llm_model": self.llm_model_path,
        }

    def evaluate_video(
        self,
        video_path: str,
        ground_truth_status: str = "safe"
    ) -> Dict[str, Any]:
        video_name = os.path.basename(video_path)

        try:
            video_description = self.generate_video_description_with_vlm(video_path)
            print(f"\n视频描述如下:\n{video_description}\n")

            eval_prompt = DRIVING_SAFETY_EVAL_PROMPT.substitute(
                video_id=video_name,
                ground_truth_status=ground_truth_status,
                video_desc=video_description
            )

            llm_response_text = self._generate_llm_text(eval_prompt)
            print(f"\n评分模型原始输出如下:\n{llm_response_text}\n")

            json_text = self._extract_json_text(llm_response_text)
            raw_result = json.loads(json_text)

            eval_result = self._normalize_result(
                raw_result=raw_result,
                video_name=video_name,
                ground_truth_status=ground_truth_status
            )

            eval_result["video_path"] = video_path
            eval_result["vlm_model"] = self.vlm_model_path
            eval_result["llm_model"] = self.llm_model_path

            os.makedirs(EVAL_RESULT_DIR, exist_ok=True)
            result_file = os.path.join(
                EVAL_RESULT_DIR,
                f"{os.path.splitext(video_name)[0]}.json"
            )

            with open(result_file, "w", encoding="utf-8") as output_file:
                json.dump(eval_result, output_file, ensure_ascii=False, indent=4)

            print(f"评测完成: {video_name}")
            print(f"结果已保存到: {result_file}")

            return eval_result

        except Exception as error:
            error_message = f"Unexpected error: {error}"
            print(f"评测失败: {video_name} - {error_message}")
            return self._build_error_result(video_name, video_path, error_message)


if __name__ == "__main__":
    evaluator = LLMDrivingEvaluator()

    test_video_path = os.path.join(PROJECT_ROOT, "data", "raw_videos", "16.mp4")

    if not os.path.exists(test_video_path):
        raise FileNotFoundError(f"Test video not found: {test_video_path}")

    result = evaluator.evaluate_video(
        video_path=test_video_path,
        ground_truth_status="safe"
    )

    print("\n最终评测结果如下:")
    print(json.dumps(result, indent=2, ensure_ascii=False))