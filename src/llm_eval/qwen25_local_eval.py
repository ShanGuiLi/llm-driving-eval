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


MODE_DESCRIBE = "describe"
MODE_SCORE = "score"

# 手动切换运行模式
CURRENT_MODE = MODE_SCORE


class LLMDrivingEvaluator:
    def __init__(self, mode: str = MODE_DESCRIBE) -> None:
        if mode not in {MODE_DESCRIBE, MODE_SCORE}:
            raise ValueError(f"Invalid mode: {mode}")

        self.mode = mode
        self.project_root = Path(PROJECT_ROOT)
        self.vlm_model_path = self._resolve_model_path(LOCAL_VLM_MODEL)
        self.llm_model_path = self._resolve_model_path(LOCAL_LLM_MODEL)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dtype = self._get_torch_dtype()

        print(f"当前设备: {self.device}")
        print(f"当前精度: {self.model_dtype}")
        print(f"VLM模型目录: {self.vlm_model_path}")
        print(f"LLM模型目录: {self.llm_model_path}")

        # 目录管理
        self.raw_video_dir = self.project_root / "data" / "raw_videos" / "mixed_group_06"
        self.desc_root_dir = self.project_root / "data" / "video_descriptions"
        self.qwen25_desc_dir = self.desc_root_dir / "qwen25_vl"
        self.qwen35_desc_dir = self.desc_root_dir / "qwen35_vl" / "mixed_group_06"
        self.result_dir = Path(EVAL_RESULT_DIR)

        self.qwen25_desc_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        # VLM
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

        # LLM
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
                max_new_tokens=1024,
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

    @staticmethod
    def _parse_vlm_output_to_dict(raw_text: str) -> Dict[str, Any]:
        """
        目标：把 VLM 输出解析成真正的 dict，而不是存成 description 字符串。
        """
        json_text = LLMDrivingEvaluator._extract_json_text(raw_text)
        parsed = json.loads(json_text)

        if isinstance(parsed, dict):
            return parsed

        raise ValueError("VLM output is not a JSON object")

    def describe_video_to_json(self, video_path: str) -> Dict[str, Any]:
        video_stem = Path(video_path).stem
        raw_description = self.generate_video_description_with_vlm(video_path)
        parsed_description = self._parse_vlm_output_to_dict(raw_description)

        # 避免内外 video_id 重复
        parsed_description.pop("video_id", None)

        result = {
            "video_id": video_stem,
            **parsed_description
        }

        save_path = self.qwen35_desc_dir / f"{video_stem}.json"
        with open(save_path, "w", encoding="utf-8") as output_file:
            json.dump(result, output_file, ensure_ascii=False, indent=4)

        print(f"视频描述已保存: {save_path}")
        return result

    def describe_videos_in_directory(self, video_dir: str, recursive: bool = True) -> List[Dict[str, Any]]:
        video_dir_path = Path(video_dir)
        if not video_dir_path.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")

        if recursive:
            video_files = sorted([
                p for p in video_dir_path.rglob("*")
                if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
            ])
        else:
            video_files = sorted([
                p for p in video_dir_path.iterdir()
                if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
            ])

        if not video_files:
            print(f"未找到视频文件: {video_dir}")
            return []

        all_results: List[Dict[str, Any]] = []
        print(f"共发现 {len(video_files)} 个视频，开始生成描述...")

        for idx, video_file in enumerate(video_files, start=1):
            print(f"\n[{idx}/{len(video_files)}] 处理视频: {video_file}")
            try:
                result = self.describe_video_to_json(str(video_file))
                all_results.append(result)
            except Exception as error:
                print(f"处理失败: {video_file} - {error}")
                all_results.append({
                    "video_id": video_file.stem,
                    "video_path": str(video_file),
                    "error": str(error)
                })

        return all_results

    @staticmethod
    def _load_description_json(description_json_path: str) -> Dict[str, Any]:
        if not os.path.exists(description_json_path):
            raise FileNotFoundError(f"Description JSON not found: {description_json_path}")

        with open(description_json_path, "r", encoding="utf-8") as input_file:
            data = json.load(input_file)

        if not isinstance(data, dict):
            raise ValueError(f"Description JSON root must be an object: {description_json_path}")

        return data

    @staticmethod
    def _extract_video_description_text(description_data: Dict[str, Any]) -> str:
        """
        评分阶段把结构化 JSON 再序列化成 prompt 可读文本。
        外层 video_id 不参与评分描述内容。
        """
        description_content = {
            key: value
            for key, value in description_data.items()
            if key != "video_id"
        }

        return json.dumps(description_content, ensure_ascii=False, indent=2)

    @staticmethod
    def _extract_video_id(description_json_path: str, description_data: Dict[str, Any]) -> str:
        video_id = description_data.get("video_id")
        if isinstance(video_id, str) and video_id.strip():
            return video_id.strip()

        return Path(description_json_path).stem

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
                max_new_tokens=512,
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
            reasoning = "Ground truth itself is unsafe, so this sample is excluded from normal evaluation."
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
                    reasoning = f"The generated output contains issues at {', '.join(attack_level)} level."
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
        description_json_path: str,
        error_message: str
    ) -> Dict[str, Any]:
        return {
            "video_id": video_name,
            "description_json_path": description_json_path,
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

    def evaluate_description_json(
        self,
        description_json_path: str,
        ground_truth_status: str = "safe"
    ) -> Dict[str, Any]:
        try:
            description_data = self._load_description_json(description_json_path)
            video_id = self._extract_video_id(description_json_path, description_data)
            video_description = self._extract_video_description_text(description_data)

            print(f"\n当前评测文件: {description_json_path}")
            print(f"\n送入LLM的描述如下:\n{video_description}\n")

            eval_prompt = DRIVING_SAFETY_EVAL_PROMPT.substitute(
                video_id=video_id,
                ground_truth_status=ground_truth_status,
                video_desc=video_description
            )

            llm_response_text = self._generate_llm_text(eval_prompt)
            print(f"\n评分模型原始输出如下:\n{llm_response_text}\n")

            json_text = self._extract_json_text(llm_response_text)
            raw_result = json.loads(json_text)

            eval_result = self._normalize_result(
                raw_result=raw_result,
                video_name=video_id,
                ground_truth_status=ground_truth_status
            )

            eval_result["description_json_path"] = description_json_path
            eval_result["vlm_model"] = self.vlm_model_path
            eval_result["llm_model"] = self.llm_model_path

            result_file = self.result_dir / f"{video_id}.json"
            with open(result_file, "w", encoding="utf-8") as output_file:
                json.dump(eval_result, output_file, ensure_ascii=False, indent=4)

            print(f"评测完成: {video_id}")
            print(f"结果已保存到: {result_file}")
            return eval_result

        except Exception as error:
            video_name = Path(description_json_path).stem
            error_result = self._build_error_result(
                video_name=video_name,
                description_json_path=description_json_path,
                error_message=f"Unexpected error: {error}"
            )

            result_file = self.result_dir / f"{video_name}.json"
            with open(result_file, "w", encoding="utf-8") as output_file:
                json.dump(error_result, output_file, ensure_ascii=False, indent=4)

            print(f"评测失败: {video_name} - {error}")
            print(f"错误结果已保存到: {result_file}")
            return error_result

    def evaluate_description_directory(
        self,
        description_dir: str,
        ground_truth_status: str = "safe"
    ) -> List[Dict[str, Any]]:
        description_dir_path = Path(description_dir)
        if not description_dir_path.exists():
            raise FileNotFoundError(f"Description directory not found: {description_dir}")

        json_files = sorted(description_dir_path.glob("*.json"))
        if not json_files:
            print(f"未找到描述文件: {description_dir}")
            return []

        all_results: List[Dict[str, Any]] = []
        print(f"共发现 {len(json_files)} 个描述文件，开始评分...")

        for idx, json_file in enumerate(json_files, start=1):
            print(f"\n[{idx}/{len(json_files)}] 处理描述文件: {json_file}")
            result = self.evaluate_description_json(
                description_json_path=str(json_file),
                ground_truth_status=ground_truth_status
            )
            all_results.append(result)

        return all_results


def main():
    evaluator = LLMDrivingEvaluator(mode=CURRENT_MODE)

    print(f"=== Qwen2.5 Local Evaluator 启动 | 当前模式: {CURRENT_MODE} ===")

    if CURRENT_MODE == MODE_DESCRIBE:
        evaluator.describe_videos_in_directory(
            video_dir=str(evaluator.raw_video_dir),
            recursive=True
        )
    elif CURRENT_MODE == MODE_SCORE:
        evaluator.evaluate_description_directory(
            description_dir=str(evaluator.qwen35_desc_dir),
            ground_truth_status="safe"
        )


if __name__ == "__main__":
    main()