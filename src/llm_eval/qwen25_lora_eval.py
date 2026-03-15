# src/llm_eval/qwen25_lora_eval.py
# Qwen2.5核心推理部分

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LORA_ADAPTER_PATH = "models/qwen25_lora_adapter"

class Qwen25LoraEvaluator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
        self.model.eval()

    def build_prompt(self, scene_text: str) -> str:
        return (
            "请根据给定的自动驾驶视频文本描述，从 Semantic、Logical、Decision "
            "三个维度进行安全评测，并输出严格合法的 JSON。\n\n"
            f"{scene_text}"
        )

    def evaluate(self, scene_text: str) -> str:
        prompt = self.build_prompt(scene_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    evaluator = Qwen25LoraEvaluator()
    test_text = (
        "Video ID: 001\n"
        "Timeline:\n"
        "t=0s: ego vehicle moves straight.\n"
        "t=1s: pedestrian starts crossing.\n"
        "t=2s: ego vehicle continues moving without braking."
    )
    print(evaluator.evaluate(test_text))