# src/llm_eval/qwen25_lora_eval.py

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# =========================
# 基础配置
# =========================

BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LORA_ADAPTER_PATH = "models/qwen25_lora_ckpt"

USE_4BIT = True
MAX_NEW_TOKENS = 256


def print_gpu_info():
    """打印GPU信息。"""
    print("========== GPU信息 ==========")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"检测到GPU数量: {device_count}")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_mem_gb = props.total_memory / 1024**3
            print(f"GPU {i}: {props.name}, 总显存: {total_mem_gb:.2f} GB")
    else:
        print("未检测到可用GPU，将尝试在CPU上运行。")
    print("============================")


def load_tokenizer():
    """加载Tokenizer。优先使用LoRA目录中的tokenizer。"""
    print("开始加载Tokenizer...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            LORA_ADAPTER_PATH,
            use_fast=False,
            trust_remote_code=True,
        )
        print("已从LoRA目录加载Tokenizer。")
    except Exception:
        print("LoRA目录下Tokenizer加载失败，回退到基础模型Tokenizer。")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME,
            use_fast=False,
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer无pad_token，已自动设置为eos_token。")

    print("Tokenizer加载完成。")
    return tokenizer


def load_base_model():
    """加载基础模型。"""
    print("开始加载基础模型...")

    if USE_4BIT and torch.cuda.is_available():
        print("当前使用4bit量化加载基础模型。")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=quantization_config,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        if torch.cuda.is_available():
            print("当前使用fp16加载基础模型。")
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            print("当前使用CPU加载基础模型。")
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                trust_remote_code=True,
            )

    print("基础模型加载完成。")
    return model


def load_lora_model():
    """加载基础模型并挂载LoRA Adapter。"""
    tokenizer = load_tokenizer()
    base_model = load_base_model()

    print("开始挂载LoRA Adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER_PATH,
    )
    print("LoRA Adapter挂载完成。")

    model.eval()
    return tokenizer, model


def build_messages(user_input: str):
    """
    构造聊天消息。
    按你的要求，prompt内容使用英文。
    """
    system_prompt = (
        "You are an autonomous driving video safety evaluation assistant. "
        "Analyze the given driving scenario and return a concise structured judgment in JSON format."
    )

    user_prompt = (
        "Evaluate whether the following driving scenario is safe.\n\n"
        f"Scenario:\n{user_input}\n\n"
        "Return a JSON object with fields: score, risk, decision, reason."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_response(tokenizer, model, user_input: str):
    """执行推理。"""
    messages = build_messages(user_input)

    # 优先使用chat template
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # 兜底方案
        text = (
            f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
            f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    print("========== 输入Prompt预览 ==========")
    print(text[:1000])
    print("====================================")

    inputs = tokenizer(text, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    output_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return response_text


def try_parse_json(text: str):
    """尝试把模型输出解析为JSON。"""
    try:
        return json.loads(text), True
    except Exception:
        pass

    # 尝试截取最外层 JSON
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate), True
        except Exception:
            pass

    return text, False


def main():
    print("========== 开始执行LoRA推理 ==========")
    print_gpu_info()

    tokenizer, model = load_lora_model()

    test_input = (
        "The ego vehicle approaches an intersection with a red traffic light. "
        "A pedestrian is crossing near the crosswalk, and the vehicle slows down and stops before the line."
    )

    print("开始执行推理...")
    response_text = generate_response(tokenizer, model, test_input)

    print("\n========== 原始生成结果 ==========")
    print(response_text)

    parsed_result, is_json = try_parse_json(response_text)

    print("\n========== 解析结果 ==========")
    if is_json:
        print(json.dumps(parsed_result, ensure_ascii=False, indent=2))
    else:
        print("输出不是合法JSON，保留原始文本：")
        print(parsed_result)

    print("========== 推理完成 ==========")


if __name__ == "__main__":
    main()