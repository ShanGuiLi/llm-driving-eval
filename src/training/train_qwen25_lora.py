# src/training/train_qwen25_lora.py

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

from src.config.sft_config import (
    BASE_MODEL_NAME,
    TRAIN_FILE,
    VAL_FILE,
    OUTPUT_DIR,
    MAX_LENGTH,
    LEARNING_RATE,
    NUM_EPOCHS,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PER_DEVICE_EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    USE_4BIT,
    MAX_STEPS,
    LOGGING_STEPS,
    SAVE_STEPS,
    EVAL_STEPS,
)


def build_text(example):
    """构造Qwen聊天格式文本。"""
    system_prompt = (
        "You are an autonomous driving video safety evaluation assistant. "
        "You must analyze the given driving scenario and return a concise structured judgment."
    )

    user_prompt = (
        f"{example['instruction']}\n\n"
        f"Input:\n{example['input']}\n\n"
        f"Return the result in JSON format."
    )

    assistant_response = example["output"]

    text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_response}<|im_end|>"
    )
    return {"text": text}


def tokenize_function(example, tokenizer):
    """分词并构建labels。"""
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


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
        print("未检测到可用GPU。")
    print("============================")


def load_tokenizer():
    """加载分词器。"""
    print("开始加载Tokenizer...")
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


def load_model():
    """加载基础模型。"""
    print("开始加载基础模型...")

    if USE_4BIT:
        print("当前使用4bit量化加载模型。")
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

        print("基础模型加载完成，开始执行k-bit训练预处理...")
        model = prepare_model_for_kbit_training(model)
        print("k-bit训练预处理完成。")
    else:
        print("当前不使用量化，采用fp16加载模型。")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    return model


def apply_lora(model):
    """注入LoRA。"""
    print("开始注入LoRA配置...")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("LoRA注入完成。")
    return model


def load_and_prepare_dataset(tokenizer):
    """加载并处理数据集。"""
    print("开始加载数据集...")

    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"训练集不存在: {TRAIN_FILE}")
    if not os.path.exists(VAL_FILE):
        raise FileNotFoundError(f"验证集不存在: {VAL_FILE}")

    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": VAL_FILE,
        },
    )

    print("原始数据集加载完成。")
    print(dataset)

    print("开始构造对话文本...")
    dataset = dataset.map(build_text)

    print("开始分词...")
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        remove_columns=dataset["train"].column_names,
    )

    print("数据集处理完成。")
    return dataset


def build_trainer(model, tokenizer, dataset):
    """构建Trainer。"""
    print("开始构建TrainingArguments...")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        gradient_checkpointing=True,
        report_to="none",
        max_steps=MAX_STEPS,
        remove_unused_columns=False,
    )

    print("TrainingArguments构建完成。")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )

    print("Trainer构建完成。")
    return trainer


def sanity_check(tokenizer):
    """简单检查模板。"""
    print("开始执行模板检查...")
    sample = {
        "instruction": "Evaluate whether the driving video is safe.",
        "input": "The ego vehicle slows down and stops at a red light.",
        "output": "{\"score\": 5, \"risk\": \"low\", \"decision\": \"safe\"}",
    }

    text = build_text(sample)["text"]
    print("模板样例预览：")
    print(text[:500])

    tokenized = tokenizer(text, truncation=True, max_length=MAX_LENGTH)
    print(f"模板检查完成，token数量: {len(tokenized['input_ids'])}")


def main():
    print("========== 开始执行Qwen2.5 LoRA训练流程 ==========")
    print_gpu_info()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = load_tokenizer()
    sanity_check(tokenizer)

    model = load_model()
    model = apply_lora(model)

    dataset = load_and_prepare_dataset(tokenizer)
    trainer = build_trainer(model, tokenizer, dataset)

    print("开始训练...")
    trainer.train()
    print("训练完成。")

    print("开始保存LoRA Adapter和Tokenizer...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA Adapter已保存到: {OUTPUT_DIR}")

    print("========== 全部流程执行完成 ==========")


if __name__ == "__main__":
    main()