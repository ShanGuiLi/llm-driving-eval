# src/training/train_qwen25_lora.py
# Qwen2.5核心训练部分

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
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
    USE_8BIT
)

def build_text(example):
    text = (
        f"<|im_start|>system\n"
        f"你是一名自动驾驶视频安全评测助手。<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{example['instruction']}\n\n{example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{example['output']}<|im_end|>"
    )
    return {"text": text}

def tokenize_function(example, tokenizer):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        load_in_8bit=USE_8BIT,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 你贴的 prepare_model_for_int8_training 在新版本里通常换成这个
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        # 建议先只训attention投影层，稳定一点
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        # 如果后面显存和效果允许，再尝试加上：
        # ["gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": VAL_FILE
        }
    )

    dataset = dataset.map(build_text)
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        remove_columns=dataset["train"].column_names
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_total_limit=2,
        gradient_checkpointing=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    # 保存LoRA adapter
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()