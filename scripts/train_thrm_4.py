import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# manually wrap peft since we dropped SFTTrainer
model = get_peft_model(model, peft_config)

dataset = load_dataset("json", data_files="thrm_training_v2.jsonl", split="train")

def mask_prompt_labels(example):
    full_text = example["text"]
    separator = "<|start_header_id|>assistant<|end_header_id|>\n\n"

    parts = full_text.split(separator)
    if len(parts) >= 2:
        prompt_str = parts[0] + separator
    else:
        prompt_str = full_text

    tokenized = tokenizer(full_text, truncation=True, max_length=4096)
    prompt_tokenized = tokenizer(prompt_str, truncation=True, max_length=4096)

    prompt_len = len(prompt_tokenized["input_ids"])

    labels = tokenized["input_ids"].copy()
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100

    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(mask_prompt_labels, remove_columns=["text"])

collator = DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True)

training_args = TrainingArguments(
    output_dir="./thrm_qlora_v4",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=500,
    bf16=True,
    optim="paged_adamw_8bit",
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=collator,
)

trainer.train()

trainer.model.save_pretrained("./thrm_qlora_v4")
tokenizer.save_pretrained("./thrm_qlora_v4")