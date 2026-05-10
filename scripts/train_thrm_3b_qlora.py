import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# load generated data
data = load_dataset("json", data_files="thrm_cpt_training_data.jsonl", split="train")

# setup tokenizer and base model
model_id = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True
)
model = prepare_model_for_kbit_training(model)

# lora config
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

# basic training args
training_args = TrainingArguments(
    output_dir="./thrm_3b_qlora_final",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=500,
    fp16=True,
    optim="paged_adamw_8bit"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    dataset_text_field="text",
    max_seq_length=2048,
    args=training_args
)

trainer.train()
trainer.model.save_pretrained("thrm_3b_qlora_final")