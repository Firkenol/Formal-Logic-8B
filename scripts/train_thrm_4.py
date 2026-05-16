import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

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

dataset = load_dataset("json", data_files="thrm_training_v2.jsonl", split="train")

response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# dropped SFTConfig for standard TrainingArguments to bypass trl version mismatches
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

# moved max_seq_length directly into SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=peft_config,
    data_collator=collator,
    max_seq_length=4096,
)

trainer.train()

trainer.model.save_pretrained("./thrm_qlora_v4")
tokenizer.save_pretrained("./thrm_qlora_v4")