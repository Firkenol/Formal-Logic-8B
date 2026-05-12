print("--- STARTING V3 TRAINING SCRIPT ---")
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

print("Loading model and tokenizer...")
model_id = "meta-llama/Meta-Llama-3.1-8B"
out_dir = "./thrm_qlora_v3"

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_cfg, device_map="auto")
model = prepare_model_for_kbit_training(model)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

print("Defining LoRA config...")
peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

print("Loading dataset...")
dataset = load_dataset("parquet", data_files="https://huggingface.co/datasets/bigcode/commitpackft/resolve/refs%2Fconvert%2Fparquet/python/train/0000.parquet", split="train[:2000]")

def format_prompt(ex): 
    return f"<|begin_of_text|>Resolve this issue with a git patch.\n\nIssue:\n{ex.get('message', '')}\n\nPatch:\n```diff\n{ex.get('patch', '')}\n```<|end_of_text|>"

args = TrainingArguments(
    output_dir=out_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=300,
    bf16=True,
    optim="paged_adamw_32bit",
    report_to="none"
)

print("Initializing Trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=args,
    processing_class=tokenizer,
    formatting_func=format_prompt
)

print("Starting training loop...")
trainer.train()
trainer.save_model(out_dir)
print("--- TRAINING COMPLETE ---")