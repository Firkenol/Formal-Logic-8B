import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

model_id = "meta-llama/Meta-Llama-3.1-8B"

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Setup 4-Bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 3. Load Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# 4. Setup LoRA Config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 5. Load Dataset (Change file name if you renamed it earlier)
dataset = load_dataset("json", data_files="thrm_dataset.jsonl", split="train")

# 6. Setup New SFTConfig (Arguments moved here)
training_args = SFTConfig(
    output_dir="./thrm_3b_qlora_final",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=500,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="wandb",
    dataset_text_field="text",
    max_seq_length=2048,
)

# 7. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=peft_config,
)

# 8. Run Training
trainer.train()

# 9. Save Weights
trainer.model.save_pretrained("./thrm_3b_qlora_final")
tokenizer.save_pretrained("./thrm_3b_qlora_final")