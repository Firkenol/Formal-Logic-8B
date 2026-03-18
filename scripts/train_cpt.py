import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import huggingface_hub
import torch

# Authenticate with Azure Key Vault to retrieve Hugging Face token
key_vault_name = os.environ.get("KEY_VAULT_NAME")
if not key_vault_name:
    raise ValueError("KEY_VAULT_NAME environment variable is not set.")

kv_uri = f"https://{key_vault_name}.vault.azure.net"
credential = DefaultAzureCredential()
client = SecretClient(vault_url=kv_uri, credential=credential)

hf_token = client.get_secret("HF-TOKEN").value
huggingface_hub.login(token=hf_token)

# Project configuration
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
output_dir = "./cpt-informal-model"

def main():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_cache=False
    )

    # Load dataset
    # Replace 'path/to/dataset' with the actual Hugging Face dataset path or local file
    dataset = load_dataset("text", data_files={"train": "data/train.txt"})
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=2048)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
        num_train_epochs=1,
        bf16=True,
        report_to="wandb",
        deepspeed="scripts/deepspeed_zero3.yaml"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )

    trainer.train()
    
    # Save the final model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()