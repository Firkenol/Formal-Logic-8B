import torch
import json
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

base_model_id = "meta-llama/Meta-Llama-3.1-8B"
adapter_path = "./thrm_3b_qlora_final"
output_file = "ood_predictions.jsonl"

datasets_to_test = [
    "microsoft/FEA-Bench"
]

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
if tokenizer.pad_token is None: 
    tokenizer.pad_token = tokenizer.eos_token
    
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

with open(output_file, "w") as f:
    for dataset_name in datasets_to_test:
        try:
            dataset = load_dataset(dataset_name, split="test[:100]") 
            
            for i in tqdm(range(len(dataset)), desc=dataset_name):
                instance = dataset[i]
                
                instance_id = instance.get("instance_id", f"{dataset_name}_{i}")
                problem_statement = instance.get("problem_statement", instance.get("text", instance.get("pr_body", "")))
                
                if not problem_statement:
                    continue
                    
                prompt = f"<|begin_of_text|>You are an expert software engineer. Resolve this issue by writing a git patch.\n\nIssue:\n{problem_statement}\n\nPatch:\n```diff\n"
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=1024, 
                        temperature=0.2, 
                        do_sample=True, 
                        pad_token_id=tokenizer.pad_token_id, 
                        eos_token_id=tokenizer.eos_token_id
                    )
                    
                generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                
                patch_text = generated_text
                if "```diff" in generated_text: 
                    patch_text = generated_text.split("```diff")[1].split("```")[0].strip()
                elif "```" in generated_text: 
                    patch_text = generated_text.split("```")[1].strip()
                    
                prediction = {
                    "instance_id": str(instance_id), 
                    "model_patch": patch_text, 
                    "dataset": dataset_name
                }
                f.write(json.dumps(prediction) + "\n")
                f.flush()
        except Exception as e:
            print(f"skipping {dataset_name}: {e}")