import os
import json
import re
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

base_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter_path = "./thrm_qlora_v4"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

print("Loading dataset...")
dataset = load_dataset("parquet", data_files="hf://datasets/bigcode/commitpackft@~parquet/python/train/*.parquet", split="train[:100]")

def format_prompt(problem_statement):
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a senior software engineer. Output a unified git diff to fix the issue. "
        "Use the standard --- a/file and +++ b/file format.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{problem_statement}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def grade_patches(file_path):
    results = {"total": 0, "valid_syntax": 0, "valid_targets": 0, "perfect_structure": 0}
    with open(file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                patch = data.get("model_patch", "")
                if not patch:
                    continue
                results["total"] += 1
                has_syntax = "--- " in patch and "+++ " in patch
                if has_syntax: results["valid_syntax"] += 1
                has_targets = re.search(r'--- a/.*?\n\+\+\+ b/.*?', patch) is not None
                if has_targets: results["valid_targets"] += 1
                has_hunks = re.search(r'@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@', patch) is not None
                if has_syntax and has_targets and has_hunks: results["perfect_structure"] += 1
            except json.JSONDecodeError:
                continue
    return results

def run_inference(model, tokenizer, output_file, desc):
    model.eval()
    with open(output_file, "w") as f:
        for i in tqdm(range(len(dataset)), desc=desc):
            problem_statement = dataset[i].get("message", "")
            if not problem_statement:
                continue
            
            prompt = format_prompt(problem_statement)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=512, temperature=0.2, 
                    do_sample=True, pad_token_id=tokenizer.eos_token_id, 
                    eos_token_id=tokenizer.eos_token_id
                )
                
            generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            prediction = {
                "instance_id": f"commitpackft_python_{i}",
                "model_patch": generated_text.strip(),
                "dataset": "bigcode/commitpackft"
            }
            f.write(json.dumps(prediction) + "\n")
            f.flush()

print("Loading Base Model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")

run_inference(model, tokenizer, "ood_baseline_preds.jsonl", "Baseline Inference")
base_results = grade_patches("ood_baseline_preds.jsonl")

print("Loading QLoRA Adapter...")
model = PeftModel.from_pretrained(model, adapter_path)

run_inference(model, tokenizer, "ood_thrm_preds.jsonl", "THRM Adapter Inference")
thrm_results = grade_patches("ood_thrm_preds.jsonl")

with open("ood_final_comparison.txt", "w") as f:
    f.write("=== RESEARCH COMPARISON: OOD STRUCTURE ===\n\n")
    f.write("BASELINE (Meta-Llama-3.1-8B-Instruct)\n")
    f.write(f"Valid Syntax: {base_results['valid_syntax']}/{base_results['total']}\n")
    f.write(f"Perfect Structure: {base_results['perfect_structure']}/{base_results['total']}\n\n")
    
    f.write("THRM FINETUNE (QLoRA Adapter)\n")
    f.write(f"Valid Syntax: {thrm_results['valid_syntax']}/{thrm_results['total']}\n")
    f.write(f"Perfect Structure: {thrm_results['perfect_structure']}/{thrm_results['total']}\n")

print("Done. Check ood_final_comparison.txt")