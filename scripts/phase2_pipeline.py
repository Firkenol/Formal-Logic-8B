import os
import subprocess
import json
import re
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# 1. Trigger Docker Evaluation Harness IMMEDIATELY
print("Running SWE-bench Docker evaluation...")
swe_bench_dir = os.path.expanduser("~/Formal-Logic-8B/SWE-bench")
subprocess.run([
    "python", "-m", "swebench.harness.run_evaluation",
    "--predictions_path", "../predictions.jsonl",
    "--run_id", "thrm_3b_qlora_eval",
    "--max_workers", "4",
    "--dataset_name", "princeton-nlp/SWE-bench_Lite",
    "--split", "test",
    "--log_dir", "../eval_logs"
], cwd=swe_bench_dir)

# 2. Load Model for OOD Generation
print("Starting OOD generation...")
base_model_id = "meta-llama/Meta-Llama-3.1-8B"
adapter_path = "./thrm_3b_qlora_final"
output_file = "ood_predictions.jsonl"
datasets_to_test = ["microsoft/FEA-Bench"]

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# 3. Generate OOD Patches
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
                        **inputs, max_new_tokens=1024, temperature=0.2, 
                        do_sample=True, pad_token_id=tokenizer.pad_token_id, 
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
        except Exception:
            pass

# 4. Grade OOD Structure
print("Grading OOD structure...")
results = {"total": 0, "valid_syntax": 0, "valid_targets": 0, "perfect_structure": 0}

with open(output_file, "r") as f:
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

with open("ood_results.txt", "w") as f:
    if results["total"] > 0:
        f.write("-" * 30 + "\n")
        f.write(f"Total Eval: {results['total']}\n")
        f.write(f"Valid Syntax: {results['valid_syntax']} ({(results['valid_syntax']/results['total'])*100:.1f}%)\n")
        f.write(f"Valid Targets: {results['valid_targets']} ({(results['valid_targets']/results['total'])*100:.1f}%)\n")
        f.write(f"Perfect Structure: {results['perfect_structure']} ({(results['perfect_structure']/results['total'])*100:.1f}%)\n")
        f.write("-" * 30 + "\n")

print("Done. Results saved to ood_results.txt")
