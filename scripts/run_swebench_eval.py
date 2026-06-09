import os
import json
import subprocess
import time
import requests
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

base_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter_path = "./thrm_qlora_v4"
predictions_file = "swebench_thrm_predictions.jsonl"
swe_bench_dir = os.path.expanduser("~/Formal-Logic-8B/SWE-bench")

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

print("Loading SWE-bench Lite (SMOKE TEST: 10 Issues)...")
dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test[:10]")

def get_oracle_file_context(instance):
    """Extracts file paths from the hidden gold patch and downloads the raw code from GitHub."""
    repo = instance['repo']
    commit = instance['base_commit']
    gold_patch = instance['patch']
    
    # 1. Identify which files need to be edited (without looking at the actual diff logic)
    file_paths = set()
    for line in gold_patch.split('\n'):
        if line.startswith('--- a/'):
            file_paths.add(line[6:])
    
    # 2. Download the historical file contents
    file_context = ""
    for fp in file_paths:
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{fp}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                content = resp.text
                
                # Truncate massive files to prevent A100 VRAM OOM crashes
                lines = content.split('\n')
                if len(lines) > 3000:
                    content = '\n'.join(lines[:3000]) + "\n\n... [TRUNCATED FOR LENGTH] ..."
                    
                file_context += f"--- {fp} ---\n```python\n{content}\n```\n\n"
        except Exception as e:
            file_context += f"--- {fp} ---\n[Error retrieving file from GitHub]\n\n"
            
    return file_context

def format_prompt(instance, file_context):
    problem_statement = instance["problem_statement"]
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a senior software engineer. Output a unified git diff to fix the issue. "
        "Use the standard --- a/file and +++ b/file format.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Issue Description:\n{problem_statement}\n\n"
        f"Available File Context:\n{file_context}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

print("Loading Base Model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")

print("Loading QLoRA Adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

print("Generating Patches and Recording Efficiency Metrics...")
with open(predictions_file, "w") as f, open("thrm_efficiency_metrics.csv", "w") as metrics_file:
    metrics_file.write("instance_id,input_tokens,output_tokens,generation_time_sec\n")
    
    for i in tqdm(range(len(dataset)), desc="SWE-bench Inference"):
        instance = dataset[i]
        
        # Dynamically fetch the code context from GitHub
        file_context = get_oracle_file_context(instance)
        prompt = format_prompt(instance, file_context)
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_token_count = inputs["input_ids"].shape[1]
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=1024, temperature=0.2, 
                do_sample=True, pad_token_id=tokenizer.eos_token_id, 
                eos_token_id=tokenizer.eos_token_id
            )
            
        end_time = time.time()
        generation_time = end_time - start_time
        output_token_count = outputs[0].shape[0] - input_token_count
            
        generated_text = tokenizer.decode(outputs[0][input_token_count:], skip_special_tokens=True)
        
        metrics_file.write(f"{instance['instance_id']},{input_token_count},{output_token_count},{generation_time:.2f}\n")
        metrics_file.flush()
        
        prediction = {
            "instance_id": instance["instance_id"],
            "model_name_or_path": "thrm_qlora_v4",
            "model_patch": generated_text.strip()
        }
        f.write(json.dumps(prediction) + "\n")
        f.flush()

print("Running SWE-bench Docker evaluation...")
# Execute using the virtual environment to prevent dependency conflicts
subprocess.run([
    os.path.expanduser("~/Formal-Logic-8B/thrm_env/bin/python"), "-m", "swebench.harness.run_evaluation",
    "--predictions_path", os.path.abspath(predictions_file),
    "--run_id", "thrm_v4_eval",
    "--max_workers", "4",
    "--dataset_name", "princeton-nlp/SWE-bench_Lite",
    "--split", "test[:10]"
], cwd=swe_bench_dir)

print("SWE-bench Smoke Test Complete.")