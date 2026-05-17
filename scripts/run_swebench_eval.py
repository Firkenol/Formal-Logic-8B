import os
import json
import subprocess
import time
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
# SLICED TO 10 FOR SMOKE TEST
dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test[:10]")

def format_prompt(instance):
    problem_statement = instance["problem_statement"]
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a senior software engineer. Output a unified git diff to fix the issue. "
        "Use the standard --- a/file and +++ b/file format.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{problem_statement}<|eot_id|>"
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
    # Write the CSV header for your research data
    metrics_file.write("instance_id,input_tokens,output_tokens,generation_time_sec\n")
    
    for i in tqdm(range(len(dataset)), desc="SWE-bench Inference"):
        instance = dataset[i]
        prompt = format_prompt(instance)
        
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
        
        # Log the efficiency data for your paper
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
subprocess.run([
    "python3", "-m", "swebench.harness.run_evaluation",
    "--predictions_path", os.path.abspath(predictions_file),
    "--run_id", "thrm_v4_eval",
    "--max_workers", "4",
    "--dataset_name", "princeton-nlp/SWE-bench_Lite",
    "--split", "test[:10]" # IMPORTANT: Tell the harness to only evaluate the first 10
], cwd=swe_bench_dir)

print("SWE-bench Smoke Test Complete.")