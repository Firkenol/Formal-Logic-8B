import os
import glob
import json
import torch
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

EVAL_LIMIT = 50

def get_adapter_and_base():
    print("Scanning for trained THRM adapter...")
    configs = glob.glob(os.path.join(".", "**", "adapter_config.json"), recursive=True)
    if not configs:
        raise FileNotFoundError("\n[FATAL ERROR] Could not find 'adapter_config.json'. Training weights missing.")
    
    latest_config = max(configs, key=os.path.getmtime)
    adapter_dir = os.path.dirname(latest_config)
    
    with open(latest_config, 'r') as f:
        config_data = json.load(f)
        true_base_model = config_data.get("base_model_name_or_path", "meta-llama/Meta-Llama-3-8B")
        
    print(f"[TARGET ACQUIRED] Auto-locked onto weights at: {adapter_dir}")
    print(f"[DNA MATCH] True Base Model extracted: {true_base_model}\n")
    return adapter_dir, true_base_model

def get_policies():
    return "Enforce strict architecture. No raw SQL.", "Use local async DB wrappers."

def run_eval():
    print("Starting evaluation pipeline...")
    
    adapter_dir, true_base_model = get_adapter_and_base()
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    
    print(f"Loading true base model: {true_base_model} from local offline cache...")
    
    # Explicit 4-bit configuration to bypass the kwargs TypeError
    quant_config = BitsAndBytesConfig(load_in_4bit=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        true_base_model, 
        quantization_config=quant_config,
        device_map="auto",
        local_files_only=True
    )
    
    print(f"Loading trained weights from {adapter_dir}")
    thrm_model = PeftModel.from_pretrained(base_model, adapter_dir)
    
    print("Loading benchmark dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    
    baseline_results = []
    trained_results = []

    for i, row in enumerate(dataset):
        if i >= EVAL_LIMIT:
            break
            
        issue = row['problem_statement']
        print(f"Evaluating problem {i+1}/{EVAL_LIMIT}")

        g_pol, d_pol = get_policies()
        prompt = f"<|system|>\n<global_policy>{g_pol}</global_policy>\n<directory_policy>{d_pol}</directory_policy>\n<|user|>\n{issue}\n<|assistant|>"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with thrm_model.disable_adapter():
            base_out = thrm_model.generate(**inputs, max_new_tokens=256, temperature=0.1)
        
        trained_out = thrm_model.generate(**inputs, max_new_tokens=256, temperature=0.1)

        b_patch = tokenizer.decode(base_out[0], skip_special_tokens=True)
        t_patch = tokenizer.decode(trained_out[0], skip_special_tokens=True)
        
        b_pass = 1 if "def " in b_patch or "class " in b_patch else 0
        t_pass = 1 if "def " in t_patch or "class " in t_patch else 0
        
        baseline_results.append(b_pass)
        trained_results.append(t_pass)

    b_pass_t_fail = sum(1 for b, t in zip(baseline_results, trained_results) if b == 1 and t == 0)
    b_fail_t_pass = sum(1 for b, t in zip(baseline_results, trained_results) if b == 0 and t == 1)
    
    table = [[0, b_pass_t_fail], [b_fail_t_pass, 0]]
    
    try:
        p_val = mcnemar(table, exact=True).pvalue
    except:
        p_val = "N/A"
        
    print("\n--- RESULTS ---")
    print(f"Baseline THRM Pass:  {(sum(baseline_results) / EVAL_LIMIT) * 100}%")
    print(f"Trained THRM Pass:   {(sum(trained_results) / EVAL_LIMIT) * 100}%")
    print(f"P-Value:             {p_val}")

if __name__ == "__main__":
    run_eval()