import os
import time
import json
import torch
import subprocess
import tempfile
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Set to 50 for a fast verification run, change to 500 for the final paper stats
EVAL_LIMIT = 500

def get_adapter_and_base():
    adapter_dir = "./thrm_qlora_v4"
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError("[FATAL] Fine-tuned adapter directory not found at ./thrm_qlora_v4")
    
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path, 'r') as f:
        config_data = json.load(f)
        true_base_model = config_data.get("base_model_name_or_path", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        
    return adapter_dir, true_base_model

def extract_and_check_diff(model_output):
    """
    Extracts the unified diff block from model text and validates structure.
    Handles standard markdown blocks and raw un-wrapped git diff text blocks.
    """
    if "```diff" in model_output:
        diff_block = model_output.split("```diff")[1].split("```")[0]
    elif "```" in model_output:
        diff_block = model_output.split("```")[1].split("```")[0]
    else:
        diff_block = model_output

    # Fallback checking for structural git headers if extraction failed
    if "---" not in diff_block or "+++" not in diff_block:
        return 0, 0
        
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(diff_block)
        temp_path = f.name
        
    try:
        # Run ruff to catch implementation drift and syntax compliance
        result = subprocess.run(['ruff', 'check', temp_path], capture_output=True, text=True)
        drift_errors = len(result.stdout.split('\n')) - 1 if result.stdout else 0
        
        # Valid patch requires standard hunk markers
        is_valid_patch = 1 if "@@" in diff_block else 0
        return is_valid_patch, drift_errors
    except Exception:
        return 0, 0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def get_policies():
    return "Enforce strict architecture. No raw SQL.", "Use local async DB wrappers."

def run_eval():
    adapter_dir, true_base_model = get_adapter_and_base()
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        true_base_model, 
        quantization_config=quant_config,
        device_map="auto"
    )
    
    thrm_model = PeftModel.from_pretrained(base_model, adapter_dir)
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    
    b_results, t_results = [], []
    b_times, t_times = [], []
    b_drifts, t_drifts = [], []

    print(f"[INFO] Initializing evaluation loop for {EVAL_LIMIT} instances.")

    for i, row in enumerate(dataset):
        if i >= EVAL_LIMIT: 
            break
            
        issue = row['problem_statement']
        
        g_pol, d_pol = get_policies()
        
        # Explicit structure matching Llama 3.1 Instruct tokens to fix early EOS firing
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"<global_policy>{g_pol}</global_policy>\n"
            f"<directory_policy>{d_pol}</directory_policy><|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{issue}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000).to("cuda")
        
        # Condition 2: Baseline evaluation (Adapter turned off)
        start_b = time.time()
        with thrm_model.disable_adapter():
            base_out = thrm_model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
        b_time = time.time() - start_b
        
        # Condition 3: Trained THRM evaluation (Adapter active)
        start_t = time.time()
        trained_out = thrm_model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
        t_time = time.time() - start_t

        b_patch = tokenizer.decode(base_out[0], skip_special_tokens=True)
        t_patch = tokenizer.decode(trained_out[0], skip_special_tokens=True)
        
        b_pass, b_drift = extract_and_check_diff(b_patch)
        t_pass, t_drift = extract_and_check_diff(t_patch)
        
        b_results.append(b_pass)
        t_results.append(t_pass)
        b_times.append(b_time)
        t_times.append(t_time)
        b_drifts.append(b_drift)
        t_drifts.append(t_drift)
        
        if (i + 1) % 10 == 0 or (i + 1) == EVAL_LIMIT:
            print(f"Progress: {i+1}/{EVAL_LIMIT} evaluated.")

    # Statistical Evaluation Block
    b_pass_t_fail = sum(1 for b, t in zip(b_results, t_results) if b == 1 and t == 0)
    b_fail_t_pass = sum(1 for b, t in zip(b_results, t_results) if b == 0 and t == 1)
    table = [[0, b_pass_t_fail], [b_fail_t_pass, 0]]
    p_val_mcnemar = mcnemar(table, exact=True).pvalue if (b_pass_t_fail + b_fail_t_pass) > 0 else 1.0

    total_b_time, total_t_time = sum(b_times), sum(t_times)
    
    # Run normality checks to choose parametric vs non-parametric variance tests
    if len(t_drifts) >= 3:
        _, shapiro_p = stats.shapiro(t_drifts)
        if shapiro_p < 0.05:
            _, drift_p_val = stats.wilcoxon(b_drifts, t_drifts)
        else:
            _, drift_p_val = stats.ttest_rel(b_drifts, t_drifts)
    else:
        drift_p_val = float('nan')

    print("\n--- PAPER METRICS RESULTS ---")
    print(f"[H1] Baseline C2 Pass Rate: {(sum(b_results) / EVAL_LIMIT) * 100:.2f}%")
    print(f"[H1] Trained THRM C3 Pass Rate: {(sum(t_results) / EVAL_LIMIT) * 100:.2f}%")
    print(f"[H1] McNemar P-Value: {p_val_mcnemar:.4f}")
    print(f"[H2] Avg C2 Drift Errors: {sum(b_drifts)/EVAL_LIMIT:.2f}")
    print(f"[H2] Avg C3 Drift Errors: {sum(t_drifts)/EVAL_LIMIT:.2f}")
    print(f"[H2] Drift Sig (P-Value): {drift_p_val:.4f}")
    print(f"[H5] Compute Time Ratio (C3/C2): {total_t_time / total_b_time:.2f}x")

if __name__ == "__main__":
    run_eval()