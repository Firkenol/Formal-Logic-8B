import os
import json
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

model_id = "meta-llama/Meta-Llama-3.1-8B"
out_json = "ood_predictions_baseline_pure.jsonl"
out_text = "ood_results_baseline_pure.txt"

print("Loading model...")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_cfg,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

print("Loading data...")
dataset = load_dataset(
    "parquet",
    data_files="https://huggingface.co/datasets/bigcode/commitpackft/resolve/refs%2Fconvert%2Fparquet/python/train/0000.parquet",
    split="train[:100]"
)

print("Generating...")
with open(out_json, "w") as f:
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        msg = item.get("message", "")
        if not msg:
            continue
            
        prompt = f"<|begin_of_text|>Resolve this issue with a git patch.\n\nIssue:\n{msg}\n\nPatch:\n```diff\n"
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
            
        gen = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        final_p = gen
        if "```diff" in gen:
            final_p = gen.split("```diff")[1].split("```")[0].strip()
        elif "```" in gen:
            final_p = gen.split("```")[1].strip()
            
        f.write(json.dumps({"instance_id": f"base_{i}", "model_patch": final_p}) + "\n")
        f.flush()

print("Grading...")
total, perfect = 0, 0
pattern = r'@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@'

with open(out_json, "r") as f:
    for line in f:
        data = json.loads(line)
        p = data.get("model_patch", "")
        if not p:
            continue
        total += 1
        has_ab = "--- a/" in p and "+++ b/" in p
        has_hunk = re.search(pattern, p) is not None
        if has_ab and has_hunk:
            perfect += 1

res = f"Total: {total}\nPerfect Structure: {perfect} ({(perfect/total)*100 if total > 0 else 0:.1f}%)\n"
with open(out_text, "w") as f:
    f.write(res)
print(res)