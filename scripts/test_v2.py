import json, re, torch, sys
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

base_id = "meta-llama/Meta-Llama-3.1-8B"
adapter_path = "./thrm_qlora_v2"
out_json = "ood_predictions_v2.jsonl"

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
base_model = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb_cfg, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base_id)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

dataset = load_dataset("parquet", data_files="https://huggingface.co/datasets/bigcode/commitpackft/resolve/refs%2Fconvert%2Fparquet/python/train/0000.parquet", split="train[:100]")

total, perfect = 0, 0
pattern = r'@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@'

with open(out_json, "w") as f:
    for i in tqdm(range(len(dataset))):
        msg = dataset[i].get("message", "")
        if not msg: continue
        
        prompt = f"<|begin_of_text|>Resolve this issue with a git patch.\n\nIssue:\n{msg}\n\nPatch:\n```diff\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.2, do_sample=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            
        gen = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        p = gen.split("```diff")[1].split("```")[0].strip() if "```diff" in gen else gen.split("```")[1].strip() if "```" in gen else gen
            
        f.write(json.dumps({"id": f"v2_{i}", "patch": p}) + "\n")
        f.flush()
        
        total += 1
        if "--- a/" in p and "+++ b/" in p and re.search(pattern, p): perfect += 1

score = (perfect / total) * 100 if total > 0 else 0
print(f"V2 ADAPTER SCORE: {score:.1f}%")

if score < 40.0:
    print("CRITICAL FAILURE: Model is still looping or writing garbage. Aborting pipeline to save credits.")
    sys.exit(1) # This tells the bash script to STOP
print("DIAGNOSTIC PASSED. Model is structurally sound. Proceeding to SWE-Bench.")