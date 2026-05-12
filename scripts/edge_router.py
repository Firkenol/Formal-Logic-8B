import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

class THRMEdgeRouter:
    def __init__(self, base_model="meta-llama/Meta-Llama-3.1-8B", adapter_path="./thrm_qlora_v3"):
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_cfg, device_map="auto")
        self.model = PeftModel.from_pretrained(base, adapter_path)

    def compile_dag_context(self, repo_path, target_dir):
        g_pol = open(os.path.join(repo_path, "global.policy")).read() if os.path.exists(os.path.join(repo_path, "global.policy")) else ""
        d_pol = open(os.path.join(repo_path, target_dir, "directory.policy")).read() if os.path.exists(os.path.join(repo_path, target_dir, "directory.policy")) else ""
        return g_pol, d_pol

    def get_prompt(self, issue, context=""):
        # This "One-Shot" example teaches the Base model the exact formatting required.
        example = """Example Issue: Fix division by zero in stats.py
Example Patch:
```diff
--- a/stats.py
+++ b/stats.py
@@ -10,1 +10,4 @@
-    return a / b
+    if b == 0:
+        return 0
+    return a / b
```"""
        return f"{example}\n\n{context}\nIssue:\n{issue}\n\nSolution Patch:\n```diff\n--- a/"

    def generate_baseline(self, issue_text):
        prompt = self.get_prompt(issue_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=2048, temperature=0.1, do_sample=True, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id)
        raw = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return "--- a/" + raw.split("```")[0]

    def route_task(self, repo_path, target_dir, issue_text):
        g_pol, d_pol = self.compile_dag_context(repo_path, target_dir)
        context = f"Global Policy: {g_pol}\nDirectory Policy: {d_pol}"
        prompt = self.get_prompt(issue_text, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=2048, temperature=0.1, do_sample=True, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id)
        raw = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return "--- a/" + raw.split("```")[0]