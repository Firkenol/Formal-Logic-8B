import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

class THRMEdgeRouter:
    def __init__(self, base_model="meta-llama/Meta-Llama-3.1-8B", adapter_path="./thrm_qlora_v3"):
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        
        base = AutoModelForCausalLM.from_pretrained(
            base_model, 
            quantization_config=bnb_cfg, 
            device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)

    def compile_dag_context(self, repo_path, target_dir):
        g_pol = open(os.path.join(repo_path, "global.policy")).read() if os.path.exists(os.path.join(repo_path, "global.policy")) else ""
        d_pol = open(os.path.join(repo_path, target_dir, "directory.policy")).read() if os.path.exists(os.path.join(repo_path, target_dir, "directory.policy")) else ""
        return g_pol, d_pol

    def generate_baseline(self, issue_text):
        # THE FIX: Added strict rules to prevent test file modification
        prompt = f"CRITICAL RULE: DO NOT modify any files in tests/ directories or any test_*.py files. Only modify the core source code to fix the bug.\n\nIssue:\n{issue_text}\n\nSolution Patch:\n```diff\n--- a/"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=2048, 
            temperature=0.1, 
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        input_length = inputs['input_ids'].shape[1]
        raw_output = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return "--- a/" + raw_output.split("```")[0]

    def route_task(self, repo_path, target_dir, issue_text):
        g_pol, d_pol = self.compile_dag_context(repo_path, target_dir)
        
        # THE FIX: Added strict rules to prevent test file modification
        prompt = f"Global Policy:\n{g_pol}\n\nDirectory Policy:\n{d_pol}\n\nCRITICAL RULE: DO NOT modify any files in tests/ directories or any test_*.py files. Only modify the core source code to fix the bug.\n\nIssue:\n{issue_text}\n\nSolution Patch:\n```diff\n--- a/"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=2048, 
            temperature=0.2, 
            repetition_penalty=1.15, 
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        input_length = inputs['input_ids'].shape[1]
        raw_output = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return "--- a/" + raw_output.split("```")[0]