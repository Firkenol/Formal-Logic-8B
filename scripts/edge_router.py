import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class THRMEdgeRouter:
    def __init__(self, base_model="meta-llama/Meta-Llama-3.1-8B", adapter_path="./thrm_qlora_v3"):
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base = AutoModelForCausalLM.from_pretrained(base_model, load_in_4bit=True, device_map="auto")
        self.model = PeftModel.from_pretrained(base, adapter_path)

    def compile_dag_context(self, repo_path, target_dir):
        g_pol = open(os.path.join(repo_path, "global.policy")).read() if os.path.exists(os.path.join(repo_path, "global.policy")) else ""
        d_pol = open(os.path.join(repo_path, target_dir, "directory.policy")).read() if os.path.exists(os.path.join(repo_path, target_dir, "directory.policy")) else ""
        return g_pol, d_pol

    def generate_baseline(self, issue_text):
        prompt = f"<|user|>\n{issue_text}\n<|assistant|>\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        # THE FIX: repetition_penalty=1.15
        outputs = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.2, repetition_penalty=1.15, do_sample=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("<|assistant|>")[-1]

    def route_task(self, repo_path, target_dir, issue_text):
        g_pol, d_pol = self.compile_dag_context(repo_path, target_dir)
        prompt = f"<|system|>\n<global_policy trust='HIGH'>{g_pol}</global_policy>\n<directory_policy trust='MEDIUM'>{d_pol}</directory_policy>\n<|user|>\n{issue_text}\n<|assistant|>"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        # THE FIX: repetition_penalty=1.15
        outputs = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.2, repetition_penalty=1.15, do_sample=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("<|assistant|>")[-1]