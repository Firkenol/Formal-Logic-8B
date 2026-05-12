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
        # THE FIX: Hardcoding Llama 3.1's exact prompt format to bypass the missing template error
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nPlease provide a git patch (diff) to solve this issue:\n{issue_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
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
        return self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    def route_task(self, repo_path, target_dir, issue_text):
        g_pol, d_pol = self.compile_dag_context(repo_path, target_dir)
        
        # THE FIX: Hardcoding Llama 3.1's exact system and user prompt format
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a SWE. Output only a valid git patch. <global_policy>{g_pol}</global_policy>\n<directory_policy>{d_pol}</directory_policy><|eot_id|><|start_header_id|>user<|end_header_id|>\n\nProvide a git patch for this issue:\n{issue_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
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
        return self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)