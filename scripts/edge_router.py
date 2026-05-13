import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

class THRMEdgeRouter:
    def __init__(self, base_model="meta-llama/Meta-Llama-3.1-8B", adapter_path="./thrm_qlora_v3"):
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_cfg, device_map="auto")
        self.model = PeftModel.from_pretrained(base, adapter_path)

    def generate_baseline(self, issue_text):
        prompt = f"Issue:\n{issue_text}\n\nTask: Provide the full code for the fix.\n\n[TARGET_FILE]\n"
        return self._run(prompt)

    def route_task(self, repo_path, target_dir, issue_text):
        g_pol = open(os.path.join(repo_path, "global.policy")).read() if os.path.exists(os.path.join(repo_path, "global.policy")) else ""
        prompt = f"Policy: {g_pol}\nIssue: {issue_text}\n\nTask: Provide the full code for the fix.\n\n[TARGET_FILE]\n"
        return self._run(prompt)

    def _run(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=3500, # Llama 4 can handle longer contexts
            temperature=0.1, 
            do_sample=True, 
            pad_token_id=self.tokenizer.eos_token_id
        )
        # We return the whole thing including the markers so the orchestrator can parse it
        return self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)