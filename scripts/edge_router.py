import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

class THRMEdgeRouter:
    def __init__(self, base_model="meta-llama/Meta-Llama-3.1-8B", adapter_path="./thrm_qlora_v3"):
        # THE FIX: Set clean_up_tokenization_spaces=False to stop the tokenizer from corrupting code indentation
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, clean_up_tokenization_spaces=False)
        
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_cfg, device_map="auto")
        self.model = PeftModel.from_pretrained(base, adapter_path)

    def route_task(self, repo_path, target_dir, issue_text):
        # We give it a pattern to follow, not a new language to learn
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a senior software engineer. Output a unified git diff to fix the issue. Use the standard --- a/file and +++ b/file format.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nIssue:\n{issue_text}\n\nSolution:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n```diff\n--- a/"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=2500, 
            temperature=0.1, # Keep it low for structural precision
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Return the raw text—our new Orchestrator will handle the cleaning
        return "--- a/" + self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    def generate_baseline(self, issue_text):
        # Same logic for baseline to keep comparison fair
        return self.route_task("", "", issue_text)