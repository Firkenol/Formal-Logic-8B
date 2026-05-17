import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter_path = "./thrm_qlora_v4"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Loading QLoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

dummy_issue = """Fix the zero division error in the math module.

Available File Context:
--- a/src/math_utils.py
+++ b/src/math_utils.py
@@ -10,5 +10,5 @@
 def divide(a, b):
-    return a / b
+    # Need to handle b=0
"""

# Must match exactly what the model saw during training
prompt = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "You are a senior software engineer. Output a unified git diff to fix the issue. "
    "Use the standard --- a/file and +++ b/file format.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    f"{dummy_issue}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("Generating patch...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.2, 
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

print("\n--- RAW MODEL OUTPUT ---")
print(generated_text)
print("------------------------")