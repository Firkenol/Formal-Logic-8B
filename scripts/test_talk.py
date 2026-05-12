import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print("Loading model...")
adapter_path = "./thrm_qlora_v3"
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

base = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B", quantization_config=bnb_cfg, device_map="auto")
model = PeftModel.from_pretrained(base, adapter_path)

print("Talking to AI...")
prompt = "Write a simple python python script to print hello world:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Notice we removed the strict EOS token limit to see what it naturally does
outputs = model.generate(**inputs, max_new_tokens=50)
print("\n--- AI RESPONSE ---")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))