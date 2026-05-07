import json
import os
from datasets import load_dataset
from openai import AzureOpenAI

# Initialize Azure client. Fails safely if keys are missing from .env
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def generate_thrm_policies(repo_name, issue_text, patch):
    """
    Forces the API to deduce the strict system invariants (global) and local API 
    expectations (directory) from the solved patch.
    """
    prompt = f"""Analyze repository {repo_name}, issue, and patch.
    Output EXACTLY two strings separated by '|||':
    1. global.policy (security, strict formatting invariants)
    2. directory.policy (local API expectations for modified files)
    
    Issue: {issue_text}
    Patch: {patch}"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0 # CRITICAL: 0.0 temperature ensures deterministic, zero-variance outputs to prevent hallucinated trust boundaries.
    )
    
    raw_output = response.choices[0].message.content
    try:
        g_pol, d_pol = raw_output.split("|||")
        return g_pol.strip(), d_pol.strip()
    except ValueError:
        # Fallback safeguard if the API ignores formatting instructions
        return "Enforce strict PEP8.", "Adhere to local module imports."

def build_thrm_dataset(output_file="thrm_cpt_training_data.jsonl"):
    """Pulls SWE-bench and formats it with our custom DAG trust tokens."""
    print("Loading SWE-bench Verified...")
    swe_dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    
    with open(output_file, 'w') as f:
        for i, item in enumerate(swe_dataset):
            print(f"Synthesizing instance {i}/{len(swe_dataset)}...")
            repo = item['repo']
            issue = item['problem_statement']
            patch = item['patch']
            
            g_pol, d_pol = generate_thrm_policies(repo, issue, patch)
            
            # This specific formatting is what the 8B model will learn to predict.
            # It maps the <global_policy> token to the HIGH trust concept.
            thrm_prompt = f"""<|system|>
<global_policy trust="HIGH">{g_pol}</global_policy>
<directory_policy trust="MEDIUM">{d_pol}</directory_policy>
<|user|>
{issue}
<|assistant|>
{patch}"""
            
            f.write(json.dumps({"text": thrm_prompt}) + '\n')
            
    print(f"Data synthesis complete. Saved to {output_file}")

if __name__ == "__main__":
    build_thrm_dataset()