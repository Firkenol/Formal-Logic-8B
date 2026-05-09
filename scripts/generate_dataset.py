import json
import logging
import time
from datasets import load_dataset
from openai import OpenAI

logging.basicConfig(filename='generation_progress.log', level=logging.INFO, format='%(asctime)s - %(message)s')
client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-dummy")

def generate_thrm_policies(repo_name, issue_text, patch):
    # Truncate inputs to fit 16k context window safely
    issue_snip = issue_text[:12000]
    patch_snip = patch[:18000]
    
    prompt = f"Analyze {repo_name}.\nIssue: {issue_snip}\nPatch: {patch_snip}\nOutput 'global_policy ||| directory_policy'"
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800
        )
        content = response.choices[0].message.content
        if "|||" in content:
            g, d = content.split("|||")
            return g.strip(), d.strip()
        return "Standard Policy", "Local Logic"
    except Exception as e:
        logging.error(f"Error on {repo_name}: {e}")
        return "Standard Policy", "Local Logic"

def run_all():
    out_file = "thrm_cpt_training_data.jsonl"
    open(out_file, 'w').close() 
    
    targets = [
        ("princeton-nlp/SWE-bench_Verified", "test"),
        ("awslabs/fea-bench", "test"),
        ("race-bench/race-bench", "test")
    ]

    for path, split in targets:
        logging.info(f"Starting dataset: {path}")
        try:
            ds = load_dataset(path, split=split)
            with open(out_file, 'a') as f:
                for i, item in enumerate(ds):
                    repo = item.get('repo','')
                    issue = item.get('problem_statement', item.get('instruction', ''))
                    patch = item.get('patch', item.get('solution', ''))
                    
                    g, d = generate_thrm_policies(repo, issue, patch)
                    
                    # Ensure no extra spaces inside the tags per your previous formatting needs
                    entry = {"text": f"<|system|>\n<global_policy>{g}</global_policy>\n<directory_policy>{d}</directory_policy>\n<|user|>\n{issue}\n<|assistant|>\n{patch}"}
                    f.write(json.dumps(entry) + '\n')
                    
                    if i % 5 == 0:
                        logging.info(f"{path} progress: {i}/{len(ds)}")
        except Exception as e:
            logging.error(f"Dataset {path} failed: {e}")

if __name__ == "__main__":
    time.sleep(10) # Buffer for vLLM routing
    run_all()