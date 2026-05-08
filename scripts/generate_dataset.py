import json
import logging
from datasets import load_dataset
from openai import OpenAI

# log progress to file instead of printing to terminal
logging.basicConfig(
    filename='generation_progress.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-dummy"
)

def generate_thrm_policies(repo_name, issue_text, patch):
    prompt = f"""Analyze repository {repo_name}, issue, and patch.
    Output EXACTLY two strings separated by '|||':
    1. global.policy (security, strict formatting invariants)
    2. directory.policy (local API expectations for modified files)
    
    Issue: {issue_text}
    Patch: {patch}"""
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw_output = response.choices[0].message.content
        g_pol, d_pol = raw_output.split("|||")
        return g_pol.strip(), d_pol.strip()
    except Exception as e:
        logging.error(f"Failed policy generation for {repo_name}: {e}")
        return "Enforce strict architecture.", "Adhere to local constraints."

def process_dataset(dataset_path, split_name, output_file):
    logging.info(f"Loading {dataset_path}...")
    
    try:
        dataset = load_dataset(dataset_path, split=split_name)
    except Exception as e:
        logging.error(f"Could not load {dataset_path}. Ensure the path is correct. Error: {e}")
        return

    with open(output_file, 'a') as f:
        for i, item in enumerate(dataset):
            if i % 10 == 0:
                logging.info(f"{dataset_path} progress: {i}/{len(dataset)}")
            
            repo = item.get('repo', 'unknown_repo')
            issue = item.get('problem_statement', item.get('instruction', ''))
            patch = item.get('patch', item.get('solution', ''))
            
            g_pol, d_pol = generate_thrm_policies(repo, issue, patch)
            
            thrm_prompt = f"""<|system|>
<global_policy trust="HIGH">{g_pol}</global_policy>
<directory_policy trust="MEDIUM">{d_pol}</directory_policy>
<|user|>
{issue}
<|assistant|>
{patch}"""
            
            f.write(json.dumps({"text": thrm_prompt}) + '\n')

if __name__ == "__main__":
    out_file = "thrm_cpt_training_data.jsonl"
    
    # clear existing file before starting the batch
    open(out_file, 'w').close()
    
    # update paths if fea/race are saved locally instead of huggingface hubs
    datasets_to_run = [
        ("princeton-nlp/SWE-bench_Verified", "test"),
        ("awslabs/fea-bench", "test"), 
        ("race-bench/race-bench", "test")
    ]
    
    for ds_path, split in datasets_to_run:
        process_dataset(ds_path, split, out_file)
        
    logging.info("All datasets complete.")