import os, subprocess, tempfile, numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from datasets import load_dataset
from edge_router import THRMEdgeRouter

class SWEBenchOrchestrator:
    def __init__(self, router):
        self.router = router
        self.b_res, self.t_res, self.b_tok, self.t_tok = [], [], [], []

    def setup_repo(self, repo, commit, td):
        subprocess.run(["git", "clone", f"https://github.com/{repo}.git", td], check=True, capture_output=True)
        subprocess.run(["git", "checkout", commit], cwd=td, check=True, capture_output=True)

    def apply_patch(self, patch_str, td):
        with open(os.path.join(td, "patch.diff"), "w") as f: f.write(patch_str)
        return subprocess.run(["git", "apply", "patch.diff"], cwd=td, capture_output=True).returncode == 0

    def verify(self, td):
        try: return subprocess.run(["pytest"], cwd=td, capture_output=True, timeout=120).returncode == 0
        except: return False

    def run_eval(self, limit=500):
        print("Initializing SWE-bench Evaluation Protocol...")
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        
        # Ensure output directory exists before 12 hours of compute
        os.makedirs("logs", exist_ok=True)

        for i, row in enumerate(dataset):
            if i >= limit: break
            print(f"Evaluating Instance {i+1}/{limit}: {row['repo']}")
            
            with tempfile.TemporaryDirectory() as b_dir, tempfile.TemporaryDirectory() as t_dir:
                self.setup_repo(row['repo'], row['base_commit'], b_dir)
                self.setup_repo(row['repo'], row['base_commit'], t_dir)

                b_patch = self.router.generate_baseline(row['problem_statement'])
                self.b_tok.append(len(self.router.tokenizer.encode(row['problem_statement'])))
                self.b_res.append(1 if self.apply_patch(b_patch, b_dir) and self.verify(b_dir) else 0)

                t_patch = self.router.route_task(t_dir, ".", row['problem_statement'])
                self.t_tok.append(len(self.router.tokenizer.encode(row['problem_statement'])))
                self.t_res.append(1 if self.apply_patch(t_patch, t_dir) and self.verify(t_dir) else 0)

        p_val = mcnemar([[0, sum(1 for b,t in zip(self.b_res, self.t_res) if b==1 and t==0)], [sum(1 for b,t in zip(self.b_res, self.t_res) if b==0 and t==1), 0]], exact=True).pvalue
        comp = 1 - (np.mean(self.t_tok) / np.mean(self.b_tok))
        
        # Save final metrics safely
        final_stats = f"Baseline Pass: {sum(self.b_res)/limit*100:.2f}%\nTHRM Pass: {sum(self.t_res)/limit*100:.2f}%\nP-Value: {p_val}\nCompression: {comp*100:.2f}%\n"
        print("\n--- FINAL PAPER RESULTS ---")
        print(final_stats)
        
        with open("logs/FINAL_PAPER_RESULTS.txt", "w") as f:
            f.write(final_stats)

if __name__ == "__main__":
    router = THRMEdgeRouter(adapter_path="./thrm_qlora_v3")
    SWEBenchOrchestrator(router).run_eval(limit=500)