import os, subprocess, tempfile, sys, datetime, re
from datasets import load_dataset
from edge_router import THRMEdgeRouter

class SWEBenchOrchestrator:
    def __init__(self, router):
        self.router = router
        run_id = datetime.datetime.now().strftime("%H%M%S")
        self.log_dir = f"logs/run_{run_id}"
        os.makedirs(self.log_dir, exist_ok=True)

    def setup_repo(self, repo, commit, td):
        url = f"https://github.com/{repo}.git"
        subprocess.run(["git", "clone", url, td], check=True, capture_output=True)
        subprocess.run(["git", "checkout", commit], cwd=td, check=True, capture_output=True)

    def extract_and_apply(self, ai_text, td, mode):
        # 1. Scraping: Find the diff even if the AI surrounded it with chatter
        patch = ai_text.split("```diff")[-1].split("```")[0].strip()
        if not patch.startswith("--- a/"):
            # Fallback if AI forgot the markdown tags
            match = re.search(r"--- a/.*", ai_text, re.DOTALL)
            patch = match.group(0) if match else ai_text

        # 2. Sanitization: Ensure a trailing newline (Git's biggest pet peeve)
        patch = patch.strip() + "\n"

        patch_path = os.path.join(td, f"fix_{mode}.patch")
        with open(patch_path, "w") as f:
            f.write(patch)

        # 3. Execution: Try to apply it
        res = subprocess.run(["git", "apply", "--whitespace=fix", f"fix_{mode}.patch"], cwd=td, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  [!] {mode.upper()} Patch Failed: {res.stderr.strip().splitlines()[0] if res.stderr else 'Malformed Diff'}")
            return False
        return True

    def verify(self, td):
        try:
            # Short timeout to keep the "quick turnaround" promise
            res = subprocess.run([sys.executable, "-m", "pytest"], cwd=td, capture_output=True, timeout=60)
            return res.returncode == 0
        except: return False

    def run_eval(self, limit=3):
        print(f"Starting session. Logs at: {self.log_dir}")
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        results = {"baseline": 0, "thrm": 0}

        for i, row in enumerate(dataset):
            if i >= limit: break
            print(f"\n[{i+1}/{limit}] {row['repo']} @ {row['base_commit'][:7]}")
            
            with tempfile.TemporaryDirectory() as b_dir, tempfile.TemporaryDirectory() as t_dir:
                self.setup_repo(row['repo'], row['base_commit'], b_dir)
                self.setup_repo(row['repo'], row['base_commit'], t_dir)

                # Test Baseline
                b_out = self.router.generate_baseline(row['problem_statement'])
                if self.extract_and_apply(b_out, b_dir, "baseline") and self.verify(b_dir):
                    results["baseline"] += 1
                    print("  [✓] Baseline Solved")

                # Test THRM (Your Trained Model)
                t_out = self.router.route_task(t_dir, ".", row['problem_statement'])
                if self.extract_and_apply(t_out, t_dir, "thrm") and self.verify(t_dir):
                    results["thrm"] += 1
                    print("  [✓] THRM Solved")

        print(f"\n--- SCORECARD: Baseline {results['baseline']} | THRM {results['thrm']} ---")

if __name__ == "__main__":
    # Ensure this points to your specific Llama 4 Maverick/Scout adapter
    router = THRMEdgeRouter(adapter_path="./thrm_qlora_v3")
    SWEBenchOrchestrator(router).run_eval(limit=3)