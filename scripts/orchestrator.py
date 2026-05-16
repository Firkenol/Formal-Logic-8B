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
            # find diff block
            patch = ai_text.split("```diff")[-1].split("```")[0].strip()
            if not patch.startswith("--- a/"):
                # fallback regex
                match = re.search(r"--- a/.*", ai_text, re.DOTALL)
                patch = match.group(0) if match else ai_text

            # ensure trailing newline
            patch = patch.strip() + "\n"

            patch_path = os.path.join(td, f"fix_{mode}.patch")
            with open(patch_path, "w") as f:
                f.write(patch)

            # -f prevents patch from pausing to ask for stdin if the diff is fucked
            res = subprocess.run(
                ["patch", "-p1", "--fuzz=3", "-f", "-i", f"fix_{mode}.patch"], 
                cwd=td, 
                capture_output=True, 
                text=True
            )
            
            if res.returncode != 0:
                err_msg = res.stderr.strip().splitlines()[0] if res.stderr else "Malformed Diff / Patch Failed"
                print(f"  [!] {mode.upper()} Patch Failed: {err_msg}")
                return False
                
            return True

    def verify(self, td):
        try:
            # Short timeout to keep the "quick turnaround" promise
            res = subprocess.run([sys.executable, "-m", "pytest"], cwd=td, capture_output=True, timeout=60)
            return res.returncode == 0
        except: 
            return False

    def run_eval(self, limit=1):
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

                # Test THRM
                t_out = self.router.route_task(t_dir, ".", row['problem_statement'])
                if self.extract_and_apply(t_out, t_dir, "thrm") and self.verify(t_dir):
                    results["thrm"] += 1
                    print("  [✓] THRM Solved")

        print(f"\n--- SCORECARD: Baseline {results['baseline']} | THRM {results['thrm']} ---")

if __name__ == "__main__":
    import sys
    print("\n[SYSTEM] Script triggered. Executing...", flush=True)
    
    print("[SYSTEM] Initializing THRMEdgeRouter. Loading weights to GPU...", flush=True)
    # If it hangs, it will hang right after printing the line above.
    router = THRMEdgeRouter(adapter_path="./thrm_qlora_v3")
    
    print("[SYSTEM] Model loaded successfully. Starting SWEBenchOrchestrator...", flush=True)
    SWEBenchOrchestrator(router).run_eval(limit=3)  