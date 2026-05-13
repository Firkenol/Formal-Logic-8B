import os, subprocess, tempfile, sys, datetime
from datasets import load_dataset
from edge_router import THRMEdgeRouter

class SWEBenchOrchestrator:
    def __init__(self, router):
        self.router = router
        # Create a unique folder for this specific run
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs/run_{run_id}"
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"Logging this session to: {self.log_dir}")

    def setup_repo(self, repo, commit, td):
        url = f"https://github.com/{repo}.git"
        subprocess.run(["git", "clone", url, td], check=True, capture_output=True)
        subprocess.run(["git", "checkout", commit], cwd=td, check=True, capture_output=True)

    def force_overwrite(self, ai_output, td, instance_id, mode):
        # Save the raw output immediately for your review
        raw_log_path = os.path.join(self.log_dir, f"raw_{instance_id}_{mode}.txt")
        with open(raw_log_path, "w") as f:
            f.write(ai_output)

        try:
            # Look for our new markers
            if "[TARGET_FILE]" not in ai_output or "[NEW_CODE_START]" not in ai_output:
                print(f"\n  !! FORMAT ERROR in {mode} !!")
                print(f"  AI said: {ai_output[:300]}...") # Print first 300 chars so you can see the mess
                return False
            
            file_path = ai_output.split("[TARGET_FILE]")[1].split("[NEW_CODE_START]")[0].strip()
            new_content = ai_output.split("[NEW_CODE_START]")[1].split("```")[0].strip()
            
            full_path = os.path.join(td, file_path)
            with open(full_path, "w") as f:
                f.write(new_content)
            return True
        except Exception as e:
            print(f"  -> Parse Error: {e}")
            return False

    def verify(self, td):
        try: 
            res = subprocess.run([sys.executable, "-m", "pytest"], cwd=td, capture_output=True, timeout=120)
            return res.returncode == 0
        except: return False

    def run_eval(self, limit=3):
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        b_count, t_count = 0, 0

        for i, row in enumerate(dataset):
            if i >= limit: break
            print(f"\n[{i+1}/{limit}] Testing {row['repo']}...")
            
            with tempfile.TemporaryDirectory() as b_dir, tempfile.TemporaryDirectory() as t_dir:
                self.setup_repo(row['repo'], row['base_commit'], b_dir)
                self.setup_repo(row['repo'], row['base_commit'], t_dir)

                b_out = self.router.generate_baseline(row['problem_statement'])
                if self.force_overwrite(b_out, b_dir, i, "baseline") and self.verify(b_dir):
                    b_count += 1
                    print("  [+] Baseline PASSED")

                t_out = self.router.route_task(t_dir, ".", row['problem_statement'])
                if self.force_overwrite(t_out, t_dir, i, "thrm") and self.verify(t_dir):
                    t_count += 1
                    print("  [+] THRM PASSED")

        print(f"\n--- FINAL: Baseline {b_count}/{limit} | THRM {t_count}/{limit} ---")

if __name__ == "__main__":
    router = THRMEdgeRouter(adapter_path="./thrm_qlora_v3")
    SWEBenchOrchestrator(router).run_eval(limit=3)