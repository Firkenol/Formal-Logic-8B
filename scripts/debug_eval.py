import os, subprocess, tempfile
from datasets import load_dataset
from edge_router import THRMEdgeRouter

print("Loading model...")
router = THRMEdgeRouter(adapter_path="./thrm_qlora_v3")
dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
row = dataset[0]

print(f"\n--- TESTING INSTANCE: {row['repo']} ---")
baseline_patch = router.generate_baseline(row['problem_statement'])

print("\n--- GENERATED PATCH (First 300 chars) ---")
print(baseline_patch[:300] + "\n...[truncated]")

with tempfile.TemporaryDirectory() as td:
    print("\nCloning repo...")
    subprocess.run(["git", "clone", f"https://github.com/{row['repo']}.git", td], capture_output=True)
    subprocess.run(["git", "checkout", row['base_commit']], cwd=td, capture_output=True)

    with open(os.path.join(td, "patch.diff"), "w") as f:
        f.write(baseline_patch)

    print("\nApplying patch...")
    apply_res = subprocess.run(["git", "apply", "patch.diff"], cwd=td, capture_output=True, text=True)
    print("Success!" if apply_res.returncode == 0 else f"GIT ERROR:\n{apply_res.stderr}")

    print("\nRunning Pytest...")
    test_res = subprocess.run(["pytest"], cwd=td, capture_output=True, text=True)
    print("Success!" if test_res.returncode == 0 else f"PYTEST ERROR:\n{test_res.stderr[:500]}")