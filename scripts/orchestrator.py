#!/usr/bin/env python3
# scripts/orchestrator.py
#
# All imports that touch CUDA live inside if __name__ == "__main__".
# Any library that re-imports this module in a worker process will find
# nothing to execute and exit cleanly.

import os
import re
import subprocess
import sys
import tempfile
import datetime


class SWEBenchOrchestrator:
    def __init__(self, router):
        self.router = router
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs/run_{run_id}"
        os.makedirs(self.log_dir, exist_ok=True)

    # ------------------------------------------------------------------

    def setup_repo(self, repo: str, commit: str, td: str) -> None:
        url = f"https://github.com/{repo}.git"
        subprocess.run(["git", "clone", url, td], check=True, capture_output=True)
        subprocess.run(["git", "checkout", commit], cwd=td, check=True, capture_output=True)

    # ------------------------------------------------------------------

    def clean_model_output(self, text: str) -> str:
        if "```diff" in text:
            text = text.split("```diff")[-1].split("```")[0]

        match = re.search(r"(--- a/\S+.*)", text, re.DOTALL)
        if not match:
            return ""
        text = match.group(1)

        # Collapse hallucinated repeated path segments, e.g.
        # a/astropy/astropy/astropy/time/time.py -> a/astropy/time/time.py
        def fix_path(line):
            if not (line.startswith("--- a/") or line.startswith("+++ b/")):
                return line
            prefix = line[:6]
            parts = line[6:].split("/")
            deduped = []
            for p in parts:
                if not deduped or p != deduped[-1]:
                    deduped.append(p)
            return prefix + "/".join(deduped)

        lines = [fix_path(l) for l in text.splitlines()]
        return "\n".join(lines).strip() + "\n"

    # ------------------------------------------------------------------

    def extract_and_apply(self, ai_text: str, td: str, mode: str, log_prefix: str) -> bool:
        raw_path = os.path.join(self.log_dir, f"{log_prefix}_{mode}.txt")
        with open(raw_path, "w") as fh:
            fh.write(ai_text)

        patch = self.clean_model_output(ai_text)

        if not patch.startswith("--- a/"):
            print(f"  [!] {mode.upper()} Patch Failed: could not extract a valid diff")
            return False

        patch_filename = f"fix_{mode}.patch"
        with open(os.path.join(td, patch_filename), "w") as fh:
            fh.write(patch)

        res = subprocess.run(
            ["patch", "-p1", "--fuzz=3", "-f", "-i", patch_filename],
            cwd=td,
            capture_output=True,
            text=True,
        )

        if res.returncode != 0:
            first_err = res.stderr.strip().splitlines()[0] if res.stderr else "Malformed diff / patch failed"
            print(f"  [!] {mode.upper()} Patch Failed: {first_err}")
            failed_path = os.path.join(self.log_dir, f"{log_prefix}_{mode}_failed.patch")
            with open(failed_path, "w") as fh:
                fh.write(patch)
            return False

        return True

    # ------------------------------------------------------------------

    def verify(self, td: str) -> bool:
        try:
            res = subprocess.run(
                [sys.executable, "-m", "pytest"],
                cwd=td,
                capture_output=True,
                timeout=60,
            )
            return res.returncode == 0
        except Exception:
            return False

    # ------------------------------------------------------------------

    def run_eval(self, limit: int = 3) -> None:
        from datasets import load_dataset

        print(f"Starting session. Logs at: {self.log_dir}")
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        results = {"baseline": 0, "thrm": 0}

        for i, row in enumerate(dataset):
            if i >= limit:
                break

            print(f"\n[{i + 1}/{limit}] {row['repo']} @ {row['base_commit'][:7]}")
            log_prefix = f"raw_{i}"

            with (
                tempfile.TemporaryDirectory() as b_dir,
                tempfile.TemporaryDirectory() as t_dir,
            ):
                self.setup_repo(row["repo"], row["base_commit"], b_dir)
                self.setup_repo(row["repo"], row["base_commit"], t_dir)

                b_out = self.router.generate_baseline(row["problem_statement"])
                if self.extract_and_apply(b_out, b_dir, "baseline", log_prefix) and self.verify(b_dir):
                    results["baseline"] += 1
                    print("  [+] Baseline solved")

                t_out = self.router.route_task(t_dir, ".", row["problem_statement"])
                if self.extract_and_apply(t_out, t_dir, "thrm", log_prefix) and self.verify(t_dir):
                    results["thrm"] += 1
                    print("  [+] THRM solved")

        print(
            f"\n--- SCORECARD: Baseline {results['baseline']}/{limit}"
            f" | THRM {results['thrm']}/{limit} ---"
        )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from edge_router import THRMEdgeRouter

    print("[SYSTEM] Script triggered. Executing...", flush=True)
    print("[SYSTEM] Initializing THRMEdgeRouter. Loading weights to GPU...", flush=True)

    router = THRMEdgeRouter(adapter_path="./thrm_qlora_v3")

    print("[SYSTEM] Model loaded successfully. Starting SWEBenchOrchestrator...", flush=True)
    SWEBenchOrchestrator(router).run_eval(limit=3)