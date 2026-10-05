#!/usr/bin/env python3
# scripts/thrm_demo.py
#
# End-to-end demo: issue + real source file -> THRM adapter -> unified diff
# -> git apply -> pytest, with the test suite run before and after.
#
# Usage (from the repo root):
#   python scripts/thrm_demo.py
#   python scripts/thrm_demo.py --repo demo_repo --issue demo_repo/issue.md \
#       --files src/utils/data_loader.py --adapter ./thrm_3b_qlora_final

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thrm_format

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

STOP_STRINGS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end_of_text|>", "```"]


def parse_args():
    p = argparse.ArgumentParser(description="THRM patch-generation demo")
    p.add_argument("--repo", default="demo_repo")
    p.add_argument("--issue", default="demo_repo/issue.md")
    p.add_argument("--files", nargs="+", default=["src/utils/data_loader.py"],
                   help="repo-relative files to show the model")
    p.add_argument("--adapter", default="./thrm_3b_qlora_final")
    p.add_argument("--base-model", default=None,
                   help="defaults to the base model recorded in the adapter config")
    p.add_argument("--global-policy", default="Standard Policy")
    p.add_argument("--directory-policy", default="Local Logic")
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--attempts", type=int, default=5,
                   help="candidate patches to try; attempt 1 is greedy, the rest are sampled")
    p.add_argument("--temperature", type=float, default=0.6,
                   help="sampling temperature for attempts 2+")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Prompt format lives in thrm_format.py, shared with the v5 training data.
# The assistant turn is pre-filled with the diff header so the model
# continues straight into hunks.
# ----------------------------------------------------------------------------

def build_prompt(issue, repo, files, global_policy, directory_policy):
    sources = []
    for rel in files:
        with open(os.path.join(repo, rel), encoding="utf-8") as fh:
            sources.append((rel, fh.read()))
    target = files[0]
    prefill = f"diff --git a/{target} b/{target}\n--- a/{target}\n+++ b/{target}\n@@"
    prompt = thrm_format.build_prompt(issue, sources, global_policy, directory_policy) + prefill
    return prompt, prefill


def clean_patch(text):
    for s in STOP_STRINGS:
        text = text.split(s)[0]
    lines = text.rstrip().splitlines()
    # keep only diff-shaped lines; drop trailing prose the model may append
    keep = []
    for line in lines:
        if line.startswith(("diff --git", "--- ", "+++ ", "@@", "+", "-", " ", "\\")) or line == "":
            keep.append(line)
        else:
            break
    while keep and keep[-1] == "":
        keep.pop()
    return recount_hunks(keep)


def recount_hunks(lines):
    # The model often gets the @@ line counts wrong; rebuild them from the
    # hunk body so both git apply and patch accept it.
    out, i = [], 0
    while i < len(lines):
        m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)", lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        body = []
        i += 1
        while i < len(lines) and not lines[i].startswith(("@@", "diff --git", "--- ")):
            body.append(lines[i] if lines[i] else " ")  # bare blank = blank context line
            i += 1
        old = sum(1 for l in body if l[0] in " -")
        new = sum(1 for l in body if l[0] in " +")
        out.append(f"@@ -{m.group(1)},{old} +{m.group(2)},{new} @@{m.group(3)}")
        out.extend(body)
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------

def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def run_tests(cwd):
    res = run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"], cwd)
    summary = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else res.stderr.strip()
    return res.returncode == 0, summary


def banner(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def main():
    args = parse_args()

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()

    with open(args.issue, encoding="utf-8") as fh:
        issue = fh.read()

    # Work on a throwaway git copy so the demo repo stays pristine.
    work = tempfile.mkdtemp(prefix="thrm_demo_")
    shutil.copytree(args.repo, work, dirs_exist_ok=True)
    run(["git", "init", "-q"], work)
    run(["git", "add", "-A"], work)
    run(["git", "-c", "user.name=demo", "-c", "user.email=demo@local", "commit", "-qm", "base"], work)

    banner("ISSUE")
    print(issue.strip())

    banner("STEP 1  Test suite BEFORE patch")
    ok_before, summary = run_tests(work)
    print(("PASS  " if ok_before else "FAIL  ") + summary)

    banner("STEP 2  Loading model")
    base_id = args.base_model or PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    print(f"base model : {base_id}")
    print(f"adapter    : {args.adapter}")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prompt, prefill = build_prompt(issue, args.repo, args.files,
                                   args.global_policy, args.directory_policy)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    base_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stop_strings=STOP_STRINGS,
        tokenizer=tokenizer,
        repetition_penalty=1.05,
    )

    solved_on = None
    ok_after = False
    for attempt in range(1, args.attempts + 1):
        banner(f"ATTEMPT {attempt}/{args.attempts}  "
               + ("greedy" if attempt == 1 else f"sampled, T={args.temperature}"))

        # reset the working copy to the original buggy state
        run(["git", "checkout", "-q", "--", "."], work)
        run(["git", "clean", "-qfd"], work)

        gen_kwargs = dict(base_kwargs)
        if attempt == 1:
            gen_kwargs.update(do_sample=False)
        else:
            gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=0.95)

        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - t0
        raw = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
        patch = clean_patch(prefill + raw)
        print(patch)
        print(f"[{out.shape[1] - prompt_len} tokens in {elapsed:.1f}s]")

        with open(os.path.join(work, "fix.patch"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(patch)

        # strict apply first, then a fuzzy fallback that tolerates small
        # context mismatches (e.g. one blank line instead of two)
        res = run(["git", "apply", "--recount", "--ignore-whitespace", "fix.patch"], work)
        method = "git apply"
        if res.returncode != 0:
            res = run(["patch", "-p1", "--fuzz=3", "--no-backup-if-mismatch", "-i", "fix.patch"], work)
            method = "patch --fuzz=3"
        os.remove(os.path.join(work, "fix.patch"))
        if res.returncode != 0:
            err = (res.stderr or res.stdout).strip().splitlines()
            print("APPLY FAIL  " + (err[-1] if err else "patch did not apply"))
            continue
        print(f"APPLY OK    via {method}")

        ok_after, summary = run_tests(work)
        print(("TESTS PASS  " if ok_after else "TESTS FAIL  ") + summary)
        if ok_after:
            solved_on = attempt
            break

    banner("RESULT")
    print(f"tests before: {'pass' if ok_before else 'fail'}")
    if solved_on and not ok_before:
        print(f"RESOLVED on attempt {solved_on} of {args.attempts}")
        print(run(["git", "diff"], work).stdout)
    else:
        print(f"NOT RESOLVED in {args.attempts} attempts")
    print(f"Working copy kept at: {work}")


if __name__ == "__main__":
    main()
