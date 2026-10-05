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
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy decoding (deterministic, best for live demos)")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Prompt: must match the training format in generate_dataset.py exactly:
#   <|system|>\n<global_policy>..</global_policy>\n<directory_policy>..</directory_policy>
#   \n<|user|>\n{issue}\n<|assistant|>\n{patch}
# Training rows had no file contents, so the source is appended to the issue
# text, and the assistant turn is pre-filled with the diff header so the model
# continues straight into hunks.
# ----------------------------------------------------------------------------

def build_prompt(issue, repo, files, global_policy, directory_policy):
    context = []
    for rel in files:
        with open(os.path.join(repo, rel), encoding="utf-8") as fh:
            context.append(f"Contents of {rel}:\n{fh.read()}")
    user = issue.strip() + "\n\n" + "\n\n".join(context)

    target = files[0]
    prefill = f"diff --git a/{target} b/{target}\n--- a/{target}\n+++ b/{target}\n@@"
    prompt = (
        "<|system|>\n"
        f"<global_policy>{global_policy}</global_policy>\n"
        f"<directory_policy>{directory_policy}</directory_policy>\n"
        "<|user|>\n"
        f"{user}\n"
        "<|assistant|>\n"
        f"{prefill}"
    )
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
    return "\n".join(keep) + "\n"


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

    banner("STEP 3  Generating patch")
    prompt, prefill = build_prompt(issue, args.repo, args.files,
                                   args.global_policy, args.directory_policy)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stop_strings=STOP_STRINGS,
        tokenizer=tokenizer,
        repetition_penalty=1.05,
    )
    if args.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - t0
    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True, clean_up_tokenization_spaces=False)
    patch = clean_patch(prefill + raw)
    print(patch)
    print(f"[{n_new} tokens in {elapsed:.1f}s]")

    with open(os.path.join(work, "fix.patch"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(patch)

    banner("STEP 4  Applying patch with git")
    res = run(["git", "apply", "--recount", "--ignore-whitespace", "--verbose", "fix.patch"], work)
    if res.returncode != 0:
        print("FAIL  patch did not apply\n" + res.stderr.strip())
        print(f"\nWorking copy kept at: {work}")
        sys.exit(1)
    print("OK    " + (res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "applied"))

    banner("STEP 5  Test suite AFTER patch")
    ok_after, summary = run_tests(work)
    print(("PASS  " if ok_after else "FAIL  ") + summary)

    banner("RESULT")
    print(f"before: {'pass' if ok_before else 'fail'}   after: {'pass' if ok_after else 'fail'}")
    print("RESOLVED" if ok_after and not ok_before else "NOT RESOLVED")
    print(f"Working copy kept at: {work}")


if __name__ == "__main__":
    main()
