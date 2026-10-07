#!/usr/bin/env python3
# scripts/eval_v6.py
#
# Clean evaluation: base model vs. base + THRM adapter on SWE-bench Lite.
# None of the Lite repositories appear in the v5/v6 training data
# (build_training_data_v5.py excludes them), and both conditions get the
# identical prompt and are scored by the official SWE-bench Docker harness.
#
# Setting: oracle localization. The prompt shows the files the reference fix
# edits, windowed around the edited regions (same as training). This is easier
# than the public leaderboard setting, so the absolute numbers are not
# comparable to it; the base-vs-adapter difference is the result.
#
#   python scripts/eval_v6.py prepare  --base-model Qwen/Qwen2.5-Coder-7B --n 100
#   CUDA_VISIBLE_DEVICES=0 python scripts/eval_v6.py generate --adapter ./thrm_v6 --shard 0 --num-shards 4
#   python scripts/eval_v6.py score

import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_training_data_v5 import fetch_file, split_patch, window_file
from thrm_demo import STOP_STRINGS, clean_patch
from thrm_format import build_prompt

DATASET = "princeton-nlp/SWE-bench_Lite"
CONDITIONS = ("base", "thrm")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "generate", "score"])
    p.add_argument("--out-dir", default="eval_v6")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-7B")
    p.add_argument("--adapter", default="./thrm_v6")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max-prompt-tokens", type=int, default=3072)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--full-file-lines", type=int, default=250)
    p.add_argument("--cache-dir", default=".cache_v5_files")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--skip-harness", action="store_true",
                   help="score: rebuild the table from existing harness logs only")
    return p.parse_args()


# ----------------------------------------------------------------------------

def prepare(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    rows = sorted(load_dataset(DATASET, split="test"), key=lambda r: r["instance_id"])
    random.Random(0).shuffle(rows)

    picked, skipped = [], 0
    for row in rows:
        hunks_by_file = split_patch(row["patch"])
        if not hunks_by_file or len(hunks_by_file) > 2:
            skipped += 1
            continue
        rng = random.Random(zlib.crc32(row["instance_id"].encode()))
        files = []
        for path, hunks in hunks_by_file.items():
            src = fetch_file(row["repo"], row["base_commit"], path, args.cache_dir)
            if src is None:
                break
            files.append((path, window_file(src.splitlines(), hunks, rng, args.full_file_lines)))
        else:
            target = files[0][0]
            prefill = f"diff --git a/{target} b/{target}\n--- a/{target}\n+++ b/{target}\n@@"
            prompt = build_prompt(row["problem_statement"], files) + prefill
            if len(tokenizer(prompt)["input_ids"]) <= args.max_prompt_tokens:
                picked.append({"instance_id": row["instance_id"], "repo": row["repo"],
                               "prompt": prompt, "prefill": prefill})
                if len(picked) >= args.n:
                    break
                continue
        skipped += 1

    path = os.path.join(args.out_dir, "instances.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for e in picked:
            fh.write(json.dumps(e) + "\n")
    print(f"picked {len(picked)} Lite instances ({skipped} skipped: >2 files, fetch failure, "
          f"or prompt over {args.max_prompt_tokens} tokens) -> {path}")


# ----------------------------------------------------------------------------

def generate(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()

    with open(os.path.join(args.out_dir, "instances.jsonl"), encoding="utf-8") as fh:
        instances = [json.loads(l) for l in fh]
    mine = instances[args.shard::args.num_shards]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stop_strings=STOP_STRINGS,
        tokenizer=tokenizer,
    )
    outs = {c: open(os.path.join(args.out_dir, f"pred_{c}.shard{args.shard}.jsonl"), "w",
                    encoding="utf-8") for c in CONDITIONS}
    for i, inst in enumerate(mine, 1):
        inputs = tokenizer(inst["prompt"], return_tensors="pt").to(model.device)
        n_prompt = inputs["input_ids"].shape[1]
        for cond in CONDITIONS:
            t0 = time.time()
            with torch.inference_mode():
                if cond == "base":
                    with model.disable_adapter():
                        out = model.generate(**inputs, **gen_kwargs)
                else:
                    out = model.generate(**inputs, **gen_kwargs)
            raw = tokenizer.decode(out[0][n_prompt:], skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
            patch = clean_patch(inst["prefill"] + raw)
            outs[cond].write(json.dumps({
                "instance_id": inst["instance_id"],
                "model_name_or_path": cond,
                "model_patch": patch,
            }) + "\n")
            outs[cond].flush()
            print(f"[shard {args.shard}] {i}/{len(mine)} {inst['instance_id']} {cond} "
                  f"{out.shape[1] - n_prompt} tok {time.time() - t0:.1f}s", flush=True)
    for fh in outs.values():
        fh.close()


# ----------------------------------------------------------------------------

def classify(cond, iid):
    """Per-instance outcome from the harness logs:
    resolved / unresolved (patch applied, tests ran), apply_fail, or env_error
    (Docker or harness problem, so the model was never actually tested)."""
    d = os.path.join("logs", "run_evaluation", f"v6_{cond}", cond, iid)
    report = os.path.join(d, "report.json")
    if os.path.exists(report):
        with open(report) as fh:
            return "resolved" if json.load(fh).get(iid, {}).get("resolved") else "unresolved"
    log = os.path.join(d, "run_instance.log")
    if os.path.exists(log):
        with open(log, errors="replace") as fh:
            if "Patch Apply Failed" in fh.read():
                return "apply_fail"
    return "env_error"


def score(args):
    with open(os.path.join(args.out_dir, "instances.jsonl"), encoding="utf-8") as fh:
        ids = [json.loads(l)["instance_id"] for l in fh]

    if not args.skip_harness:
        for cond in CONDITIONS:
            merged = os.path.abspath(os.path.join(args.out_dir, f"pred_{cond}.jsonl"))
            with open(merged, "w", encoding="utf-8") as out:
                for shard in sorted(glob.glob(os.path.join(args.out_dir, f"pred_{cond}.shard*.jsonl"))):
                    with open(shard, encoding="utf-8") as fh:
                        out.write(fh.read())
            # Instances that already have a report are skipped by the harness,
            # so a rerun only retries the ones that errored.
            subprocess.run([
                sys.executable, "-m", "swebench.harness.run_evaluation",
                "--dataset_name", DATASET, "--split", "test",
                "--predictions_path", merged,
                "--max_workers", str(args.max_workers),
                "--run_id", f"v6_{cond}",
                "--instance_ids", *ids,
            ])

    outcome = {c: {i: classify(c, i) for i in ids} for c in CONDITIONS}
    # Only instances where both conditions got a working environment are a
    # fair comparison.
    paired = [i for i in ids if all(outcome[c][i] != "env_error" for c in CONDITIONS)]

    def row(label, cond, subset):
        o = [outcome[cond][i] for i in subset]
        n = len(o) or 1
        applied = sum(x in ("resolved", "unresolved") for x in o)
        resolved = o.count("resolved")
        return (f"| {label} | {len(o)} | {applied} ({100 * applied / n:.0f}%) | "
                f"{resolved} ({100 * resolved / n:.1f}%) |")

    lines = [
        "# THRM v6 clean evaluation",
        "",
        f"SWE-bench Lite, {len(ids)} instances, none from repositories used in training.",
        "Oracle localization (edited files shown, windowed around the edit), greedy decoding,",
        "identical prompts for both conditions, scored by the official SWE-bench Docker harness.",
        "",
        f"## Paired comparison ({len(paired)} instances where both conditions were tested)",
        "",
        "| Condition | Tested | Patch applied | Resolved |",
        "|---|---|---|---|",
        row("Qwen2.5-Coder-7B (no adapter)", "base", paired),
        row("+ THRM v6 adapter", "thrm", paired),
        "",
        "## Coverage",
        "",
        "| Condition | Tested | Patch failed to apply | Environment error (not tested) |",
        "|---|---|---|---|",
    ]
    for cond, label in (("base", "Qwen2.5-Coder-7B (no adapter)"), ("thrm", "+ THRM v6 adapter")):
        vals = list(outcome[cond].values())
        tested = sum(v in ("resolved", "unresolved", "apply_fail") for v in vals)
        lines.append(f"| {label} | {tested} | {vals.count('apply_fail')} | {vals.count('env_error')} |")
    with open(os.path.join(args.out_dir, "outcomes.json"), "w") as fh:
        json.dump(outcome, fh, indent=1)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out_dir, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    a = parse_args()
    {"prepare": prepare, "generate": generate, "score": score}[a.stage](a)
