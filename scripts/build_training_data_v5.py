#!/usr/bin/env python3
# scripts/build_training_data_v5.py
#
# Builds THRM v5 training data from the SWE-bench *train* split, which uses
# different repositories from the Verified/Lite test sets, so evaluation on
# those stays clean. Unlike v1-v4, every example includes the source code the
# patch edits, so the model learns to find the broken line instead of
# reproducing memorised patches.
#
# Usage (from the repo root):
#   python scripts/build_training_data_v5.py --max-examples 3000

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thrm_format import ELLIPSIS, build_prompt

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")
EVAL_SETS = [
    ("princeton-nlp/SWE-bench_Verified", "test"),
    ("princeton-nlp/SWE-bench_Lite", "test"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="meta-llama/Meta-Llama-3.1-8B")
    p.add_argument("--out-dir", default="data_v5")
    p.add_argument("--max-examples", type=int, default=3000)
    p.add_argument("--val-size", type=int, default=150)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-files", type=int, default=2)
    p.add_argument("--full-file-lines", type=int, default=250,
                   help="files up to this many lines are shown whole")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default=".cache_v5_files")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Patch parsing
# ----------------------------------------------------------------------------

def split_patch(patch):
    """Return {path: [(old_start, old_len), ...]} for files modified in place."""
    files, cur = {}, None
    for line in patch.splitlines():
        if line.startswith("diff --git"):
            cur = None
        elif line.startswith("--- "):
            src = line[4:].strip()
            cur = src[2:] if src.startswith("a/") else None  # skip /dev/null (new files)
            if cur:
                files.setdefault(cur, [])
        elif cur and line.startswith("@@"):
            m = HUNK_RE.match(line)
            if m:
                files[cur].append((int(m.group(1)), int(m.group(2) or 1)))
    return {k: v for k, v in files.items() if v}


def window_file(lines, hunks, rng, full_file_lines):
    """Show the whole file if small, otherwise windows around each hunk with
    random padding so the edit is not always centred."""
    if len(lines) <= full_file_lines:
        return "\n".join(lines)
    ranges = []
    for start, length in hunks:
        before, after = rng.randint(15, 80), rng.randint(15, 80)
        lo = max(0, start - 1 - before)
        hi = min(len(lines), start - 1 + length + after)
        ranges.append([lo, hi])
    ranges.sort()
    merged = [ranges[0]]
    for lo, hi in ranges[1:]:
        if lo <= merged[-1][1] + 5:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    parts = []
    if merged[0][0] > 0:
        parts.append(ELLIPSIS)
    for idx, (lo, hi) in enumerate(merged):
        if idx:
            parts.append(ELLIPSIS)
        parts.extend(lines[lo:hi])
    if merged[-1][1] < len(lines):
        parts.append(ELLIPSIS)
    return "\n".join(parts)


# ----------------------------------------------------------------------------
# Source fetching (cached on disk so reruns are free)
# ----------------------------------------------------------------------------

def fetch_file(repo, commit, path, cache_dir):
    key = hashlib.sha1(f"{repo}@{commit}:{path}".encode()).hexdigest()
    cache_path = os.path.join(cache_dir, key)
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fh:
            return fh.read()
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200:
                with open(cache_path, "w", encoding="utf-8") as fh:
                    fh.write(resp.text)
                return resp.text
            if resp.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return None


def build_example(row, rng_seed, args):
    rng = random.Random(rng_seed)
    hunks_by_file = split_patch(row["patch"])
    if not hunks_by_file or len(hunks_by_file) > args.max_files:
        return None
    files = []
    for path, hunks in hunks_by_file.items():
        src = fetch_file(row["repo"], row["base_commit"], path, args.cache_dir)
        if src is None:
            return None
        files.append((path, window_file(src.splitlines(), hunks, rng, args.full_file_lines)))
    patch = row["patch"].strip() + "\n"
    if not patch.startswith("diff --git"):
        return None
    return {
        "instance_id": row["instance_id"],
        "repo": row["repo"],
        "prompt": build_prompt(row["problem_statement"], files),
        "completion": patch,
    }


# ----------------------------------------------------------------------------

def main():
    args = parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Anything in an evaluation set is excluded by instance id and by repo.
    eval_ids, eval_repos = set(), set()
    for name, split in EVAL_SETS:
        for r in load_dataset(name, split=split):
            eval_ids.add(r["instance_id"])
            eval_repos.add(r["repo"])
    print(f"excluding {len(eval_ids)} eval instances across {len(eval_repos)} repos")

    train = load_dataset("princeton-nlp/SWE-bench", split="train")
    rows = [r for r in train if r["instance_id"] not in eval_ids and r["repo"] not in eval_repos]
    print(f"{len(train)} train rows -> {len(rows)} after excluding eval repos")
    if not rows:
        sys.exit("no training rows left after excluding evaluation repos")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    # oversample candidates; many get dropped for length or fetch failures
    rows = rows[: args.max_examples * 3]

    examples = []
    skipped = {"shape": 0, "length": 0}
    with cf.ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(build_example, r, args.seed + i, args): i for i, r in enumerate(rows)}
        for n, fut in enumerate(cf.as_completed(futures), 1):
            ex = fut.result()
            if ex is None:
                skipped["shape"] += 1
            else:
                n_tok = len(tokenizer(ex["prompt"] + ex["completion"])["input_ids"]) + 1
                if n_tok > args.max_tokens:
                    skipped["length"] += 1
                else:
                    examples.append(ex)
            if n % 200 == 0:
                print(f"  processed {n}/{len(rows)}  kept {len(examples)}  skipped {skipped}", flush=True)
            if len(examples) >= args.max_examples + args.val_size:
                for f in futures:
                    f.cancel()
                break

    # Validation split is held out by repository, so validation loss measures
    # transfer to unseen codebases rather than memorisation.
    examples.sort(key=lambda e: e["instance_id"])
    # Smallest repos first, so the held-out repos don't take a large repo
    # (and most of the training data) with them.
    counts = {}
    for e in examples:
        counts[e["repo"]] = counts.get(e["repo"], 0) + 1
    repos = sorted(counts, key=lambda r: (counts[r], r))
    val_repos, val = set(), []
    for repo in repos:
        if len(val) >= args.val_size:
            break
        val_repos.add(repo)
        val.extend(e for e in examples if e["repo"] == repo)
    val = val[: args.val_size]
    train_ex = [e for e in examples if e["repo"] not in val_repos][: args.max_examples]

    for name, data in (("train", train_ex), ("val", val)):
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for e in data:
                fh.write(json.dumps(e) + "\n")
        print(f"wrote {len(data)} examples to {path}")
    print(f"train repos: {len({e['repo'] for e in train_ex})}   val repos: {sorted(val_repos)}")
    print("examples per repo:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
    print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
