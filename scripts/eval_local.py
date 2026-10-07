#!/usr/bin/env python3
# scripts/eval_local.py
#
# Docker-free scoring of the patches eval_v6.py generated. Each predicted
# patch and the reference (gold) patch are applied to the original source
# files at the issue's base commit, and the two edits are compared:
#
#   applied        patch applies (same commands the SWE-bench harness tries)
#   right file     edits at least one file the reference fix edits
#   right lines    edits overlap the lines the reference fix changes (+-3 lines)
#   exact fix      resulting files identical to the reference fix (ignoring
#                  trailing whitespace)
#
# Both conditions are scored on the same instances, so the comparison is paired.
#
#   python scripts/eval_local.py

import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_training_data_v5 import fetch_file

DATASET = "princeton-nlp/SWE-bench_Lite"
CONDITIONS = (("base", "Qwen2.5-Coder-7B (no adapter)"), ("thrm", "+ THRM v6 adapter"))
LINE_TOLERANCE = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="eval_v6")
    p.add_argument("--cache-dir", default=".cache_v5_files")
    return p.parse_args()


def patch_paths(patch):
    paths = []
    for line in patch.splitlines():
        if line.startswith("--- a/"):
            p = line[6:].strip()
            if p not in paths:
                paths.append(p)
    return paths


def apply(patch, originals):
    """Apply patch to a scratch copy of originals; return {path: new_text} or None."""
    work = tempfile.mkdtemp(prefix="evl_")
    try:
        for path, text in originals.items():
            full = os.path.join(work, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        with open(os.path.join(work, ".fix.patch"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(patch)
        for cmd in (["git", "apply", "--verbose", ".fix.patch"],
                    ["patch", "--batch", "--fuzz=5", "-p1", "-i", ".fix.patch"]):
            res = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
            if res.returncode == 0:
                out = {}
                for path in originals:
                    full = os.path.join(work, path)
                    if os.path.exists(full):
                        with open(full, encoding="utf-8", newline="") as fh:
                            out[path] = fh.read()
                return out
            # undo any partial application before the next attempt
            for path, text in originals.items():
                with open(os.path.join(work, path), "w", encoding="utf-8", newline="") as fh:
                    fh.write(text)
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def changed_lines(old, new):
    """Line indices in old that were modified, deleted, or had text inserted at."""
    sm = difflib.SequenceMatcher(None, old.splitlines(), new.splitlines(), autojunk=False)
    lines = set()
    for tag, i1, i2, _, _ in sm.get_opcodes():
        if tag != "equal":
            lines.update(range(i1, max(i2, i1 + 1)))
    return lines


def norm(text):
    return "\n".join(l.rstrip() for l in text.splitlines()).strip()


def score_one(pred_patch, gold_patch, row, cache_dir):
    paths = list(dict.fromkeys(patch_paths(gold_patch) + patch_paths(pred_patch)))
    originals = {}
    for path in paths:
        src = fetch_file(row["repo"], row["base_commit"], path, cache_dir)
        if src is not None:
            originals[path] = src
    gold = apply(gold_patch, originals)
    if gold is None:
        return None  # reference patch itself doesn't apply to the fetched files; skip
    result = {"applied": False, "right_file": False, "right_lines": False, "exact": False}
    if not pred_patch.strip():
        return result
    pred = apply(pred_patch, originals)
    if pred is None:
        return result
    result["applied"] = True

    gold_files = {p for p in originals if gold.get(p) != originals[p]}
    pred_files = {p for p in originals if pred.get(p) != originals[p]}
    result["right_file"] = bool(gold_files & pred_files)
    for path in gold_files & pred_files:
        g = changed_lines(originals[path], gold[path])
        q = changed_lines(originals[path], pred[path])
        if any(abs(a - b) <= LINE_TOLERANCE for a in q for b in g):
            result["right_lines"] = True
            break
    result["exact"] = bool(pred_files) and pred_files == gold_files and all(
        norm(pred[p]) == norm(gold[p]) for p in gold_files)
    return result


def main():
    args = parse_args()
    from datasets import load_dataset

    with open(os.path.join(args.out_dir, "instances.jsonl"), encoding="utf-8") as fh:
        ids = [json.loads(l)["instance_id"] for l in fh]
    rows = {r["instance_id"]: r for r in load_dataset(DATASET, split="test")}

    preds = {}
    for cond, _ in CONDITIONS:
        preds[cond] = {}
        path = os.path.join(args.out_dir, f"pred_{cond}.jsonl")
        if not os.path.exists(path):
            shards = sorted(f for f in os.listdir(args.out_dir)
                            if f.startswith(f"pred_{cond}.shard"))
            lines = []
            for s in shards:
                with open(os.path.join(args.out_dir, s), encoding="utf-8") as fh:
                    lines += fh.readlines()
        else:
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
        for l in lines:
            p = json.loads(l)
            preds[cond][p["instance_id"]] = p["model_patch"]

    scores = {cond: {} for cond, _ in CONDITIONS}
    skipped = []
    for n, iid in enumerate(ids, 1):
        row = rows[iid]
        per = {cond: score_one(preds[cond].get(iid, ""), row["patch"], row, args.cache_dir)
               for cond, _ in CONDITIONS}
        if any(v is None for v in per.values()):
            skipped.append(iid)
            continue
        for cond in per:
            scores[cond][iid] = per[cond]
        if n % 10 == 0:
            print(f"  scored {n}/{len(ids)}", flush=True)

    scored = sorted(scores["base"])
    metrics = (("applied", "Patch applies"), ("right_file", "Edits right file"),
               ("right_lines", "Edits right lines"), ("exact", "Identical to reference fix"))
    n = len(scored) or 1
    lines = [
        "# THRM v6 patch quality (paired, Docker-free)",
        "",
        f"{len(scored)} SWE-bench Lite instances, none from repositories used in training,",
        "same prompt for both conditions, greedy decoding. Each patch is applied to the",
        "original source at the issue's base commit and compared with the reference fix.",
        f"'Right lines' means the edit is within {LINE_TOLERANCE} lines of a line the reference fix changes.",
        "",
        "| Metric | " + " | ".join(label for _, label in CONDITIONS) + " | Only base | Only adapter |",
        "|---|---|---|---|---|",
    ]
    for key, label in metrics:
        base = {i for i in scored if scores["base"][i][key]}
        thrm = {i for i in scored if scores["thrm"][i][key]}
        lines.append(f"| {label} | {len(base)} ({100 * len(base) / n:.0f}%) | "
                     f"{len(thrm)} ({100 * len(thrm) / n:.0f}%) | "
                     f"{len(base - thrm)} | {len(thrm - base)} |")
    if skipped:
        lines += ["", f"Skipped {len(skipped)} instances whose reference patch did not apply "
                      f"to the fetched source: {', '.join(skipped)}"]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out_dir, "results_local.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(args.out_dir, "scores_local.json"), "w") as fh:
        json.dump(scores, fh, indent=1)
    print(text)


if __name__ == "__main__":
    main()
