import json
import re

prediction_file = "ood_predictions.jsonl"
results = {"total": 0, "valid_syntax": 0, "valid_targets": 0, "perfect_structure": 0}

with open(prediction_file, "r") as f:
    for line in f:
        try:
            data = json.loads(line)
            patch = data.get("model_patch", "")
            if not patch:
                continue
                
            results["total"] += 1
            
            has_syntax = "--- " in patch and "+++ " in patch
            if has_syntax: 
                results["valid_syntax"] += 1
                
            has_targets = re.search(r'--- a/.*?\n\+\+\+ b/.*?', patch) is not None
            if has_targets: 
                results["valid_targets"] += 1
                
            has_hunks = re.search(r'@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@', patch) is not None
            
            if has_syntax and has_targets and has_hunks:
                results["perfect_structure"] += 1
        except json.JSONDecodeError:
            continue

if results["total"] > 0:
    print("-" * 30)
    print(f"total eval: {results['total']}")
    print(f"valid syntax: {results['valid_syntax']} ({(results['valid_syntax']/results['total'])*100:.1f}%)")
    print(f"valid targets: {results['valid_targets']} ({(results['valid_targets']/results['total'])*100:.1f}%)")
    print(f"perfect structure: {results['perfect_structure']} ({(results['perfect_structure']/results['total'])*100:.1f}%)")
    print("-" * 30)
else:
    print("no predictions evaluated")