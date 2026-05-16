import json

in_path = "thrm_cpt_training_data.jsonl"
out_path = "thrm_training_v2.jsonl"

with open(in_path) as fin, open(out_path, "w") as fout:
    for line in fin:
        row = json.loads(line)
        text = row["text"]

        # Extract user content (between <|user|> and <|assistant|>)
        user_part = text.split("<|user|>")[-1].split("<|assistant|>")[0].strip()

        # Extract completion (everything after <|assistant|>)
        completion = text.split("<|assistant|>")[-1].strip()

        # Skip if completion doesn't look like a real diff
        if not completion.startswith("diff --git") and "--- a/" not in completion:
            continue

        # Rebuild with correct Llama-3 template
        formatted = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a senior software engineer. Output a unified git diff to fix the issue. "
            "Use the standard --- a/file and +++ b/file format."
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_part}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{completion}"
            "<|eot_id|>"
        )
        fout.write(json.dumps({"text": formatted}) + "\n")

print("Done")