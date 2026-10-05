#!/usr/bin/env python3
# scripts/train_thrm_v5.py
#
# QLoRA fine-tune of Meta-Llama-3.1-8B on data_v5 (see build_training_data_v5.py).
# Changes from thrm_3b_qlora_final:
#   - training data has the source code in the prompt and comes from repos
#     that are not in SWE-bench Verified/Lite
#   - loss only on the patch tokens, and an EOS so the model learns to stop
#   - about 2 passes over the data instead of about 16
#   - LoRA on all attention and MLP projections, not just q/v
#   - validation loss on held-out repos; the best checkpoint is kept
#
# Usage (from the repo root), one process per GPU:
#   torchrun --nproc_per_node 4 scripts/train_thrm_v5.py

import argparse
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="meta-llama/Meta-Llama-3.1-8B")
    p.add_argument("--data-dir", default="data_v5")
    p.add_argument("--output-dir", default="./thrm_v5")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4,
                   help="effective batch = batch-size x grad-accum x number of GPUs")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--lora-r", type=int, default=32)
    return p.parse_args()


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def tokenize(example, tokenizer, max_tokens):
    # Prompt and completion are tokenized separately and concatenated, so the
    # label mask lands exactly on the first patch token.
    prompt_ids = tokenizer(example["prompt"])["input_ids"]  # includes BOS
    completion_ids = tokenizer(example["completion"], add_special_tokens=False)["input_ids"]
    completion_ids.append(tokenizer.eos_token_id)
    if len(prompt_ids) + len(completion_ids) > max_tokens:
        return None
    return {
        "input_ids": prompt_ids + completion_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(completion_ids)),
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def main():
    args = parse_args()

    # One process per GPU under torchrun; plain `python` uses GPU 0.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    datasets = {}
    for split in ("train", "val"):
        raw = load_jsonl(os.path.join(args.data_dir, f"{split}.jsonl"))
        rows = [t for t in (tokenize(e, tokenizer, args.max_tokens) for e in raw) if t]
        datasets[split] = ListDataset(rows)
        if local_rank == 0:
            print(f"{split}: {len(rows)} examples ({len(raw) - len(rows)} dropped for length)")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": local_rank},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()

    if not trainer.is_world_process_zero():
        return
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "train_log.json"), "w") as fh:
        json.dump(trainer.state.log_history, fh, indent=1)
    print(f"saved best adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
