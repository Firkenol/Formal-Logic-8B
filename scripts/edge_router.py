# edge_router.py
#
# Pure transformers inference — no vLLM, no flashinfer, no JIT compilation,
# no CUDA toolkit version dependency.  The only runtime requirement is that
# the CUDA *driver* is present, which it is (nvidia-smi works).
#
# CUDA_VISIBLE_DEVICES must be set before any import touches libcuda.
# Doing it here at module top is the only position that reliably fires
# before torch probes the device list on import.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger(__name__)


class THRMEdgeRouter:
    def __init__(
        self,
        base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        adapter_path: str = "./thrm_qlora_v3",
    ):
        self.device = torch.device("cuda:0")

        logger.info("Loading tokenizer from %s", adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info("Loading base model %s", base_model)
        # device_map=None + explicit .to() bypasses accelerate's multi-GPU
        # dispatch, which is what triggered NCCL init and the original deadlock.
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        base = base.to(self.device)

        logger.info("Attaching adapter from %s", adapter_path)
        self.model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
        self.model.eval()

        logger.info(
            "Ready on %s — %.2f GiB allocated",
            self.device,
            torch.cuda.memory_allocated(self.device) / 1024 ** 3,
        )

        # Sampling params equivalent to the vLLM config
        self._gen_kwargs = dict(
            max_new_tokens=2500,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(issue_text: str) -> str:
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a senior software engineer. "
            "Output a unified git diff to fix the issue. "
            "Use the standard --- a/file and +++ b/file format."
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"Issue:\n{issue_text}\n\n"
            "Solution:"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "```diff\n--- a/"
        )

    @staticmethod
    def _prefix(raw: str) -> str:
        return "--- a/" + raw

    @torch.inference_mode()
    def _generate(self, prompt: str, use_adapter: bool) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)

        prompt_len = inputs["input_ids"].shape[1]

        # Switch adapter on/off without reloading the model
        if use_adapter:
            self.model.enable_adapter_layers()
        else:
            self.model.disable_adapter_layers()

        output_ids = self.model.generate(**inputs, **self._gen_kwargs)
        generated = output_ids[0][prompt_len:]
        return self._prefix(self.tokenizer.decode(generated, skip_special_tokens=True))

    # ------------------------------------------------------------------
    # Public interface — original signatures preserved
    # ------------------------------------------------------------------

    def route_task(self, repo_path: str, target_dir: str, issue_text: str) -> str:
        """Generate a patch using the fine-tuned adapter."""
        return self._generate(self._build_prompt(issue_text), use_adapter=True)

    def generate_baseline(self, issue_text: str) -> str:
        """Generate a patch from the base model without the adapter."""
        return self._generate(self._build_prompt(issue_text), use_adapter=False)