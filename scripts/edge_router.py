# edge_router.py
#
# CUDA_VISIBLE_DEVICES must be set before any import touches libcuda.
# vLLM's platforms/__init__.py forks a subprocess to enumerate device
# capabilities; if more than one device is visible, that subprocess races
# the parent on Python's import lock and deadlocks silently.
# Pinning to device 0 here, at the very top of the module, is the only
# position that fires before vLLM's import-time platform probe runs.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_WORKER_USE_RAY"] = "0"       # disable Ray backend entirely
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class THRMEdgeRouter:
    def __init__(
        self,
        base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        adapter_path: str = "./thrm_qlora_v3",
    ):
        # tensor_parallel_size=1 with a single visible device means vLLM never
        # attempts to init a NCCL communicator ring across multiple GPUs.
        # An 8B model in bfloat16 is ~16 GiB; 0.85 utilisation on an 80 GB
        # A100 leaves plenty of headroom for the KV cache.
        self.llm = LLM(
            model=base_model,
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=64,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            enforce_eager=True,       # skip CUDA graph capture; faster cold start
            trust_remote_code=False,
        )
        self.adapter_path = adapter_path

        self.sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=2500,
            skip_special_tokens=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
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
    def _prefix_output(raw: str) -> str:
        # The prompt already primed the model mid-header; re-attach the
        # prefix so the caller receives a syntactically complete diff.
        return "--- a/" + raw

    # ------------------------------------------------------------------
    # Public interface — original signatures preserved exactly
    # ------------------------------------------------------------------

    def route_task(self, repo_path: str, target_dir: str, issue_text: str) -> str:
        """Generate a patch using the fine-tuned adapter."""
        prompt = self._build_prompt(issue_text)
        lora_req = LoRARequest("thrm_adapter", 1, self.adapter_path)
        outputs = self.llm.generate(
            [prompt],
            sampling_params=self.sampling_params,
            lora_request=lora_req,
        )
        return self._prefix_output(outputs[0].outputs[0].text)

    def generate_baseline(self, issue_text: str) -> str:
        """Generate a patch from the base model without the adapter."""
        prompt = self._build_prompt(issue_text)
        outputs = self.llm.generate([prompt], sampling_params=self.sampling_params)
        return self._prefix_output(outputs[0].outputs[0].text)