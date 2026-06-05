# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Integration test for custom proposer class in speculative decoding.

Usage:
    .venv/bin/python test_custom_proposer.py
"""

import os

import torch

from vllm import LLM, SamplingParams
from vllm.config import VllmConfig

MODEL_ID = "facebook/opt-125m"
NUM_SPEC_TOKENS = 5


class DummyDraftProposer:
    """Custom proposer class that repeats the last token of each sequence.

    This demonstrates the class-based custom proposer interface.
    """

    def __init__(self, vllm_config: VllmConfig):
        """Initialize the custom proposer.

        Args:
            vllm_config: vLLM configuration containing model and speculative settings.
        """
        self.num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        print(
            f"[DummyDraftProposer.__init__] num_speculative_tokens="
            f"{self.num_speculative_tokens}, max_model_len={self.max_model_len}"
        )

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: int,
        token_ids_cpu: torch.Tensor,
        slot_mappings: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """Generate draft tokens by repeating the last token of each sequence.

        Args:
            sampled_token_ids: Recently sampled token IDs per request.
            num_tokens_no_spec: Number of non-speculative tokens per request.
            token_ids_cpu: Full token IDs tensor on CPU.
            slot_mappings: Slot mapping for KV cache (optional).

        Returns:
            List of draft token sequences for each request.
        """
        # Cross-process flag to prove this method was executed
        with open("proposer_called.flag", "w") as f:
            f.write("called")

        batch_size = len(sampled_token_ids)
        last_tokens = [seq[-1] for seq in sampled_token_ids]
        drafts = [[t] * self.num_speculative_tokens for t in last_tokens]
        print(
            f"[DummyDraftProposer.propose] batch_size={batch_size}, "
            f"num_speculative_tokens={self.num_speculative_tokens}, "
            f"drafts_shape={len(drafts)}x{len(drafts[0])}"
        )
        return drafts


if __name__ == "__main__":
    print("=" * 60)
    print("Custom Proposer Backend Integration Test")
    print("=" * 60)

    # Cleanup any leftover flag from previous failed runs
    if os.path.exists("proposer_called.flag"):
        os.remove("proposer_called.flag")

    llm = LLM(
        model=MODEL_ID,
        speculative_config={
            "model": f"{__name__}.DummyDraftProposer",
            "num_speculative_tokens": NUM_SPEC_TOKENS,
        },
        gpu_memory_utilization=0.4,
        enforce_eager=True,
    )

    prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(
        max_tokens=32,
        temperature=0.0,
    )

    print(f"\nRunning generate with {len(prompts)} prompt(s)...\n")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"Prompt:          {prompt!r}")
        print(f"Generated text:  {generated!r}")
        print("-" * 60)

    # Verify the custom proposer's propose() was actually called across processes
    assert os.path.exists("proposer_called.flag"), (
        "The custom proposer's propose() method was never called!"
    )
    os.remove("proposer_called.flag")

    print("✓ Custom proposer was actively used during generation!")
    print("Test completed successfully.")
