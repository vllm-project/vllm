# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Integration test for custom_proposer_backend in speculative decoding.

Usage:
    .venv/bin/python test_custom_proposer.py
"""

import torch

from vllm import LLM, SamplingParams

MODEL_ID = "facebook/opt-125m"
NUM_SPEC_TOKENS = 5


def dummy_draft_func(batch_input_ids, draft_len, **kwargs):
    """Repeat the last token of each sequence draft_len times.

    Returns:
        torch.Tensor of shape [batch_size, draft_len], dtype torch.long.
    """
    batch_size = len(batch_input_ids)
    last_tokens = [seq[-1] for seq in batch_input_ids]
    drafts = torch.tensor(
        [[t] * draft_len for t in last_tokens],
        dtype=torch.long,
    )
    print(
        f"[dummy_draft_func] batch_size={batch_size}, "
        f"draft_len={draft_len}, shape={drafts.shape}"
    )
    return drafts


if __name__ == "__main__":
    print("=" * 60)
    print("Custom Proposer Backend Integration Test")
    print("=" * 60)

    llm = LLM(
        model=MODEL_ID,
        speculative_config={
            "num_speculative_tokens": NUM_SPEC_TOKENS,
            "custom_proposer_backend": f"{__name__}.dummy_draft_func",
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

    print("Test completed successfully.")
