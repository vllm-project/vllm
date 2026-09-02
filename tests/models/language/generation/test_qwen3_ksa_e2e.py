# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from ....utils import large_gpu_mark
from ...registry import HF_EXAMPLE_MODELS

MODEL_INFO = HF_EXAMPLE_MODELS.get_hf_info("Qwen3KSAForCausalLM")


@pytest.mark.slow_test
@large_gpu_mark(min_gb=32)
def test_summary_boundary_matches_hf_reference(vllm_runner) -> None:
    # This prompt crosses the 1,032-token text-cache window. The expected
    # continuation comes from the pinned Hugging Face implementation with
    # use_cache=False, so it is independent of vLLM's cache implementation.
    prompt_token_ids = list(range(100, 1133))

    with vllm_runner(
        MODEL_INFO.default,
        revision=MODEL_INFO.revision,
        dtype=MODEL_INFO.dtype,
        enforce_eager=MODEL_INFO.enforce_eager,
        enable_prefix_caching=MODEL_INFO.enable_prefix_caching,
        max_model_len=2048,
        max_num_seqs=1,
        max_num_batched_tokens=256,
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.4,
        hf_overrides=MODEL_INFO.hf_overrides,
    ) as runner:
        output_token_ids = runner.generate_greedy([prompt_token_ids], max_tokens=3)[0][
            0
        ]

    assert output_token_ids[len(prompt_token_ids) :] == [82, 13, 220]
