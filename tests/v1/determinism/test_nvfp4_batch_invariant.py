# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import random

import pytest
import torch
from utils import (
    _extract_step_logprobs,
    _random_prompt,
    skip_unsupported,
)

from vllm import LLM, SamplingParams

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="NVFP4 tests require torch.float8_e4m3fn support.",
)

NVFP4_TEST_MODEL = os.getenv(
    "VLLM_TEST_NVFP4_MODEL", "nm-testing/TinyLlama-1.1B-Chat-v1.0-NVFP4"
)


def _make_llm(max_num_seqs: int, backend: str) -> LLM:
    return LLM(
        model=NVFP4_TEST_MODEL,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=float(
            os.getenv("VLLM_NVFP4_TEST_GPU_MEMORY_UTILIZATION", "0.05")
        ),
        max_model_len=int(os.getenv("VLLM_NVFP4_TEST_MAX_MODEL_LEN", "2048")),
        dtype="auto",
        tensor_parallel_size=int(os.getenv("VLLM_NVFP4_TEST_TP_SIZE", "1")),
        enable_prefix_caching=False,
        enforce_eager=True,
        attention_config={"backend": backend},
    )


@skip_unsupported
@pytest.mark.parametrize("backend", ["FLASH_ATTN"])
def test_dense_nvfp4_generation_is_deterministic_across_batch_sizes_e2e(backend):
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)

    num_trials = int(os.getenv("VLLM_NVFP4_NEEDLE_TRIALS", "2"))
    max_batch_size = int(os.getenv("VLLM_NVFP4_NEEDLE_BATCH_SIZE", "8"))
    min_random_prompt = int(os.getenv("VLLM_NVFP4_MIN_PROMPT", "32"))
    max_random_prompt = int(os.getenv("VLLM_NVFP4_MAX_PROMPT", "96"))
    assert max_batch_size >= 2, "Batch size should be >= 2 to test invariance."

    sampling = SamplingParams(
        temperature=float(os.getenv("VLLM_NVFP4_NEEDLE_TEMPERATURE", "0.6")),
        top_p=float(os.getenv("VLLM_NVFP4_NEEDLE_TOP_P", "0.95")),
        max_tokens=int(os.getenv("VLLM_NVFP4_NEEDLE_MAX_TOKENS", "16")),
        seed=20240919,
        logprobs=5,
    )
    needle_prompt = "Write one factual sentence about the moon."

    llm = None
    baseline_completion = None
    baseline_logprobs = None
    try:
        llm = _make_llm(max_num_seqs=max_batch_size, backend=backend)
        baseline_output = llm.generate([needle_prompt], sampling, use_tqdm=False)[0]
        baseline_completion = baseline_output.outputs[0]
        baseline_logprobs, baseline_token_ids = _extract_step_logprobs(baseline_output)
        assert baseline_logprobs is not None
        assert baseline_token_ids is not None
        for _ in range(num_trials):
            batch_size = random.randint(max_batch_size // 2, max_batch_size)
            needle_pos = random.randint(0, batch_size - 1)
            prompts: list[str] = []
            for idx in range(batch_size):
                if idx == needle_pos:
                    prompts.append(needle_prompt)
                else:
                    prompts.append(_random_prompt(min_random_prompt, max_random_prompt))

            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            needle_output = outputs[needle_pos]
            needle_completion = needle_output.outputs[0]
            needle_logprobs, needle_token_ids = _extract_step_logprobs(needle_output)
            assert needle_logprobs is not None
            assert needle_token_ids is not None

            assert needle_output.prompt == needle_prompt
            assert baseline_completion is not None
            assert baseline_logprobs is not None
            assert needle_completion.token_ids == baseline_completion.token_ids
            assert needle_completion.text == baseline_completion.text
            torch.testing.assert_close(needle_logprobs, baseline_logprobs)
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()
