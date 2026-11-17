# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU watermarking: opening tokens use ordinary sampling."""

import os

import pytest

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

pytestmark = pytest.mark.cpu_model

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")

CONTEXT_WIDTH = 4


def test_cpu_watermarking_samples_opening_tokens_normally() -> None:
    llm = LLM(
        model="facebook/opt-125m",
        watermark_config={"key": 42, "context_width": CONTEXT_WIDTH},
        max_model_len=512,
        enforce_eager=True,
    )
    prompt = "Tell me a story about an explorer who discovers a mysterious island."

    def opening(seed: int, watermarking: bool) -> list[int]:
        params = SamplingParams(
            temperature=1.0,
            max_tokens=CONTEXT_WIDTH,
            seed=seed,
            watermarking=watermarking,
            ignore_eos=True,
        )
        return list(llm.generate(prompt, params)[0].outputs[0].token_ids)

    openings = [opening(seed, watermarking=True) for seed in range(4)]

    assert openings == [opening(seed, watermarking=False) for seed in range(4)]
    assert len({tuple(tokens) for tokens in openings}) > 1
