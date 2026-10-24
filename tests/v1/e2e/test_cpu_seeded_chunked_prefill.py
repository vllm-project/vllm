# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU seeded sampling must survive a prompt that chunked prefill splits."""

import os

import pytest

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

pytestmark = pytest.mark.cpu_model

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

# Bound the KV cache so the run does not scale with host memory.
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")

MODEL = "facebook/opt-125m"
# Long enough to span several 32-token chunks; the short one fits in one.
PROMPTS = [
    "The quick brown fox jumps over the lazy dog. " * 6,
    "Once upon a time, " * 9,
    "Hello",
]
SAMPLING_PARAMS = [
    SamplingParams(temperature=0.8, seed=seed, max_tokens=12) for seed in (1, 2, 3)
]


def _generate(**overrides) -> list[list[int]]:
    llm = LLM(
        model=MODEL,
        dtype="float32",
        max_model_len=512,
        enforce_eager=True,
        enable_prefix_caching=False,
        **overrides,
    )
    outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)
    del llm
    return [list(o.outputs[0].token_ids) for o in outputs]


def test_seeded_sampling_matches_with_and_without_chunked_prefill():
    """A seeded request whose prompt is split across steps must neither crash
    the engine nor draw different tokens than when it is prefilled at once."""
    reference = _generate(enable_chunked_prefill=False)
    chunked = _generate(enable_chunked_prefill=True, max_num_batched_tokens=32)
    assert chunked == reference
