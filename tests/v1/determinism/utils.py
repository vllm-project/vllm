# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import random

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.fa_utils import flash_attn_supports_mla

skip_unsupported = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(80)),
    # Supports testing on Ampere and Ada Lovelace devices.
    # Note: For devices with SM < 90, batch invariance does not support CUDA Graphs.
    reason="Requires CUDA and >= Ampere (SM80)",
)

BACKENDS: list[str] = [
    "FLASH_ATTN",
    "TRITON_MLA",
]

# FlashInfer temporarily disabled due to invariant CTA sizes.
# See FlashInfer issue #2424
# if has_flashinfer():
#     BACKENDS.append("FLASHINFER")

if flash_attn_supports_mla():
    BACKENDS.append("FLASH_ATTN_MLA")

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
MLA_MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"


def resolve_model_name(backend: str) -> str:
    """Resolve the model name for the given backend."""
    model = os.getenv("VLLM_TEST_MODEL", DEFAULT_MODEL)
    if backend.endswith("MLA") and model == DEFAULT_MODEL:
        return MLA_MODEL
    return model


def _random_prompt(min_words: int = 1024, max_words: int = 1024 * 2) -> str:
    # Generate more realistic prompts that will actually produce varied tokens
    # Use a mix of common English text patterns

    prompt_templates = [
        # Question-answer style
        "Question: What is the capital of France?\nAnswer: The capital of France is",
        "Q: How does photosynthesis work?\nA: Photosynthesis is the process by which",
        "User: Can you explain quantum mechanics?\nAssistant: Quantum mechanics is",
        # Story/narrative style
        "Once upon a time in a distant galaxy, there lived",
        "The old man walked slowly down the street, remembering",
        "In the year 2157, humanity finally discovered",
        # Technical/code style
        "To implement a binary search tree in Python, first we need to",
        "The algorithm works by iterating through the array and",
        "Here's how to optimize database queries using indexing:",
        # Factual/informative style
        "The Renaissance was a period in European history that",
        "Climate change is caused by several factors including",
        "The human brain contains approximately 86 billion neurons which",
        # Conversational style
        "I've been thinking about getting a new laptop because",
        "Yesterday I went to the store and bought",
        "My favorite thing about summer is definitely",
    ]

    # Pick a random template
    base_prompt = random.choice(prompt_templates)

    if max_words < min_words:
        max_words = min_words
    target_words = random.randint(min_words, max_words)

    if target_words > 50:
        # For longer prompts, repeat context
        padding_text = (
            " This is an interesting topic that deserves more explanation. "
            # TODO: Update to * (target_words // 10) to better align with word ratio
            * (target_words // 50)
        )
        base_prompt = padding_text + base_prompt

    return base_prompt


def _extract_step_logprobs(request_output):
    if getattr(request_output, "outputs", None):
        inner = request_output.outputs[0]
        if hasattr(inner, "logprobs") and inner.logprobs is not None:
            t = torch.tensor(
                [
                    inner.logprobs[i][tid].logprob
                    for i, tid in enumerate(inner.token_ids)
                ],
                dtype=torch.float32,
            )
            return t, inner.token_ids

    return None, None


def is_device_capability_below_90() -> bool:
    return not current_platform.has_device_capability(90)
