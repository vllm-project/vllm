# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import random
from typing import NamedTuple

import pytest
import torch

from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backends.fa_utils import flash_attn_supports_mla


class DeviceConfig(NamedTuple):
    available: bool
    backends: list[str]


# Maps each device to its availability and supported backends.
DEVICE_BACKENDS: dict[str, DeviceConfig] = {
    "cuda": DeviceConfig(
        available=current_platform.is_cuda()
        and current_platform.has_device_capability(80),
        # FlashInfer backend temporarily disabled due to invariant CTA sizes.
        # See FlashInfer issue #2424
        backends=["FLASH_ATTN", "TRITON_ATTN", "FLEX_ATTENTION"],
    ),
    "xpu": DeviceConfig(
        available=current_platform.is_xpu() and HAS_TRITON,
        backends=["TRITON_ATTN"],
    ),
}

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
TEST_MODEL = os.getenv("VLLM_TEST_MODEL", DEFAULT_MODEL)

# Override backends for MLA models (MLA only supported on CUDA).
if os.getenv("VLLM_TEST_MODEL"):
    config = get_config(TEST_MODEL, trust_remote_code=False)
    if ModelArchConfigConvertorBase(config, config.get_text_config()).is_deepseek_mla():
        DEVICE_BACKENDS["cuda"] = DeviceConfig(
            available=DEVICE_BACKENDS["cuda"].available,
            backends=["TRITON_MLA"]
            + (["FLASH_ATTN_MLA"] if flash_attn_supports_mla() else []),
        )
        DEVICE_BACKENDS["xpu"] = DeviceConfig(
            available=DEVICE_BACKENDS["xpu"].available,
            backends=[],
        )

# Only include backends for devices that are actually available.
BACKENDS: list[str] = sorted(
    {b for cfg in DEVICE_BACKENDS.values() if cfg.available for b in cfg.backends}
)

skip_unsupported = pytest.mark.skipif(
    not any(cfg.available for cfg in DEVICE_BACKENDS.values()),
    reason="Requires CUDA >= Ampere (SM80) or Intel XPU with Triton",
)

skip_if_not_cuda = pytest.mark.skipif(
    not DEVICE_BACKENDS["cuda"].available,
    reason="Requires CUDA >= Ampere (SM80)",
)


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
