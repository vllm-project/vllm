# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import random
from typing import NamedTuple

import pytest
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_tp_group
from vllm.platforms import current_platform
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
    "rocm": DeviceConfig(
        available=current_platform.is_rocm() and HAS_TRITON,
        backends=["TRITON_ATTN"],
    ),
}

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
TEST_MODEL = os.getenv("VLLM_TEST_MODEL", DEFAULT_MODEL)

# Override backends for MLA models (MLA only supported on CUDA).
if os.getenv("VLLM_TEST_MODEL"):
    from vllm.transformers_utils.config import get_config
    from vllm.transformers_utils.model_arch_config_convertor import (
        ModelArchConfigConvertorBase,
    )

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
        DEVICE_BACKENDS["rocm"] = DeviceConfig(
            available=DEVICE_BACKENDS["rocm"].available,
            backends=["TRITON_MLA"],
        )

# Only include backends for devices that are actually available.
BACKENDS: list[str] = sorted(
    {b for cfg in DEVICE_BACKENDS.values() if cfg.available for b in cfg.backends}
)

skip_unsupported = pytest.mark.skipif(
    not any(cfg.available for cfg in DEVICE_BACKENDS.values()),
    reason="Requires CUDA >= Ampere (SM80), ROCm, or Intel XPU with Triton",
)

skip_if_not_cuda = pytest.mark.skipif(
    not DEVICE_BACKENDS["cuda"].available,
    reason="Requires CUDA >= Ampere (SM80)",
)

skip_if_not_cuda_alike = pytest.mark.skipif(
    not (DEVICE_BACKENDS["cuda"].available or DEVICE_BACKENDS["rocm"].available),
    reason="Requires CUDA >= Ampere (SM80) or ROCm",
)

skip_if_not_rocm = pytest.mark.skipif(
    not DEVICE_BACKENDS["rocm"].available,
    reason="Requires ROCm",
)


requires_mx = pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_mx()),
    reason="requires a ROCm device with native MX support (gfx95x)",
)


def order_sensitive_elements(probe: torch.Tensor) -> torch.Tensor:
    """Mask of probe elements whose reduction depends on the rank order.

    The collectives sum the ``world_size`` contributions of an element in rank
    order with an fp32 accumulator and round once on the way out, so an element
    can only notice a reordering if that accumulation is inexact for its
    operands. Summing the gathered contributions in the opposite order is the
    strongest reordering available and bounds what any other one can do: where
    it changes nothing, an invariance sweep cannot fail either.

    The all-gather is pure data movement, so every rank computes the same mask.
    """
    world_size = get_tp_group().world_size
    gathered = torch.empty(
        (world_size * probe.shape[0], *probe.shape[1:]),
        dtype=probe.dtype,
        device=probe.device,
    )
    dist.all_gather_into_tensor(
        gathered, probe.contiguous(), group=get_tp_group().device_group
    )
    gathered = gathered.view(world_size, *probe.shape)

    ascending = torch.zeros(probe.shape, dtype=torch.float32, device=probe.device)
    for contribution in gathered:
        ascending += contribution.float()
    descending = torch.zeros_like(ascending)
    for contribution in gathered.flip(0):
        descending += contribution.float()
    return ascending.to(probe.dtype) != descending.to(probe.dtype)


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


def shutdown_llm(llm) -> None:
    """Tear an ``LLM`` down so the next model load has its VRAM back.
    Note: Under ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` the
    engine still shuts down, but its VRAM is not recoverable in-process; run
    those tests in a spawned interpreter.
    """
    llm.llm_engine.engine_core.shutdown()
