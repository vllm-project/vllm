# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel selection rules shared by every linear-attention op."""

from enum import StrEnum

import pytest

from vllm.model_executor.layers.linear_attention.kernel import (
    LinearAttnKernel,
    LinearAttnSelectorConfig,
    select_linear_attn_kernel,
)
from vllm.platforms.interface import DeviceCapability, PlatformEnum

CFG = LinearAttnSelectorConfig(
    platform=PlatformEnum.CUDA, device_capability=DeviceCapability(9, 0)
)


class Fake(StrEnum):
    FAST = "fast"
    SLOW = "slow"
    FALLBACK = "fallback"
    OTHER = "other"


def _kernel(backend: Fake, *reasons: str) -> type[LinearAttnKernel]:
    return type(
        f"Fake{backend.name}",
        (LinearAttnKernel,),
        {
            "backend": backend,
            "validate_configuration": classmethod(lambda c, cfg: [*reasons]),
        },
    )


FAST = _kernel(Fake.FAST, "needs SM10x")
SLOW = _kernel(Fake.SLOW)
FALLBACK = _kernel(Fake.FALLBACK)


def test_auto_picks_the_first_valid_kernel():
    selected = select_linear_attn_kernel("op_auto", [FAST, SLOW, FALLBACK], CFG)
    assert selected is SLOW


def test_requested_valid_choice_wins_over_priority():
    selected = select_linear_attn_kernel(
        "op_explicit", [SLOW, FALLBACK], CFG, Fake.FALLBACK
    )
    assert selected is FALLBACK


def test_requested_invalid_choice_fails_with_its_reasons():
    """Users who name a backend must get it or an error, never another one."""
    with pytest.raises(ValueError, match="needs SM10x"):
        select_linear_attn_kernel("op_strict", [FAST, SLOW], CFG, Fake.FAST)


def test_requested_unknown_choice_lists_the_op_kernels():
    with pytest.raises(ValueError, match="fast, slow"):
        select_linear_attn_kernel("op_unknown", [FAST, SLOW], CFG, Fake.OTHER)


def test_no_valid_kernel_fails_with_every_reason():
    with pytest.raises(ValueError, match=r"fast.*needs SM10x.*other.*needs CPU"):
        select_linear_attn_kernel(
            "op_none", [FAST, _kernel(Fake.OTHER, "needs CPU")], CFG
        )


def test_requested_backend_uses_its_first_valid_kernel():
    """A backend may offer several kernels for an op, e.g. by decoding mode."""
    fast_cpu = _kernel(Fake.FAST)
    selected = select_linear_attn_kernel(
        "op_shared", [FAST, fast_cpu, SLOW], CFG, Fake.FAST
    )
    assert selected is fast_cpu


def test_requested_backend_fails_with_each_kernels_reasons():
    with pytest.raises(ValueError, match=r"needs SM10x \| needs CPU"):
        select_linear_attn_kernel(
            "op_shared_strict",
            [FAST, _kernel(Fake.FAST, "needs CPU"), SLOW],
            CFG,
            Fake.FAST,
        )
