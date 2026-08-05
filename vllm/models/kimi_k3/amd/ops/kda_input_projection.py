# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fail-closed AMD-only Kimi-K3 KDA input-projection dispatch."""

from __future__ import annotations

from collections.abc import Callable

import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


def _kda_input_group64_impl(
    hidden_states: torch.Tensor,
    packed_weight: torch.Tensor,
    packed_scale: torch.Tensor,
) -> torch.Tensor:
    from aiter.ops.flydsl.kimi_k3_kda_input_group64 import (
        kimi_k3_kda_input_group64,
    )

    return kimi_k3_kda_input_group64(hidden_states, packed_weight, packed_scale)


def _kda_input_group64_fake(
    hidden_states: torch.Tensor,
    packed_weight: torch.Tensor,
    packed_scale: torch.Tensor,
) -> torch.Tensor:
    del packed_weight, packed_scale
    return hidden_states.new_empty((hidden_states.shape[0], 6288))


direct_register_custom_op(
    op_name="kimi_k3_kda_input_group64",
    op_func=_kda_input_group64_impl,
    mutates_args=[],
    fake_impl=_kda_input_group64_fake,
    dispatch_key=current_platform.dispatch_key,
)


def prepack_kda_input_group64(
    bf16_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return a model-load-time pack, or ``None`` for exact fallback."""

    try:
        from aiter.ops.flydsl.kimi_k3_kda_input_group64 import (
            quantize_kimi_k3_kda_input_group64,
        )
    except (ImportError, ModuleNotFoundError):
        return None
    if (
        not torch.version.hip
        or bf16_weight.dtype != torch.bfloat16
        or tuple(bf16_weight.shape) != (6288, 7168)
        or not bf16_weight.is_cuda
        or not bf16_weight.is_contiguous()
    ):
        return None
    try:
        properties = torch.cuda.get_device_properties(bf16_weight.device)
    except (AssertionError, RuntimeError):
        return None
    if str(getattr(properties, "gcnArchName", "")).split(":", 1)[0] != "gfx950":
        return None
    return quantize_kimi_k3_kda_input_group64(bf16_weight)


def kda_input_projection(
    hidden_states: torch.Tensor,
    bf16_fallback: Callable[[torch.Tensor], torch.Tensor],
    packed_weight: torch.Tensor | None,
    packed_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Use group64 only for the exact typed contract; otherwise BF16 fallback."""

    if packed_weight is None or packed_scale is None:
        return bf16_fallback(hidden_states)
    try:
        from aiter.ops.flydsl.kimi_k3_kda_input_group64 import (
            supports_kimi_k3_kda_input_group64,
        )
    except (ImportError, ModuleNotFoundError):
        return bf16_fallback(hidden_states)
    if not supports_kimi_k3_kda_input_group64(
        hidden_states, packed_weight, packed_scale
    ):
        return bf16_fallback(hidden_states)
    return torch.ops.vllm.kimi_k3_kda_input_group64(
        hidden_states,
        packed_weight,
        packed_scale,
    )
