# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime adapter for checked-in per-token-group FP8 quant kernels."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from vllm.kernels.helion_generated.dispatcher import (
    _load_launcher,
    _runtime_platform,
    _select_bucketed_module,
    vllm_helion_generated_lib,
)
from vllm.kernels.helion_generated.manifests import MANIFESTS
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

OP_NAME = "per_token_group_fp8_quant"
NATIVE_OP_NAME = None


def _select_module(
    platform: str | None,
    hidden_size: int,
    group_size: int,
    num_tokens: int,
) -> str | None:
    return _select_bucketed_module(
        OP_NAME,
        platform,
        (hidden_size, group_size),
        num_tokens,
    )


def _expected_scale_stride(
    num_tokens: int,
    groups_per_row: int,
    column_major: bool,
    tma_aligned: bool,
) -> tuple[int, int]:
    if not column_major:
        return (groups_per_row, 1)
    if tma_aligned:
        return (1, (num_tokens + 3) // 4 * 4)
    return (1, num_tokens)


def _eligible_module(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    column_major: bool,
    tma_aligned: bool,
) -> str | None:
    if (
        input.ndim != 2
        or input.dtype != torch.bfloat16
        or not input.is_cuda
        or not input.is_contiguous()
        or output_q.shape != input.shape
        or output_q.dtype != current_platform.fp8_dtype()
        or output_q.device != input.device
        or not output_q.is_contiguous()
        or output_s.ndim != 2
        or output_s.dtype != torch.float32
        or output_s.device != input.device
    ):
        return None

    num_tokens, hidden_size = input.shape
    if group_size < 1 or group_size & (group_size - 1) or hidden_size % group_size != 0:
        return None
    groups_per_row = hidden_size // group_size
    if output_s.shape != (num_tokens, groups_per_row):
        return None
    if output_s.stride() != _expected_scale_stride(
        num_tokens, groups_per_row, column_major, tma_aligned
    ):
        return None
    return _select_module(_runtime_platform(), hidden_size, group_size, num_tokens)


def _native_fallback(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    column_major: bool,
    tma_aligned: bool,
) -> None:
    torch.ops._C.per_token_group_fp8_quant(
        input,
        output_q,
        output_s,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
        column_major,
        tma_aligned,
    )


def per_token_group_fp8_quant(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    column_major: bool = False,
    tma_aligned: bool = False,
) -> None:
    module_path = _eligible_module(
        input,
        output_q,
        output_s,
        group_size,
        column_major,
        tma_aligned,
    )
    if module_path is None:
        _native_fallback(
            input,
            output_q,
            output_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            scale_ue8m0,
            column_major,
            tma_aligned,
        )
        return
    _load_launcher(module_path)(
        input,
        output_q,
        output_s,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
        column_major,
        tma_aligned,
    )


def _fake(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    column_major: bool = False,
    tma_aligned: bool = False,
) -> None:
    return None


def selected_token_buckets(token_counts: Iterable[int]) -> tuple[int, ...]:
    platform = _runtime_platform()
    kernels = MANIFESTS.get(platform or "", {})
    available = sorted({key[2] for key in kernels})
    if not available:
        return ()
    selected = {
        next((bucket for bucket in available if bucket >= count), available[-1])
        for count in token_counts
        if count > 0
    }
    return tuple(sorted(selected))


def warmup(
    token_counts: Iterable[int],
    device: torch.device | str = "cuda",
) -> None:
    platform = _runtime_platform()
    kernels = MANIFESTS.get(platform or "", {})
    buckets = selected_token_buckets(token_counts)
    shapes = sorted({(hidden, group) for hidden, group, _ in kernels})
    if not buckets or not shapes:
        return

    fp8_dtype = current_platform.fp8_dtype()
    fp8_info = torch.finfo(fp8_dtype)
    for hidden_size, group_size in shapes:
        for num_tokens in buckets:
            input = torch.empty(
                (num_tokens, hidden_size), device=device, dtype=torch.bfloat16
            )
            output_q = torch.empty_like(input, dtype=fp8_dtype)
            output_s = torch.empty(
                (num_tokens, hidden_size // group_size),
                device=device,
                dtype=torch.float32,
            )
            for scale_ue8m0 in (False, True):
                per_token_group_fp8_quant(
                    input,
                    output_q,
                    output_s,
                    group_size,
                    1e-10,
                    fp8_info.min,
                    fp8_info.max,
                    scale_ue8m0,
                )


direct_register_custom_op(
    op_name="per_token_group_fp8_quant",
    op_func=per_token_group_fp8_quant,
    mutates_args=["output_q", "output_s"],
    fake_impl=_fake,
    target_lib=vllm_helion_generated_lib,
)
