# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opaque runtime-M dispatch for qualified SM120 block-FP8 projections."""

import torch

from vllm import _custom_ops as ops


def prepare_scale64(layer, scale):
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

    derived = scale.repeat_interleave(2, dim=0)
    storage = getattr(layer, "_cutlass_n64_scale", None)
    info = get_layerwise_info(layer)
    if info.kernel_tensors is not None:
        storage = info.kernel_tensors[1].get("_cutlass_n64_scale", storage)
    if storage is not None:
        if (
            storage.shape != derived.shape
            or storage.dtype != derived.dtype
            or storage.device != derived.device
        ):
            raise RuntimeError("SM120 derived scale storage changed during reload")
        # Reload skips non-checkpoint nonpersistent buffers before restoring
        # saved storage. Update that allocation now, while PWAL scales are live.
        storage.copy_(derived)
        derived = storage
    return derived


def select_route(a, b, sa, sb, sb64):
    m, k = a.shape
    common = (
        k in (4096, 9216)
        and b.shape == (2560, k)
        and a.dtype == b.dtype == torch.float8_e4m3fn
        and sa.dtype == sb.dtype == torch.float32
        and a.is_cuda
        and a.device == b.device == sa.device == sb.device
        and a.stride(1) == b.stride(1) == sb.stride(1) == 1
        and sa.shape == (m, k // 128)
        and sb.shape == (20, k // 128)
    )
    if common and 1 <= m <= 8:
        return "ordered"
    if (
        common
        and 16 <= m <= 128
        and m % 8 == 0
        and a.is_contiguous()
        and b.is_contiguous()
        and sb.is_contiguous()
        and sb64.dtype == torch.float32
        and sb64.device == a.device
        and sb64.shape == (40, k // 128)
        and sb64.is_contiguous()
        and sa.stride() == (1, m)
    ):
        return "n64"
    return "stock"


@torch.library.custom_op(
    "vllm::cutlass_block_fp8_sm120", mutates_args=(), device_types="cuda"
)
def cutlass_block_fp8_sm120(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    sb64: torch.Tensor,
) -> torch.Tensor:
    route = select_route(a, b, sa, sb, sb64)
    if route == "ordered":
        from .ordered_block_fp8 import ordered_block_fp8_mm

        m, k = a.shape
        out = torch.empty((m, 2560), device=a.device, dtype=torch.bfloat16)
        raw = torch.empty((k // 128, m, 2560), device=a.device, dtype=torch.float32)
        return ordered_block_fp8_mm(a, b, sa, sb, out, raw)
    if route == "n64":
        out = torch.empty(
            (a.shape[0], b.shape[0]), device=a.device, dtype=torch.bfloat16
        )
        torch.ops._C.cutlass_scaled_mm_sm120_n64(out, a, b.T, sa, sb64.T)
        return out
    return ops.cutlass_scaled_mm(
        a, b.T, scale_a=sa, scale_b=sb.T, out_dtype=torch.bfloat16
    )


@cutlass_block_fp8_sm120.register_fake
def _fake(a, b, sa, sb, sb64):
    return a.new_empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16)
