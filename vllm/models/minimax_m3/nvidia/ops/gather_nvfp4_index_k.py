# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather NVFP4 index-K pages into compact FP8 pages for the Q8KV8 prefill indexer.

A packed NVFP4 page ``[128, 72]`` holds the E2M1 codes of all 128 tokens (64 B
per token, low nibble first) followed by their E4M3 16-element block scales
(``token * 8 + group``), with no global scale. Each element dequantizes to
``e4m3_rn_satfinite(e2m1 * scale)``, the value the Q8KV4 decode indexer scores,
so prefill and decode rank pages from identical keys.

Where Triton can emit it, each four elements are one QMUL4
(``mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4``), the instruction the decode kernel
dequantizes with; otherwise the product is formed exactly in fp32 and rounded
once, which gives the same bytes.
"""

import functools

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

PAGE_SIZE = 128
HEAD_DIM = 128
E4M3_MAX = 448.0
# QMUL4 needs PTX ISA 9.4 (ptxas 13.4) and an sm_100a/sm_103a target.
QMUL4_MIN_PTX_VERSION = 94


@functools.cache
def _qmul4_supported(device_index: int) -> bool:
    """Whether Triton emits PTX that can use QMUL4 on this device.

    Triton takes the PTX ISA version from its Blackwell ptxas: the bundled one
    unless ``TRITON_PTXAS_BLACKWELL_PATH`` points at a CUDA 13.4+ ptxas.
    """
    capability = current_platform.get_device_capability(device_index)
    if capability is None or capability.to_int() not in (100, 103):
        return False
    try:
        from triton.backends.nvidia.compiler import get_ptxas, ptx_get_version

        ptxas_version = get_ptxas(capability.to_int()).version
        return ptx_get_version(ptxas_version) >= QMUL4_MIN_PTX_VERSION
    except Exception:
        # Triton-internal helpers; without them keep the portable path.
        return False


@triton.jit
def _e2m1_to_f32(code):
    # |code| 0..7 -> 0, 0.5, 1, 1.5, 2, 3, 4, 6; bit 3 is the sign.
    exponent = (code >> 1) & 3
    mantissa = (code & 1).to(tl.float32)
    normal = (2.0 + mantissa) * (1 << exponent).to(tl.float32) * 0.25
    value = tl.where(exponent == 0, 0.5 * mantissa, normal)
    # Multiply (not negate) so that code 8 stays -0, as in the decode kernel.
    return value * tl.where((code & 8) != 0, -1.0, 1.0)


@triton.jit
def _gather_nvfp4_index_k_fp8_kernel(
    kv_cache,
    block_table,
    num_pages,
    page_offsets,
    out,
    kv_page_stride,
    block_table_stride,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    E4M3_MAX: tl.constexpr,
    USE_QMUL4: tl.constexpr,
):
    req = tl.program_id(0)
    col = tl.program_id(1)
    if col >= tl.load(num_pages + req):
        return
    page = tl.load(block_table + req * block_table_stride + col).to(tl.int64)
    dst_page = (tl.load(page_offsets + req) + col).to(tl.int64)

    src = kv_cache + page * kv_page_stride
    dst = out + dst_page * (PAGE_SIZE * HEAD_DIM)
    tokens = tl.arange(0, PAGE_SIZE)[:, None]
    scales = src + PAGE_SIZE * (HEAD_DIM // 2) + tokens * (HEAD_DIM // 16)
    if USE_QMUL4:
        # A 16-bit word holds four elements of one group; multiply it by the
        # group's scale byte broadcast to E4M3x4.
        words = tl.arange(0, HEAD_DIM // 4)[None, :]
        packed4 = tl.load(
            src.to(tl.pointer_type(tl.uint16)) + tokens * (HEAD_DIM // 4) + words
        )
        scale4 = tl.load(scales + words // 4).to(tl.uint32) * 0x01010101
        values4 = tl.inline_asm_elementwise(
            "{ .reg .b16 v; cvt.u16.u32 v, $1; "
            "mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 $0, v, $2; }",
            "=r,r,r",
            [packed4.to(tl.uint32), scale4],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        tl.store(
            dst.to(tl.pointer_type(tl.uint32)) + tokens * (HEAD_DIM // 4) + words,
            values4,
        )
        return

    packed_cols = tl.arange(0, HEAD_DIM // 2)[None, :]
    packed = tl.load(src + tokens * (HEAD_DIM // 2) + packed_cols)
    # Packed byte b holds elements 2b and 2b+1, both in group b // 8.
    scale = tl.load(scales + packed_cols // 8)
    scale = scale.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    # e2m1 * e4m3 is exact in fp32; clamp, then round once to e4m3.
    low = _e2m1_to_f32(packed & 0xF) * scale
    high = _e2m1_to_f32(packed >> 4) * scale
    values = tl.interleave(low, high)
    values = tl.minimum(tl.maximum(values, -E4M3_MAX), E4M3_MAX)
    tl.store(
        dst + tokens * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :],
        values.to(tl.float8e4nv),
    )


def minimax_m3_gather_nvfp4_index_k_fp8(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    num_pages: torch.Tensor,
    page_offsets: torch.Tensor,
    max_num_pages: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Dequantize each request's leading NVFP4 pages into contiguous FP8 pages.

    Args:
        kv_cache: ``[num_blocks, 128, 72]`` uint8 packed NVFP4 pages.
        block_table: ``[num_reqs, max_blocks]`` int32 physical page ids.
        num_pages: ``[num_reqs]`` int32 pages to gather per request (its first
            ``num_pages[r]`` logical pages).
        page_offsets: ``[num_reqs]`` int32 first output page of each request.
        max_num_pages: Host upper bound of ``num_pages``.
        out: ``[>= sum(num_pages), 128, 128]`` contiguous float8_e4m3fn.

    Returns:
        ``out``.

    """
    assert kv_cache.dtype == torch.uint8 and kv_cache.shape[1:] == (
        PAGE_SIZE,
        HEAD_DIM // 2 + HEAD_DIM // 16,
    )
    assert kv_cache.stride(1) == kv_cache.shape[2] and kv_cache.stride(2) == 1
    assert out.dtype == torch.float8_e4m3fn and out.is_contiguous()
    assert out.shape[1:] == (PAGE_SIZE, HEAD_DIM)
    assert block_table.stride(1) == 1
    num_reqs = block_table.shape[0]
    if num_reqs == 0 or max_num_pages == 0:
        return out
    _gather_nvfp4_index_k_fp8_kernel[(num_reqs, max_num_pages)](
        kv_cache,
        block_table,
        num_pages,
        page_offsets,
        out,
        kv_cache.stride(0),
        block_table.stride(0),
        PAGE_SIZE=PAGE_SIZE,
        HEAD_DIM=HEAD_DIM,
        E4M3_MAX=E4M3_MAX,
        USE_QMUL4=_qmul4_supported(kv_cache.device.index or 0),
        num_warps=4,
    )
    return out
