# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU FA2 int4-KV bring-up helpers (greenfield).

Default is off: Triton remains the int4 backend until ``VLLM_XPU_FA_INT4=1``
and ``vllm_xpu_kernels.flash_attn_interface.FA_INT4_AVAILABLE`` is true.
"""

from __future__ import annotations

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import get_dtype_size

logger = init_logger(__name__)


def fa_int4_kernels_available() -> bool:
    """True when the installed xpu-kernels wheel exports FA int4 support."""
    try:
        from vllm_xpu_kernels.flash_attn_interface import FA_INT4_AVAILABLE

        return bool(FA_INT4_AVAILABLE)
    except Exception:
        return False


def fa_int4_enabled() -> bool:
    """Route int4 → FLASH_ATTN only when env + kernels agree."""
    if not envs.VLLM_XPU_FA_INT4:
        return False
    return fa_int4_kernels_available()


def ensure_int4_scale_caches(
    kv_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Carve packed K/V halves + per-token-head fp32 scale views.

    Expects logical packed shape ``(num_blocks, nkv, block_size, C)`` with
    ``C = 2 * (hs + pad)`` (136 for D=128). Mirrors Triton
    ``_ensure_scale_caches`` — do **not** ``split(head_size)`` on C=136.
    """
    num_blocks, nkv, block_size, content = kv_cache.shape
    dtype_sz = kv_cache.element_size()
    scale_pad = get_dtype_size(torch.float32) // dtype_sz
    padded_hs = content // 2
    hs = padded_hs - scale_pad

    # Packed data halves (uint8 / storage dtype).
    key_cache = kv_cache[..., :hs]
    value_cache = kv_cache[..., padded_hs : padded_hs + hs]

    raw = kv_cache.untyped_storage()
    base_f32 = torch.tensor([], dtype=torch.float32, device=kv_cache.device).set_(raw)

    def to_f32_units(elements: int) -> int:
        nbytes = elements * dtype_sz
        assert nbytes % 4 == 0
        return nbytes // 4

    strides = kv_cache.stride()
    block_f32 = to_f32_units(strides[0])
    head_f32 = to_f32_units(strides[1])
    slot_f32 = to_f32_units(strides[2])
    base_off_f32 = to_f32_units(kv_cache.storage_offset())
    k_scale_off_f32 = base_off_f32 + to_f32_units(hs)
    v_scale_off_f32 = base_off_f32 + to_f32_units(padded_hs + hs)

    k_scale = torch.as_strided(
        base_f32,
        size=(num_blocks, block_size, nkv),
        stride=(block_f32, slot_f32, head_f32),
        storage_offset=k_scale_off_f32,
    )
    v_scale = torch.as_strided(
        base_f32,
        size=(num_blocks, block_size, nkv),
        stride=(block_f32, slot_f32, head_f32),
        storage_offset=v_scale_off_f32,
    )
    return key_cache, value_cache, k_scale, v_scale
