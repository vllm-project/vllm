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


def prepare_fa_int4_query(
    query: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, float, float, torch.dtype]:
    """Match Triton ``unified_attention_int4``: RHT(Q) and scale /= rht_norm.

    KV pages are written with ``reshape_and_cache_int4`` which applies the
    same forward RHT before INT4 pack. FA decode dequants in that rotated
    basis, so Q must be rotated too (and the output inverse-rotated).
    """
    from vllm.utils.math_utils import largest_power_of_2_divisor
    from vllm.v1.attention.ops.int4_per_token_head import single_rht

    q_dtype = query.dtype
    q_rot = single_rht(query.float()).to(q_dtype)
    rht_norm = float(largest_power_of_2_divisor(query.shape[-1]))
    return q_rot, softmax_scale / rht_norm, rht_norm, q_dtype


def finalize_fa_int4_output(
    output: torch.Tensor,
    rht_norm: float,
    q_dtype: torch.dtype,
) -> None:
    """Inverse RHT on attention output (Triton ``unified_attention_int4``)."""
    from vllm.v1.attention.ops.int4_per_token_head import single_rht

    out_f = single_rht(output.float(), inverse=True) / rht_norm
    output.copy_(out_f.to(q_dtype))


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


def run_int4_triton_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    block_table: torch.Tensor,
    softmax_scale: float,
    causal,
    window_size,
    softcap: float = 0.0,
    sinks: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
    mm_prefix_range: torch.Tensor | None = None,
    rswa_prefix_lens: torch.Tensor | None = None,
    rswa_window: int | None = None,
) -> None:
    """Run Triton int4 attention on C=136 pages (prefill / q_len>1).

    M1 FA int4 kernels are decode-only (``max_seqlen_q==1``). When the FA
    backend is selected via ``VLLM_XPU_FA_INT4=1``, prefill must reuse the
    production Triton int4 path on the same packed KV.
    """
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode

    _, _, k_scale_cache, v_scale_cache = ensure_int4_scale_caches(kv_cache)
    # Triton int4 expects padded halves (data + scale pad), not packed-only.
    padded_hs = kv_cache.shape[-1] // 2
    key_cache, value_cache = kv_cache.transpose(1, 2).split(padded_hs, dim=-1)

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        alibi_slopes=alibi_slopes,
        window_size=window_size,
        block_table=block_table,
        softcap=softcap,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        sinks=sinks,
        output_scale=output_scale,
        mm_prefix_range=mm_prefix_range,
        rswa_prefix_lens=rswa_prefix_lens,
        rswa_window=rswa_window,
        kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
    )
