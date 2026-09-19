# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER fused MLA decode Q-prep for Kimi-K3 on ROCm.

On a stock build, Kimi-K3 MLA decode runs two kernels per layer per step:

1. ``vllm::concat_and_cache_mla_kernel`` -- the fp8 KV-cache write
2. ``triton_poi_fused__to_copy_cat_clamp_mul_reciprocal_view`` -- the
   ``[ql_nope | q_pe]`` concat and that query's static fp8 quant, emitted as one
   kernel from the ``mla_decode_concat_quant_fp8`` custom op

(That second step is a single op, but how it lowers depends on the serving
configuration -- under some ``--compilation-config`` settings it appears instead
as a separate concat and quant. Either way this replaces all of it.)

``aiter.ops.cache.fused_qk_rope_concat_and_cache_mla`` does the whole thing (plus
a RoPE) in one launch. K3 MLA is NoPE, so the rotation is made a no-op by feeding
the kernel ``cos = 1, sin = 0`` -- the same trick ROCm/ATOM uses to route K3
through this kernel.

The identity cos/sin cache is a SINGLE ROW. AITER's per-head and ``_opt`` decode
kernels clamp ``pos`` into ``[0, cos_cache.size(0))`` before indexing, so real
positions pass straight through; a full-length constant cache would otherwise
cost ~268 MB at K3's ``max_position_embeddings`` of 1048576 to express a
provable no-op. Its *general* decode kernel does NOT clamp, so
:func:`supports_fused_qk_prep` refuses the head counts that would select it.
"""

import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

# AITER dispatch constants, mirrored from csrc/kernels/cache_kernels.cu. The
# ``_opt`` decode kernel is chosen when all three hold, and it clamps `pos`;
# below MIN_SIZE_FOR_OPT the unclamped general kernel is selected instead.
_OPT_KV_LORA_RANK = 512
_OPT_ROT_DIM = 64
_OPT_MIN_SIZE = 2048


def supports_fused_qk_prep(
    kv_lora_rank: int, qk_rope_head_dim: int, num_heads: int, kv_cache_dtype: str
) -> bool:
    """Whether the fused kernel is safe for this layer's shapes.

    ``num_heads`` is the per-rank count. At TP=8 K3 has 12 heads/rank, so
    ``512 * 12 = 6144`` clears MIN_SIZE_FOR_OPT; at TP=32 it would be 3 heads
    and fall through to the unclamped general kernel, which the single-row
    identity cache cannot survive.
    """
    return (
        current_platform.is_rocm()
        and kv_cache_dtype.startswith("fp8")
        and kv_cache_dtype != "fp8_ds_mla"
        and kv_lora_rank == _OPT_KV_LORA_RANK
        and qk_rope_head_dim == _OPT_ROT_DIM
        and kv_lora_rank * num_heads >= _OPT_MIN_SIZE
    )


def make_identity_rope_cache(
    qk_rope_head_dim: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """``cos = 1, sin = 0`` -- one row, which the kernel's clamp makes enough."""
    half = qk_rope_head_dim // 2
    cos = torch.ones(1, half, dtype=dtype, device=device)
    sin = torch.zeros(1, half, dtype=dtype, device=device)
    return cos, sin


def _fused_mla_qk_prep_impl(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    q_out: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    q_scale: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
) -> None:
    from aiter.ops.cache import fused_qk_rope_concat_and_cache_mla

    fused_qk_rope_concat_and_cache_mla(
        ql_nope,
        q_pe,
        kv_c_normed,
        k_pe,
        kv_cache,
        q_out,
        slot_mapping,
        k_scale,
        q_scale,
        positions,
        cos_cache,
        sin_cache,
        is_neox=True,
        is_nope_first=True,
    )


def _fused_mla_qk_prep_fake(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    q_out: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    q_scale: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
) -> None:
    return


direct_register_custom_op(
    op_name="kimi_k3_fused_mla_qk_prep",
    op_func=_fused_mla_qk_prep_impl,
    mutates_args=["kv_cache", "q_out"],
    fake_impl=_fused_mla_qk_prep_fake,
)


def fused_mla_qk_prep(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    q_scale: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    head_size: int,
    q_out_dtype: torch.dtype,
) -> torch.Tensor:
    """Write the KV cache and return the assembled (fp8) decode query."""
    num_tokens, num_heads = ql_nope.shape[:2]
    # The kernel reads ql_nope densely; the W_UK bmm hands us a transposed view.
    ql_nope = ql_nope.contiguous()
    q_out = torch.empty(
        (num_tokens, num_heads, head_size),
        dtype=q_out_dtype,
        device=ql_nope.device,
    )
    torch.ops.vllm.kimi_k3_fused_mla_qk_prep(
        ql_nope,
        q_pe,
        kv_c_normed,
        k_pe,
        kv_cache.view(kv_cache.shape[0], -1, head_size),
        q_out,
        slot_mapping,
        k_scale,
        q_scale,
        positions,
        cos_cache,
        sin_cache,
    )
    return q_out
