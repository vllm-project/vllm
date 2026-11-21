# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused launches of the Muse Spark DSpark draft's context K/V on ROCm.

`ContextNorms` (`VLLM_MUSE_DSPARK_FUSED_CTX_NORMS`): `hidden_norm` and the
per-layer pre-norms in one kernel that writes the stacked bmm input, instead of
one `fused_rmsnorm_fwd` launch a norm and a `torch.stack`. Each norm keeps
`_rmsnorm_fwd_kernel`'s single-tile math, so the output is bitwise the same.

`context_kv_insert` (`VLLM_MUSE_DSPARK_FUSED_CTX_KV`): per layer, one
`fused_muse_spark_qknorm_rope_kv_insert` launch on the projected `[k | v]` rows
(no query heads) instead of the torch K-norm, RoPE and KV cache update. K is
rounded once after norm and rotation (twice on the torch path), so K cache
bytes can differ in the last bit.
"""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.utils import refill_derived
from vllm.models.muse_spark.common.ops.fused_qk_norm_rope_kv import (
    supports_fused_kv_insert,
)
from vllm.triton_utils import tl, triton

_NORM_TILE = 8192
_NORM_WARPS = 8  # fused_rmsnorm_fwd's non-residual single-tile launch


@triton.jit
def _context_norms_kernel(
    x_ptr,
    out_ptr,
    w0_ptr,
    w_ptr,
    gc_ptr,
    T,
    eps0,
    eps1,
    D: tl.constexpr,
    L: tl.constexpr,
    bD: tl.constexpr,
):
    t = tl.program_id(0) + tl.arange(0, 1)
    tm = t < T
    d = tl.arange(0, bD)
    dm = d < D
    idx = t[:, None] * D + d[None, :]
    mask = tm[:, None] & dm[None, :]

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    w0 = tl.load(w0_ptr + d, mask=dm, other=0.0).to(tl.float32) + tl.load(gc_ptr)
    sum_sq = tl.sum(x * x, axis=1)
    y = x * tl.rsqrt(sum_sq * (1.0 / D) + eps0)[:, None]
    y = y * w0[None, :]
    # hidden_norm's output, rounded as its store rounds it
    x = y.to(out_ptr.dtype.element_ty).to(tl.float32)
    for layer in tl.static_range(L):
        w = tl.load(w_ptr + layer * D + d, mask=dm, other=0.0).to(tl.float32)
        w = w + tl.load(gc_ptr + 1 + layer)
        sum_sq = tl.sum(x * x, axis=1)
        z = x * tl.rsqrt(sum_sq * (1.0 / D) + eps1)[:, None]
        z = z * w[None, :]
        tl.store(
            out_ptr + layer * T * D + idx, z.to(out_ptr.dtype.element_ty), mask=mask
        )


class ContextNorms:
    """`stack([norm(hidden_norm(x)) for norm in pre_norms])` in one launch."""

    def __init__(self, hidden_norm, pre_norms) -> None:
        self.hidden_norm = hidden_norm
        self.pre_norms = tuple(pre_norms)
        self.weights: torch.Tensor
        self.gain_centers: torch.Tensor
        self.refresh()

    def refresh(self) -> None:
        """Rederive the stacked weights after a weight (re)load, in place."""
        weights = torch.stack([n.weight for n in self.pre_norms], dim=0)
        refill_derived(self, "weights", weights.contiguous())
        refill_derived(
            self,
            "gain_centers",
            torch.tensor(
                [n.fused_gain_center for n in (self.hidden_norm, *self.pre_norms)],
                dtype=torch.float32,
                device=weights.device,
            ),
        )

    def supports(self, x: torch.Tensor) -> bool:
        norms = (self.hidden_norm, *self.pre_norms)
        return (
            x.dim() == 2
            and x.is_contiguous()
            and x.shape[-1] <= _NORM_TILE
            and all(n.supports_fused_kernel(x) and n.has_weight for n in norms)
            and not any(n.applies_weight_before_norm for n in norms)
            and len({n.variance_epsilon for n in self.pre_norms}) == 1
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        T, D = x.shape
        out = torch.empty(len(self.pre_norms), T, D, dtype=x.dtype, device=x.device)
        _context_norms_kernel[(T,)](
            x,
            out,
            self.hidden_norm.weight,
            self.weights,
            self.gain_centers,
            T,
            self.hidden_norm.variance_epsilon,
            self.pre_norms[0].variance_epsilon,
            D=D,
            L=len(self.pre_norms),
            bD=triton.next_power_of_2(D),
            num_warps=_NORM_WARPS,
        )
        return out


def context_kv_insert_supported(model) -> bool:
    """Whether every draft attention layer's backend and cache dtype take the
    fused insert (load time; `context_kv_insert` checks the bound caches)."""
    rope = model._rotary_emb
    return (
        model._head_dim in (64, 128, 256)
        and (rope is None or not rope.is_neox_style)
        and all(supports_fused_kv_insert(attn) for attn in model._attn_layers)
    )


def _packed_cache(model, kv_cache: torch.Tensor) -> bool:
    return (
        kv_cache.dim() == 4
        and kv_cache.shape[1] == model._num_kv_heads
        and kv_cache.shape[3] == 2 * model._head_dim
    )


def context_kv_insert(
    model,
    all_kv_flat: torch.Tensor,  # [L, T, 2 * kv_size], rewritten in place
    positions: torch.Tensor,
    context_slot_mapping: torch.Tensor | list[torch.Tensor | None],
) -> bool:
    """K-norm, RoPE and the KV cache write, one launch a layer.

    Returns:
        False, having written nothing, when a bound cache is not the packed
        `[blocks, kv_heads, block_size, 2 * head_dim]` layout.

    """
    if not all(_packed_cache(model, attn.kv_cache) for attn in model._attn_layers):
        return False
    rope = model._rotary_emb
    # The torch path's RoPE casts the shared cos/sin cache to the K dtype and
    # keeps it; do the same so the draft layers read the same table.
    cos_sin_cache = (
        None if rope is None else rope._match_cos_sin_cache_dtype(all_kv_flat)
    )
    per_layer = isinstance(context_slot_mapping, (list, tuple))
    for layer_idx, attn in enumerate(model._attn_layers):
        slots = context_slot_mapping[layer_idx] if per_layer else context_slot_mapping
        if slots is None:
            continue
        kv_cache = attn.kv_cache
        is_fp8 = attn.kv_cache_dtype not in ("auto", "bfloat16", "float16")
        ops.fused_muse_spark_qknorm_rope_kv_insert(
            all_kv_flat[layer_idx],
            positions,
            0,
            model._num_kv_heads,
            model._head_dim,
            model._rms_norm_eps,
            cos_sin_cache=cos_sin_cache,
            rotary_dim=0 if rope is None else rope.rotary_dim,
            slot_mapping=slots,
            kv_cache=kv_cache,
            block_size=kv_cache.shape[2],
            kv_cache_dtype=attn.kv_cache_dtype,
            k_scale=attn._k_scale if is_fp8 else None,
            v_scale=attn._v_scale if is_fp8 else None,
        )
    return True
