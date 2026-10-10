# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only BerryLM-OS model compatible with HuggingFace weights.

BerryLM is a text-only hybrid MoE decoder. Every macro-block of four layers holds
three linear-attention layers and one full-attention layer; every layer has a
sparse MoE (routed + shared expert). Two additions distinguish it from the stock
gated-delta-net hybrids in this directory:

* **KDA forget gate** on the linear-attention layers: the per-head scalar decay
  of the gated delta net is replaced by a per-channel low-rank gate
  ``g = -exp(A_log)[h] * softplus(f_up(f_down(x))[h, k] + dt_bias[h, k])`` —
  the Kimi Delta Attention recurrence on the gated-delta-net projection layout
  (``in_proj_qkv`` / ``in_proj_z`` / ``in_proj_b`` / depthwise ``conv1d``,
  SiLU-gated output RMSNorm). Kernels: in-tree ``fla.ops.kda``.
* **Gated Block AttnRes** on the residual stream: before layer *i* the stream
  ``x`` becomes ``x + tanh(gate_i) * (mix_i - x)`` where ``mix_i`` is a softmax
  mixture (over depth) of the residual streams committed at block boundaries
  and ``x`` itself. Token-local, so it does not touch any cache.

The full-attention block, the MoE block and the decoder-layer forward are the
stock hybrid-MoE building blocks, kept in this file; the linear-attention layer,
the mixer and the model loop are BerryLM-specific.
"""

from collections.abc import Iterable, Sequence
from itertools import islice
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    divide,
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.utils import resolve_layer_fused_shared_expert
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.model_executor.layers.layernorm import GemmaRMSNorm as BerryLMRMSNorm
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    mamba_v2_sharded_weight_loader,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.config_utils import (
    get_quark_ocp_mx_group_size,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsLoRA,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.third_party.flash_linear_attention.ops.kda import (
    chunk_kda_with_fused_gate,
    fused_kda_gate,
    fused_recurrent_kda,
    fused_recurrent_kda_fwd,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

if TYPE_CHECKING:
    from vllm.transformers_utils.configs.berrylm import BerryLMConfig

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Custom ops
# ---------------------------------------------------------------------------


def berrylm_kda_attention(
    mixed_qkv: torch.Tensor,
    raw_g: torch.Tensor,
    b: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward(
        mixed_qkv=mixed_qkv,
        raw_g=raw_g,
        b=b,
        core_attn_out=core_attn_out,
    )


def berrylm_kda_attention_fake(
    mixed_qkv: torch.Tensor,
    raw_g: torch.Tensor,
    b: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    return


# ---------------------------------------------------------------------------
# Fused layer entry: residual add -> block commit -> Gated Block AttnRes mixer ->
# input RMSNorm (Triton, forward only)
#
# Before every layer the stream ``x = hidden + residual`` becomes
# ``x + tanh(gate) * (mix - x)``: ``mix`` is a softmax mixture over depth of the
# residual streams committed at block boundaries and ``x`` itself (keys =
# rms_norm(source) in fp32 without affine, logit = <key, pseudo_query>, weights =
# softmax over the sources, mixture over the raw sources). The mixed stream is the
# layer's residual and its Gemma RMSNorm the layer's input. One program per token
# does the add (rounded like the torch add), writes the commit when the layer opens
# a block, reads the earlier commits through separate pointers (no stacking), mixes
# in the reference's summation order (commits first, the current stream last), blends
# with the same op-by-op rounding in the stream dtype as the torch chain and applies
# the norm. The torch chain is the fallback off-GPU.
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    HAS_TRITON = False

_ATTNRES_MAX_SOURCES = 8

if HAS_TRITON:

    @triton.jit
    def _attnres_source_row(
        i: tl.constexpr, s0, s1, s2, s3, s4, s5, s6, s7, row, offs, mask
    ):
        if i == 0:
            p = s0
        elif i == 1:
            p = s1
        elif i == 2:
            p = s2
        elif i == 3:
            p = s3
        elif i == 4:
            p = s4
        elif i == 5:
            p = s5
        elif i == 6:
            p = s6
        else:
            p = s7
        return tl.load(p + row + offs, mask=mask, other=0.0).to(tl.float32)

    @triton.jit
    def _attnres_norm_kernel(
        hidden_ptr,
        residual_ptr,
        s0,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        s7,
        query_ptr,
        gate_ptr,
        norm_w_ptr,
        normed_ptr,
        mixed_ptr,
        committed_ptr,
        D,
        stride_h,
        stride_r,
        stride_s,  # committed sources, contiguous [T, D] each
        stride_o,  # normed / mixed / committed
        eps,
        norm_eps,
        N_PREV: tl.constexpr,
        COMMIT: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        HAS_GATE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        t = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        dt: tl.constexpr = mixed_ptr.dtype.element_ty
        q = tl.load(query_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        inv_d = 1.0 / D
        row_s = t * stride_s

        x_cur = tl.load(hidden_ptr + t * stride_h + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        if HAS_RESIDUAL:
            r = tl.load(residual_ptr + t * stride_r + offs, mask=mask, other=0.0)
            x_cur = (x_cur + r.to(tl.float32)).to(dt).to(tl.float32)
        if COMMIT:
            tl.store(committed_ptr + t * stride_o + offs, x_cur.to(dt), mask=mask)

        # pass 1: logits and their running max
        rstd_cur = tl.rsqrt(tl.sum(x_cur * x_cur, axis=0) * inv_d + eps)
        logit_cur = tl.sum(x_cur * q, axis=0) * rstd_cur
        m = logit_cur
        for i in tl.static_range(N_PREV):
            x = _attnres_source_row(
                i, s0, s1, s2, s3, s4, s5, s6, s7, row_s, offs, mask
            )
            rstd = tl.rsqrt(tl.sum(x * x, axis=0) * inv_d + eps)
            m = tl.maximum(m, tl.sum(x * q, axis=0) * rstd)
        # (the commit's logit is logit_cur: already in the max)

        # pass 2: softmax denominator in source order (recomputing a logit is cheaper
        # than keeping n rows live)
        w_cur = tl.exp(logit_cur - m)
        denom = w_cur
        for i in tl.static_range(N_PREV):
            x = _attnres_source_row(
                i, s0, s1, s2, s3, s4, s5, s6, s7, row_s, offs, mask
            )
            rstd = tl.rsqrt(tl.sum(x * x, axis=0) * inv_d + eps)
            denom += tl.exp(tl.sum(x * q, axis=0) * rstd - m)
        if COMMIT:
            denom += tl.exp(logit_cur - m)

        # pass 3: mixture, committed sources first, the current stream last
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for i in tl.static_range(N_PREV):
            x = _attnres_source_row(
                i, s0, s1, s2, s3, s4, s5, s6, s7, row_s, offs, mask
            )
            rstd = tl.rsqrt(tl.sum(x * x, axis=0) * inv_d + eps)
            acc += (tl.exp(tl.sum(x * q, axis=0) * rstd - m) / denom) * x
        if COMMIT:
            acc += (tl.exp(logit_cur - m) / denom) * x_cur
        mixed = acc + (w_cur / denom) * x_cur

        if HAS_GATE:
            # The reference blends in the stream dtype op by op (`stream + tanh(gate) *
            # (mixed - stream)` on bf16 tensors rounds after every op); mirror that
            # rounding sequence so the results match ulp for ulp.
            g = tl.load(gate_ptr).to(tl.float32)
            scale = libdevice.tanh(g).to(dt).to(tl.float32)
            mixed_r = mixed.to(dt).to(tl.float32)
            diff = (mixed_r - x_cur).to(dt).to(tl.float32)
            prod = (scale * diff).to(dt).to(tl.float32)
            out = (x_cur + prod).to(dt)
        else:
            out = mixed.to(dt)
        tl.store(mixed_ptr + t * stride_o + offs, out, mask=mask)

        # Gemma RMSNorm of the mixed stream: (x * rsqrt(mean(x^2) + eps)) * (1 + w)
        y = out.to(tl.float32)
        y = y * tl.rsqrt(tl.sum(y * y, axis=0) * inv_d + norm_eps)
        w = tl.load(norm_w_ptr + offs, mask=mask, other=0.0).to(tl.float32) + 1.0
        tl.store(normed_ptr + t * stride_o + offs, (y * w).to(dt), mask=mask)


def _attnres_norm_torch(
    hidden: torch.Tensor,
    residual: torch.Tensor | None,
    completed: Sequence[torch.Tensor],
    pseudo_query: torch.Tensor,
    gate: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    norm_eps: float,
    commit: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The torch chain of the layer entry (fallback off-GPU)."""
    stream = hidden if residual is None else hidden + residual
    sources = [*completed, stream] if commit else list(completed)
    stacked = torch.stack([*sources, stream], dim=0).float()
    keys = F.rms_norm(stacked, (stream.shape[-1],), None, eps)
    logits = torch.einsum("ntd,d->nt", keys, pseudo_query.float())
    mixed = (logits.softmax(dim=0).unsqueeze(-1) * stacked).sum(dim=0).to(stream.dtype)
    if gate is not None:
        scale = torch.tanh(gate.float()).to(stream.dtype)
        mixed = stream + scale * (mixed - stream)
    x = mixed.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + norm_eps)
    normed = (x * (norm_weight.float() + 1.0)).to(stream.dtype)
    # a custom op must not return an alias of its input (first layer: stream is hidden)
    committed = (
        (stream if residual is not None else stream.clone())
        if commit
        else stream.new_empty((0, stream.shape[-1]))
    )
    return normed, mixed, committed


def berrylm_attnres_norm(
    pseudo_query: torch.Tensor,
    hidden: torch.Tensor,
    residual: torch.Tensor | None,
    completed: list[torch.Tensor],
    gate: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    norm_eps: float,
    commit: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Layer entry on vLLM's flat ``[num_tokens, hidden]`` layout. Returns
    ``(normed, mixed, committed)``: the layer input, the layer residual and, when
    ``commit`` (the layer opens a block), the stream ``hidden + residual`` to append
    to ``completed`` (an empty ``[0, hidden]`` tensor otherwise)."""
    T, D = hidden.shape
    if (
        not HAS_TRITON
        or hidden.device.type not in ("cuda", "xpu")
        or D > 8192
        or T == 0
        or len(completed) > _ATTNRES_MAX_SOURCES
    ):
        return _attnres_norm_torch(
            hidden,
            residual,
            completed,
            pseudo_query,
            gate,
            norm_weight,
            eps,
            norm_eps,
            commit,
        )
    srcs = [c.contiguous() for c in completed]
    hidden = hidden.contiguous()
    residual = residual.contiguous() if residual is not None else None
    mixed = torch.empty_like(hidden)
    normed = torch.empty_like(hidden)
    committed = torch.empty_like(hidden) if commit else hidden.new_empty((0, D))
    pq = pseudo_query if pseudo_query.dtype == torch.float32 else pseudo_query.float()
    ptrs = srcs + [hidden] * (_ATTNRES_MAX_SOURCES - len(srcs))  # unused slots
    _attnres_norm_kernel[(T,)](
        hidden,
        residual if residual is not None else hidden,
        *ptrs,
        pq,
        gate if gate is not None else pq,  # any pointer when unused
        norm_weight,
        normed,
        mixed,
        committed if commit else mixed,
        D,
        hidden.stride(0),
        residual.stride(0) if residual is not None else 0,
        D,
        D,
        eps,
        norm_eps,
        N_PREV=len(srcs),
        COMMIT=commit,
        HAS_RESIDUAL=residual is not None,
        HAS_GATE=gate is not None,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=8 if D > 1024 else 4,
    )
    return normed, mixed, committed


def berrylm_attnres_norm_fake(
    pseudo_query: torch.Tensor,
    hidden: torch.Tensor,
    residual: torch.Tensor | None,
    completed: list[torch.Tensor],
    gate: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    norm_eps: float,
    commit: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    committed = (
        torch.empty_like(hidden) if commit else hidden.new_empty((0, hidden.shape[-1]))
    )
    return torch.empty_like(hidden), torch.empty_like(hidden), committed


# ---------------------------------------------------------------------------
# Packed one-token KDA decode (Triton, forward only)
#
# A pure-decode batch reads q / k / v straight from the post-conv packed
# projection (grouped value heads: value head ``hv`` uses key head
# ``hv // (HV // H)``, no q/k copies), computes the per-channel decay
# ``-exp(A_log) * softplus(raw_g + dt_bias)`` and ``beta = sigmoid(b)`` in
# registers and updates the fp32 state in place: one launch instead of the
# split / repeat / gate / sigmoid / recurrence / copy chain. The arithmetic is the
# one of ``fused_kda_gate`` + ``fused_recurrent_kda`` op by op.
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit(
        do_not_specialize=[
            "stride_qkv_tok",
            "stride_g_tok",
            "stride_b_tok",
            "stride_idx",
        ]
    )
    def _kda_packed_decode_kernel(
        mixed_qkv,
        raw_g,
        raw_b,
        A_log,
        dt_bias,
        out,
        state,
        state_indices,
        scale,
        stride_qkv_tok,
        stride_g_tok,
        stride_b_tok,
        stride_idx,
        stride_state: tl.constexpr,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        SOFTPLUS_THRESHOLD: tl.constexpr,
    ):
        i_v, i_nh = tl.program_id(0), tl.program_id(1)
        i_n, i_hv = i_nh // HV, i_nh % HV
        i_h = i_hv // (HV // H)

        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]

        p_o = out + (i_n * HV + i_hv) * V + o_v
        state_idx = tl.load(state_indices + i_n * stride_idx).to(tl.int64)
        # padded slots point at the null block: zeros out, cache untouched
        if state_idx <= 0:
            zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
            tl.store(p_o, zero, mask=mask_v)
            return

        p_h = state + state_idx * stride_state
        p_h = p_h + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)

        p_qkv = mixed_qkv + i_n * stride_qkv_tok
        b_q = tl.load(p_qkv + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_qkv + H * K + i_h * K + o_k, mask=mask_k, other=0)
        b_k = b_k.to(tl.float32)
        b_v = tl.load(p_qkv + 2 * H * K + i_hv * V + o_v, mask=mask_v, other=0)
        b_v = b_v.to(tl.float32)
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        b_g = tl.load(raw_g + i_n * stride_g_tok + i_hv * K + o_k, mask=mask_k, other=0)
        b_bias = tl.load(dt_bias + i_hv * K + o_k, mask=mask_k, other=0)
        b_g = b_g.to(tl.float32) + b_bias.to(tl.float32)
        b_a = tl.exp(tl.load(A_log + i_hv).to(tl.float32))
        b_sp = tl.where(b_g > SOFTPLUS_THRESHOLD, b_g, 1.0 * tl.log(1.0 + tl.exp(b_g)))
        b_gk = -b_a * b_sp
        b_beta = tl.sigmoid(tl.load(raw_b + i_n * stride_b_tok + i_hv).to(tl.float32))

        b_h *= tl.exp(b_gk[None, :])
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)


def kda_packed_decode(
    mixed_qkv: torch.Tensor,
    raw_g: torch.Tensor,
    raw_b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    num_k_heads: int,
    out: torch.Tensor,
) -> None:
    """One decode token per sequence. ``mixed_qkv`` [B, 2*H*K + HV*V] post-conv,
    ``raw_g`` [B, HV*K], ``raw_b`` [B, HV] (row strides free, last dim
    contiguous), ``state`` [cache, HV, V, K] updated in place at ``state_indices``
    [B]; writes ``out`` [B, HV, V]."""
    B = mixed_qkv.shape[0]
    HV, V, K = state.shape[-3:]
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    _kda_packed_decode_kernel[(triton.cdiv(V, BV), B * HV)](
        mixed_qkv,
        raw_g,
        raw_b,
        A_log,
        dt_bias,
        out,
        state,
        state_indices,
        K**-0.5,
        mixed_qkv.stride(0),
        raw_g.stride(0),
        raw_b.stride(0),
        state_indices.stride(0),
        stride_state=state.stride(0),
        H=num_k_heads,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        SOFTPLUS_THRESHOLD=20.0,
        num_warps=4,
        num_stages=2,
    )


# ---------------------------------------------------------------------------
# Shared-expert output gate (Triton, forward only)
#
# ``out * sigmoid(x @ w_gate)`` with a one-row gate weight: cuBLAS runs the
# [T, D] x [D, 1] product as a dot + reduce pair, then sigmoid and the multiply
# follow as two more kernels. One program per token does the dot in fp32 and
# rounds like the torch chain (logit, sigmoid and product each rounded to the
# activation dtype), in place on ``out``.
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _shared_expert_gate_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        D,
        N,
        stride_x,
        stride_o,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        t = tl.program_id(0).to(tl.int64)
        dt: tl.constexpr = out_ptr.dtype.element_ty
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_ptr + t * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        logit = tl.sum(x * w, axis=0).to(dt).to(tl.float32)
        g = tl.sigmoid(logit).to(dt).to(tl.float32)
        for n0 in range(0, N, BLOCK_N):
            o_offs = n0 + tl.arange(0, BLOCK_N)
            o_mask = o_offs < N
            p = out_ptr + t * stride_o + o_offs
            o = tl.load(p, mask=o_mask, other=0.0).to(tl.float32)
            tl.store(p, (g * o).to(dt), mask=o_mask)


def shared_expert_gate_(
    x: torch.Tensor, gate_weight: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """``out *= sigmoid(x @ gate_weight.T)`` in place; ``gate_weight`` [1, D]."""
    T, D = x.shape
    if (
        not HAS_TRITON
        or x.device.type not in ("cuda", "xpu")
        or D > 8192
        or T == 0
        or out.stride(-1) != 1
        or x.stride(-1) != 1
    ):
        out.mul_(F.sigmoid(F.linear(x, gate_weight)))
        return out
    _shared_expert_gate_kernel[(T,)](
        x,
        gate_weight,
        out,
        D,
        out.shape[-1],
        x.stride(0),
        out.stride(0),
        BLOCK_D=triton.next_power_of_2(D),
        BLOCK_N=1024,
        num_warps=4,
    )
    return out


# ``berrylm_kda_attention`` is a graph-splitting op (see
# ``CompilationConfig._attention_ops``): the recurrent state update must stay out
# of the compiled pieces, exactly like the other linear-attention core ops.
# ``berrylm_attnres_norm`` is deliberately *not* a splitting op: it is an opaque op
# wrapping the fused Triton kernel (one launch per layer), which inductor must not
# trace (the fused mixer kernel was observed to miscompile), but it stays inside the
# pieces and is captured/replayed with the surrounding CUDA graph.
if not hasattr(torch.ops.vllm, "berrylm_kda_attention"):
    direct_register_custom_op(
        op_name="berrylm_kda_attention",
        op_func=berrylm_kda_attention,
        mutates_args=["core_attn_out"],
        fake_impl=berrylm_kda_attention_fake,
    )
if not hasattr(torch.ops.vllm, "berrylm_attnres_norm"):
    direct_register_custom_op(
        op_name="berrylm_attnres_norm",
        op_func=berrylm_attnres_norm,
        mutates_args=[],
        fake_impl=berrylm_attnres_norm_fake,
    )


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stock hybrid-MoE building blocks (dense shared-expert MLP, sparse MoE block,
# gated full attention). Kept in this file so the model has no dependency on
# another model file.
# ---------------------------------------------------------------------------


def _should_use_sequence_parallel(vllm_config: VllmConfig) -> bool:
    config = vllm_config.model_config.hf_text_config
    parallel_config = vllm_config.parallel_config
    return (
        parallel_config.use_sequence_parallel_moe
        and parallel_config.pipeline_parallel_size == 1
        and getattr(config, "num_experts", 0) > 0
        and not getattr(config, "mlp_only_layers", [])
        and getattr(config, "decoder_sparse_step", 1) == 1
    )


def _should_replicate_misaligned_shared_expert(
    intermediate_size: int,
    tp_size: int,
    group_size: int | None,
    enable_expert_parallel: bool,
    is_sequence_parallel: bool,
) -> bool:
    if intermediate_size <= 0 or group_size is None:
        return False

    partition_size, remainder = divmod(intermediate_size, tp_size)
    if remainder == 0 and partition_size % group_size == 0:
        return False

    if enable_expert_parallel or is_sequence_parallel:
        return True

    if remainder != 0:
        raise ValueError(
            f"Shared-expert intermediate size {intermediate_size} must be "
            f"divisible by tensor-parallel size {tp_size}."
        )

    raise ValueError(
        "The Quark OCP MX shared expert cannot be tensor-parallelized: "
        f"intermediate size {intermediate_size} with TP size {tp_size} "
        f"produces a partition of {partition_size}, which is not divisible by "
        f"the OCP MX group size {group_size}. Choose a compatible "
        "tensor-parallel size or enable expert parallelism."
    )


class BerryLMMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        expert_gate: torch.nn.Linear | None = None,
        is_sequence_parallel: bool = False,
        disable_tp: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        disable_tp = disable_tp or is_sequence_parallel
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=disable_tp,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=disable_tp,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.expert_gate = expert_gate

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)

        if self.expert_gate is not None:
            # the gate is built with quant_config=None: a plain [1, hidden] weight
            out = shared_expert_gate_(x, self.expert_gate.weight, out)

        return out


class BerryLMSparseMoeBlock(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        # Resolve shared-expert fusion first (when enabled, TP alignment constraint no
        # longer applies)
        self.is_fused_shared_expert_enabled = False
        if config.shared_expert_intermediate_size > 0:
            self.is_fused_shared_expert_enabled = resolve_layer_fused_shared_expert(
                quant_config,
                prefix,
                shared_expert_name="shared_expert",
            )

        if self.is_fused_shared_expert_enabled:
            self.replicate_shared_expert = False
        else:
            shared_expert_group_size = get_quark_ocp_mx_group_size(
                quant_config,
                f"{prefix}.shared_expert.down_proj",
            )
            self.replicate_shared_expert = _should_replicate_misaligned_shared_expert(
                config.shared_expert_intermediate_size,
                self.tp_size,
                shared_expert_group_size,
                parallel_config.enable_expert_parallel,
                self.is_sequence_parallel,
            )
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )

        if (
            self.is_fused_shared_expert_enabled
            or config.shared_expert_intermediate_size <= 0
        ):
            self.shared_expert = None
        else:
            self.shared_expert = BerryLMMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                is_sequence_parallel=self.is_sequence_parallel,
                disable_tp=self.replicate_shared_expert,
                prefix=f"{prefix}.shared_expert",
            )

        self.experts = FusedMoEFactory(
            shared_experts=(
                None if self.replicate_shared_expert else self.shared_expert
            ),
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=1 if self.shared_expert is None else None,
            fuse_shared_experts=self.is_fused_shared_expert_enabled,
            shared_expert_gate=self.shared_expert_gate
            if self.shared_expert is None
            else None,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        replicated_shared_output = (
            self.shared_expert(hidden_states)
            if self.replicate_shared_expert and self.shared_expert is not None
            else None
        )
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=hidden_states
        )
        if replicated_shared_output is not None:
            final_hidden_states += replicated_shared_output

        if self.is_sequence_parallel and not already_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(orig_shape)


class BerryLMAttention(nn.Module):
    def __init__(
        self,
        config: "BerryLMConfig",
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            dual_chunk_attention_config=self.dual_chunk_attention_config,
        )

        # Generation models leave config.is_causal unset (-> causal / DECODER);
        # encoder-style variants may set it to False through a VerifyAndUpdateConfig
        # handler.
        attn_type = (
            AttentionType.DECODER
            if getattr(config, "is_causal", True)
            else AttentionType.ENCODER_ONLY
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": self.dual_chunk_attention_config,
            }
            if self.dual_chunk_attention_config
            else {},
        )

        self.q_norm = BerryLMRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = BerryLMRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Fuse the gated split + QK-RMSNorm + (partial) NeoX RoPE + gate copy.
        # Text-only model: plain 1-D RoPE, positions are always 1-D.
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and (current_platform.is_cuda() or current_platform.is_xpu())
            and supports_dtype
        )

    def _project_qkv_gate(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return post-norm, post-RoPE (q, k, v) and the pre-sigmoid gate.

        Dispatches between the fused Triton kernel and the eager
        split + QK-RMSNorm + RoPE path. ``gate`` is ``None`` when output
        gating is disabled.
        """
        if self.use_fused_qk_norm_rope_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            q, k, gate = fused_qk_rmsnorm_rope_gate(
                q_gate,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rotary_emb.cos_sin_cache,
                positions,
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_emb.rotary_dim,
                norm_beta=1.0,
            )
            return q, k, v, gate

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v, gate

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        attn_output = self.attn(q, k, v)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_output)
        return output


class BerryLMAttnRes(nn.Module):
    """Per-layer Gated Block AttnRes mixer.

    ``pseudo_query`` [hidden] (zero == uniform mean over sources) and a scalar
    ``gate`` (zero == exact identity). Both are replicated across TP ranks: the
    mixed streams are identical on every rank because they are taken after the
    row-parallel all-reduces. Checkpoint keys: ``model.layers.{i}.attn_res.*``.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, gated: bool = True):
        super().__init__()
        self.eps = eps
        self.pseudo_query = nn.Parameter(torch.zeros(hidden_size))
        self.gate = nn.Parameter(torch.zeros(())) if gated else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        completed: Sequence[torch.Tensor],
        input_layernorm: BerryLMRMSNorm,
        commit: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mix ``hidden_states + residual`` and apply the layer's input norm; see
        ``berrylm_attnres_norm``. Returns ``(normed, mixed, committed)``."""
        return torch.ops.vllm.berrylm_attnres_norm(
            self.pseudo_query,
            hidden_states,
            residual,
            list(completed),
            self.gate,
            input_layernorm.weight,
            self.eps,
            input_layernorm.variance_epsilon,
            commit,
        )


@PluggableLayer.register("berrylm_kda_attention")
class BerryLMKDAAttention(GatedDeltaNetAttention):
    """Gated delta net with a per-channel KDA forget gate (stock GDN projection layout).

    Compared with the stock gated-delta-net layer: no ``in_proj_a`` / ``in_proj_ba``
    (``in_proj_qkv`` and ``in_proj_z`` share one GEMM), ``in_proj_b`` plus
    ``f_down_proj`` [bottleneck, hidden] (sharing a second, unquantized GEMM),
    ``f_up_proj`` [num_v_heads * head_k_dim, bottleneck] and a channel-wise
    ``dt_bias`` [num_v_heads * head_k_dim]. The recurrent state has the GDN shape,
    so the GDN cache/metadata machinery is reused unchanged; pure-decode batches run
    one packed kernel (``kda_packed_decode``).
    """

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def __init__(
        self,
        config: "BerryLMConfig",
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.gate_bottleneck = config.kda_gate_bottleneck
        if getattr(config, "kda_safe_gate", False):
            raise NotImplementedError(
                "BerryLM: the lower-bounded (safe) KDA gate is not supported; "
                "released checkpoints use kda_safe_gate=False."
            )
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"linear_num_value_heads ({self.num_v_heads}) must be a multiple "
                f"of linear_num_key_heads ({self.num_k_heads})"
            )
        self.group_size = self.num_v_heads // self.num_k_heads
        if self.num_k_heads % self.tp_size or self.num_v_heads % self.tp_size:
            raise ValueError(
                f"BerryLM linear attention needs num_k_heads ({self.num_k_heads}) "
                f"and num_v_heads ({self.num_v_heads}) divisible by "
                f"tensor_parallel_size ({self.tp_size})"
            )
        self.local_num_k_heads = divide(self.num_k_heads, self.tp_size)
        self.local_num_v_heads = divide(self.num_v_heads, self.tp_size)
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.qk_dim_local = self.local_num_k_heads * self.head_k_dim
        self.v_dim_local = self.local_num_v_heads * self.head_v_dim
        self.gate_dim_local = self.local_num_v_heads * self.head_k_dim

        # Checkpoint ships in_proj_qkv / in_proj_z / in_proj_b / f_down_proj
        # separately. q|k|v|z run as one GEMM (every partition is sharded by head, so
        # the merged layer's TP split is the natural one). The forget-gate down
        # projection (hidden -> bottleneck) and the beta projection (one row per value
        # head) run as a second, never-quantized GEMM ``in_proj_bf`` = [f_down; b]:
        # the decay gate is the precision-critical part of the recurrence, the beta
        # rows are smaller than a quantization block, and a merged layer takes a
        # single quantization scheme. It is replicated (the bottleneck feeds the
        # head-sharded up projection whole); each rank reads its own beta heads.
        self.in_proj_qkvz = MergedColumnParallelLinear(
            self.hidden_size,
            [self.key_dim, self.key_dim, self.value_dim, self.value_dim],
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvz",
        )
        self.in_proj_bf = ReplicatedLinear(
            self.hidden_size,
            self.gate_bottleneck + self.num_v_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.in_proj_bf",
        )
        self.in_proj_bf.weight.weight_loader = self._load_in_proj_bf
        self.beta_start = self.gate_bottleneck + self.tp_rank * self.local_num_v_heads
        # Low-rank forget gate up projection: bottleneck -> HV*K (sharded by value
        # head, matching dt_bias / A_log), never quantized either.
        self.f_up_proj = ColumnParallelLinear(
            self.gate_bottleneck,
            self.num_v_heads * self.head_k_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.f_up_proj",
        )
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)
        self.conv1d.weight.weight_loader = mamba_v2_sharded_weight_loader(
            [query_key_settings, query_key_settings, value_settings],
            self.tp_size,
            self.tp_rank,
        )
        self.A_log = nn.Parameter(
            torch.empty(self.local_num_v_heads, dtype=torch.float32)
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.gate_dim_local, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        output_gate_act = getattr(config, "kda_output_gate_act", "silu")
        if output_gate_act == "swish":
            output_gate_act = "silu"
        assert output_gate_act in ("silu", "sigmoid"), (
            f"unsupported kda_output_gate_act={output_gate_act}"
        )
        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            activation=output_gate_act,
            device=current_platform.current_device(),
        )
        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=self.quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self._prefill_kernels_warmed_up = False

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    # -- projections -------------------------------------------------------

    def forward(
        self, hidden_states: torch.Tensor, output: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        # vLLM main's decoder layer takes the return value; <= 0.25.1 passes a
        # preallocated ``output`` buffer.
        num_tokens = hidden_states.size(0)
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        qkv_size = (2 * self.key_dim + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        z = z.reshape(num_tokens, -1, self.head_v_dim)
        bf, _ = self.in_proj_bf(hidden_states)
        raw_g, _ = self.f_up_proj(bf[:, : self.gate_bottleneck])
        # beta logits of this rank's value heads; sigmoid inside the core
        b = bf[:, self.beta_start : self.beta_start + self.local_num_v_heads]

        core_attn_out = torch.zeros(
            (num_tokens, self.local_num_v_heads, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.berrylm_kda_attention(
            mixed_qkv, raw_g, b, core_attn_out, self.prefix
        )

        core_attn_out = self.norm(
            core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        )
        core_attn_out = core_attn_out.reshape(num_tokens, -1)
        out, _ = self.out_proj(core_attn_out)
        if output is None:
            return out
        output[:num_tokens] = out
        return None

    def _load_in_proj_bf(
        self, param: nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        """``f_down_proj`` (shard 0) -> rows [0, bottleneck), ``in_proj_b`` (shard 1)
        -> the num_v_heads rows after it; both unsharded (the layer is replicated)."""
        shard_id = getattr(loaded_weight, "shard_id", None)
        if shard_id == 0:
            rows = (0, self.gate_bottleneck)
        elif shard_id == 1:
            rows = (self.gate_bottleneck, self.gate_bottleneck + self.num_v_heads)
        else:
            raise ValueError(f"in_proj_bf: unexpected shard id {shard_id!r}")
        dst = param.data[rows[0] : rows[1]]
        assert dst.shape == loaded_weight.shape, (dst.shape, loaded_weight.shape)
        dst.copy_(loaded_weight)

    # -- helpers -----------------------------------------------------------

    def _split_qkv(
        self, mixed_qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = torch.split(
            mixed_qkv,
            [self.qk_dim_local, self.qk_dim_local, self.v_dim_local],
            dim=-1,
        )
        q = q.view(-1, self.local_num_k_heads, self.head_k_dim)
        k = k.view(-1, self.local_num_k_heads, self.head_k_dim)
        v = v.view(-1, self.local_num_v_heads, self.head_v_dim)
        if self.group_size > 1:
            # The KDA kernels do not broadcast H != HV: repeat q/k per value head.
            q = q.repeat_interleave(self.group_size, dim=1)
            k = k.repeat_interleave(self.group_size, dim=1)
        return q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)

    def _reshape_gate(
        self, raw_g: torch.Tensor, beta: torch.Tensor, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_g = rearrange(raw_g[:num_tokens], "n (h d) -> 1 n h d", d=self.head_k_dim)
        beta = beta[:num_tokens].unsqueeze(0)
        return raw_g, beta

    def _fused_gate(self, raw_g: torch.Tensor) -> torch.Tensor:
        """raw_g [1, n, HV, K] -> log-decay g [1, n, HV, K]."""
        g = fused_kda_gate(
            rearrange(raw_g, "1 n h d -> n (h d)"),
            self.A_log,
            self.head_k_dim,
            g_bias=self.dt_bias,
        )
        return g.unsqueeze(0)

    def _warmup_prefill_kernels(self, mixed_qkv: torch.Tensor) -> None:
        """Autotune the chunk KDA kernel during the profile run, while GPU memory
        is still plentiful (same reason as the GDN layer). The recurrent decode
        kernel is compiled by the CUDA-graph warmup iterations instead."""
        if self._prefill_kernels_warmed_up:
            return
        self._prefill_kernels_warmed_up = True
        if mixed_qkv is None or not mixed_qkv.is_cuda:
            return
        device, dtype, t = mixed_qkv.device, mixed_qkv.dtype, 64
        try:
            dummy_qkv = torch.randn(t, mixed_qkv.shape[-1], device=device, dtype=dtype)
            dummy_g = torch.randn(t, self.gate_dim_local, device=device, dtype=dtype)
            dummy_beta = torch.rand(
                t, self.local_num_v_heads, device=device, dtype=torch.float32
            )
            raw_g, beta = self._reshape_gate(dummy_g, dummy_beta, t)
            q, k, v = self._split_qkv(dummy_qkv)
            _, state_dtype = self.get_state_dtype()
            initial_state = torch.zeros(
                1,
                self.local_num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                device=device,
                dtype=state_dtype,
            )
            chunk_kda_with_fused_gate(
                q=q,
                k=k,
                v=v,
                raw_g=raw_g,
                beta=beta,
                A_log=self.A_log,
                g_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=torch.tensor([0, t], device=device, dtype=torch.int32),
            )
        except Exception:
            logger.warning(
                "BerryLM KDA chunk kernel warmup failed for layer %s; the first "
                "request may autotune under CUDA graph capture",
                self.prefix,
                exc_info=True,
            )

    # -- core --------------------------------------------------------------

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        # conv state in the cache dtype, recurrent KDA state in fp32 (KimiLinear /
        # training semantics); the GatedDeltaNetAttention default would take the
        # temporal state from --mamba-ssm-cache-dtype (auto = bf16).
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        raw_g: torch.Tensor,
        b: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if attn_metadata_raw is None:
            # V1 profile run.
            self._warmup_prefill_kernels(mixed_qkv)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw.get(self.prefix)
        if attn_metadata is None:
            # vLLM main: profile / warmup dummy runs may omit the mamba-family metadata
            # (glm5next does the same).
            self._warmup_prefill_kernels(mixed_qkv)
            return
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        conv_state, recurrent_state = self.kv_cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        conv_bias = getattr(self.conv1d, "bias", None)

        # 0. Pure decode (the steady state of generation): conv update, then one
        # packed kernel that reads q/k/v from the conv output, computes the gate and
        # beta in registers and writes the layer output in place.
        if (
            spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
            and non_spec_state_indices_tensor is not None
            and non_spec_state_indices_tensor.ndim == 1
        ):
            decode_state_indices = non_spec_state_indices_tensor[:num_actual_tokens]
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weights,
                conv_bias,
                self.activation,
                conv_state_indices=decode_state_indices,
                validate_data=False,
            )
            kda_packed_decode(
                mixed_qkv,
                raw_g[:num_actual_tokens],
                b[:num_actual_tokens],
                self.A_log,
                self.dt_bias,
                recurrent_state,
                decode_state_indices,
                self.local_num_k_heads,
                core_attn_out[:num_actual_tokens],
            )
            return

        beta = b[:num_actual_tokens].float().sigmoid()
        raw_g, beta = self._reshape_gate(raw_g, beta, num_actual_tokens)

        # 1. Split spec / non-spec tokens.
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                qkv_spec, g_spec, beta_spec = mixed_qkv, raw_g, beta
                qkv_non_spec = g_non_spec = beta_non_spec = None
            else:
                qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                g_spec = raw_g.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
                g_non_spec = raw_g.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
        else:
            qkv_spec = g_spec = beta_spec = None
            qkv_non_spec, g_non_spec, beta_non_spec = mixed_qkv, raw_g, beta

        # 2. Depthwise causal conv.
        if spec_sequence_masks is not None:
            assert spec_state_indices_tensor is not None
            assert qkv_spec is not None
            assert spec_query_start_loc is not None
            qkv_spec = causal_conv1d_update(
                qkv_spec,
                conv_state,
                conv_weights,
                conv_bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][
                    : attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )
        if attn_metadata.num_prefills > 0:
            assert qkv_non_spec is not None
            assert g_non_spec is not None and beta_non_spec is not None
            assert non_spec_query_start_loc is not None
            assert non_spec_state_indices_tensor is not None
            qkv_non_spec = causal_conv1d_fn(
                qkv_non_spec.transpose(0, 1),
                conv_weights,
                conv_bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            assert qkv_non_spec is not None
            assert g_non_spec is not None and beta_non_spec is not None
            assert non_spec_query_start_loc is not None
            assert non_spec_state_indices_tensor is not None
            qkv_non_spec = causal_conv1d_update(
                qkv_non_spec,
                conv_state,
                conv_weights,
                conv_bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
                validate_data=False,
            )
        else:
            qkv_non_spec = None

        # 3. Recurrence.
        if spec_sequence_masks is not None:
            assert qkv_spec is not None
            assert spec_query_start_loc is not None
            q_spec, k_spec, v_spec = self._split_qkv(qkv_spec)
            core_attn_out_spec, _ = fused_recurrent_kda_fwd(
                q=q_spec,
                k=k_spec,
                v=v_spec,
                g=self._fused_gate(g_spec),
                beta=beta_spec,
                scale=self.head_k_dim**-0.5,
                initial_state=recurrent_state,
                inplace_final_state=True,
                cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out_spec = None

        split_non_spec = (
            spec_sequence_masks is None
            and attn_metadata.num_prefills > 0
            and attn_metadata.num_decodes > 0
        )
        num_decode_tokens = attn_metadata.num_decode_tokens
        if split_non_spec:
            assert qkv_non_spec is not None
            assert g_non_spec is not None and beta_non_spec is not None
            assert non_spec_query_start_loc is not None
            assert non_spec_state_indices_tensor is not None
            q_dec, k_dec, v_dec = self._split_qkv(qkv_non_spec[:num_decode_tokens])
            core_attn_out_decode, _ = fused_recurrent_kda(
                q=q_dec,
                k=k_dec,
                v=v_dec,
                g=self._fused_gate(g_non_spec[:, :num_decode_tokens]),
                beta=beta_non_spec[:, :num_decode_tokens],
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                ssm_state_indices=non_spec_state_indices_tensor,
            )
        else:
            core_attn_out_decode = None

        if attn_metadata.num_prefills > 0:
            assert qkv_non_spec is not None
            assert g_non_spec is not None and beta_non_spec is not None
            assert non_spec_query_start_loc is not None
            assert non_spec_state_indices_tensor is not None
            if split_non_spec:
                qkv_prefill = qkv_non_spec[num_decode_tokens:]
                g_prefill = g_non_spec[:, num_decode_tokens:]
                beta_prefill = beta_non_spec[:, num_decode_tokens:]
            else:
                qkv_prefill, g_prefill, beta_prefill = (
                    qkv_non_spec,
                    g_non_spec,
                    beta_non_spec,
                )
            q_pre, k_pre, v_pre = self._split_qkv(qkv_prefill)
            prefill_state_indices = attn_metadata.prefill_state_indices
            prefill_has_initial_state = attn_metadata.prefill_has_initial_state
            assert prefill_state_indices is not None
            assert prefill_has_initial_state is not None
            initial_state = recurrent_state[prefill_state_indices]
            initial_state[~prefill_has_initial_state, ...] = 0
            core_attn_out_non_spec, last_recurrent_state = chunk_kda_with_fused_gate(
                q=q_pre,
                k=k_pre,
                v=v_pre,
                raw_g=g_prefill,
                beta=beta_prefill,
                A_log=self.A_log,
                g_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=attn_metadata.prefill_query_start_loc,
            )
            recurrent_state[prefill_state_indices] = last_recurrent_state.to(
                recurrent_state.dtype
            )
            if split_non_spec:
                core_attn_out_non_spec = torch.cat(
                    [core_attn_out_decode, core_attn_out_non_spec], dim=1
                )
        elif attn_metadata.num_decodes > 0:
            assert qkv_non_spec is not None
            assert g_non_spec is not None and beta_non_spec is not None
            assert non_spec_query_start_loc is not None
            assert non_spec_state_indices_tensor is not None
            q_dec, k_dec, v_dec = self._split_qkv(qkv_non_spec)
            core_attn_out_non_spec, _ = fused_recurrent_kda(
                q=q_dec,
                k=k_dec,
                v=v_dec,
                g=self._fused_gate(g_non_spec),
                beta=beta_non_spec,
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                ssm_state_indices=non_spec_state_indices_tensor,
            )
        else:
            core_attn_out_non_spec = None

        # 4. Merge.
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)


class BerryLMDecoderLayer(nn.Module):
    """Hybrid decoder layer: KDA linear attention or gated full attention, then the
    sparse MoE; the AttnRes mixer runs in the model loop."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        # Sequence-parallel MoE (attention reduce-scatter) is not wired for the
        # mixer; the inherited forward reads this flag.
        self.use_attn_reduce_scatter_for_moe = False

        if self.layer_type == "linear_attention":
            if not getattr(config, "kda_insert", True):
                raise NotImplementedError(
                    "BerryLM without the KDA gate (kda_insert=False) is not supported"
                )
            self.kda = BerryLMKDAAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.kda",
            )
        elif self.layer_type == "full_attention":
            self.self_attn = BerryLMAttention(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        self.mlp = BerryLMSparseMoeBlock(
            vllm_config=vllm_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = BerryLMRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = BerryLMRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_scale = False

        self.attn_res: BerryLMAttnRes | None = None
        if config.attn_res_block_size > 0:
            self.attn_res = BerryLMAttnRes(
                config.hidden_size,
                eps=getattr(config, "attn_res_eps", 1e-6),
                gated=getattr(config, "attn_res_gated", True),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        input_normed: bool = False,
        **kwargs: object,
    ):
        full_num_tokens = positions.shape[-1]

        if input_normed:
            # the AttnRes layer entry already produced (input_layernorm(mixed), mixed)
            assert residual is not None
        elif residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.use_attn_reduce_scatter_for_moe:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[:full_num_tokens]

        if self.layer_type == "linear_attention":
            hidden_states = self.kda(hidden_states=hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        if self.use_attn_reduce_scatter_for_moe:
            tp_world_size = get_tensor_model_parallel_world_size()
            # small trick using minus, eg. -17 % 8 = 7
            sp_pad = (-hidden_states.shape[0]) % tp_world_size
            # pad if not divisible by world size
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, sp_pad))
            hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.use_attn_reduce_scatter_for_moe:
            hidden_states = self.mlp(
                hidden_states,
                already_sequence_parallel=True,
            )
        else:
            hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@support_torch_compile
class BerryLMModel(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            # weight_name: (param_name, shard_id)
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".shared_expert.gate_proj": (".shared_expert.gate_up_proj", 0),
            ".shared_expert.up_proj": (".shared_expert.gate_up_proj", 1),
            ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
            ".in_proj_z": (".in_proj_qkvz", 3),
            ".f_down_proj.": (".in_proj_bf.", 0),
            ".in_proj_b.": (".in_proj_bf.", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: BerryLMConfig = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        self.quant_config = vllm_config.quant_config
        self.vocab_size = config.vocab_size
        self.attn_res_block_size = int(config.attn_res_block_size)
        if self.attn_res_block_size > 0 and get_pp_group().world_size > 1:
            # The mixer reads the residual streams committed on earlier layers,
            # which would have to travel through IntermediateTensors.
            raise NotImplementedError(
                "BerryLM AttnRes does not support pipeline parallelism yet"
            )

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        layer_types = config.layer_types
        assert layer_types is not None  # filled / validated by BerryLMConfig

        def get_layer(prefix: str):
            return BerryLMDecoderLayer(
                vllm_config,
                layer_type=layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        if get_pp_group().is_last_rank:
            self.norm = BerryLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        block_size = self.attn_res_block_size
        # Residual streams committed at block boundaries (the embedding stream
        # first); local graph intermediates, so CUDA-graph safe. The stream entering
        # the first layer of a block is committed by that layer's entry kernel.
        completed: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if block_size:
                commit = layer_idx % block_size == 0
                hidden_states, residual, committed = layer.attn_res(
                    hidden_states, residual, completed, layer.input_layernorm, commit
                )
                if commit:
                    completed.append(committed)
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                input_normed=bool(block_size),
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class BerryLMMixtureOfExperts(MixtureOfExperts):
    # set by BerryLMForCausalLM.__init__ and set_moe_parameters
    model: "BerryLMModel"
    num_moe_layers: int
    num_expert_groups: int
    num_logical_experts: int
    num_physical_experts: int
    num_local_physical_experts: int
    num_routed_experts: int
    num_shared_experts: int
    num_redundant_experts: int

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, BerryLMSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, BerryLMDecoderLayer) and isinstance(
                layer.mlp, BerryLMSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No BerryLM MoE layer found in the model.layers.")

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


class BerryLMForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    BerryLMMixtureOfExperts,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_bf": ["f_down_proj", "in_proj_b"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "BerryLM currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = BerryLMModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        # Set MoE hyperparameters
        self.set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        # KDA keeps its recurrent state in fp32 (as KimiLinear does and as the model was
        # trained); the GDN calculator would follow --mamba-ssm-cache-dtype (auto =
        # bf16) and drift over long generations.
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # A training export may still carry the MTP head; it is not part of this model
        # (vLLM main dropped AutoWeightsLoader's skip_prefixes, so the filter lives
        # here).
        loader = AutoWeightsLoader(self)
        return loader.load_weights(
            (n, w) for n, w in weights if not n.startswith("mtp.")
        )
