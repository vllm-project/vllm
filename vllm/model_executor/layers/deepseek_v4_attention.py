# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepseekV4 MLA Attention Layer
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.triton_utils import LOG2E, tl, triton
from vllm.utils.deep_gemm import fp8_einsum, use_dsv4_reference_kernels
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.ops.deepseek_v4_ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    fused_indexer_q_rope_quant,
    fused_inv_rope_fp8_quant,
    fused_q_kv_rmsnorm,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.deepseek_compressor import DeepseekCompressor
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.input_quant_fp8 import (
    QuantFP8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
)
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    DeepseekV4FlashMLASparseBackend,
    FlashMLASparseBackend,
    FlashMLASparseMetadata,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager


@triton.jit
def _dsv4_sm80_sparse_attn_split_kernel(
    q_ptr,  # (B*S, H, D) bf16
    kv_ptr,  # (B*S, T, D) bf16 — pre-gathered (zero rows for invalid)
    invalid_mask_ptr,  # (B*S, T) uint8 (1 = invalid)
    acc_split_ptr,  # (B*S, SPLIT_T, H, D_V) fp32
    max_split_ptr,  # (B*S, SPLIT_T, H) fp32
    sum_split_ptr,  # (B*S, SPLIT_T, H) fp32
    n_tokens,
    total_topk,
    sm_scale_log2,  # scale * LOG2E
    H: tl.constexpr,
    D: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    SPLIT_T: tl.constexpr,
):
    """Split-K sparse-attention decode pass 1 for V4 on SM80.

    Each program processes one chunk of `total_topk` (sized `chunk =
    ceil(total_topk / SPLIT_T)`) and emits unnormalised partial outputs:
    `acc = sum_n exp2(qk_n - max_s) * v_n`, plus the per-split max and
    sum. The combine kernel performs the cross-split LSE merge and sink
    correction.

    Splitting the topk axis lifts grid parallelism from
    `(n_tokens, ceil(H/BLOCK_H))` to `(n_tokens, SPLIT_T, ceil(H/BLOCK_H))`,
    which matters at batch=1 single-decode where only 1 SM was active.
    """
    pid_t = tl.program_id(0)
    pid_split = tl.program_id(1)
    pid_h = tl.program_id(2)

    if pid_t >= n_tokens:
        return

    chunk_size = (total_topk + SPLIT_T - 1) // SPLIT_T
    n_start_chunk = pid_split * chunk_size
    n_end_chunk = tl.minimum(n_start_chunk + chunk_size, total_topk)

    head_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_off < H

    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    # Hold q in bf16 — only used as bf16 in the inner-loop dot.
    q = tl.load(
        q_ptr + pid_t * H * D + head_off[:, None] * D + d_off[None, :],
        mask=head_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    e_max = tl.zeros((BLOCK_H,), dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_DV), dtype=tl.float32)

    n_iter = (chunk_size + BLOCK_N - 1) // BLOCK_N
    for n_block in range(n_iter):
        n_start = n_start_chunk + n_block * BLOCK_N
        n_off = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_off < n_end_chunk

        invalid_u8 = tl.load(
            invalid_mask_ptr + pid_t * total_topk + n_off,
            mask=n_mask,
            other=1,
        )
        valid = (invalid_u8 == 0) & n_mask

        # Load kv directly as bf16; tl.dot accumulates to fp32.
        kv = tl.load(
            kv_ptr + pid_t * total_topk * D + n_off[:, None] * D + d_off[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(kv))
        qk *= sm_scale_log2
        qk = tl.where(head_mask[:, None] & valid[None, :], qk, -1.0e30)

        n_e_max = tl.maximum(tl.max(qk, axis=1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        # V == K for V4 — reuse the loaded kv tile for the pv dot.
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(tl.bfloat16), kv)
        e_sum = e_sum * re_scale + tl.sum(p, axis=1)
        e_max = n_e_max

    # Store partials. Layout: (B*S, SPLIT_T, H, D_V) for acc,
    # (B*S, SPLIT_T, H) for max/sum — keeps the per-split stride contiguous
    # so the combine kernel can issue coalesced loads.
    dv_off = tl.arange(0, BLOCK_DV)
    dv_mask = dv_off < D_V
    base_acc = (
        pid_t * SPLIT_T * H * D_V
        + pid_split * H * D_V
        + head_off[:, None] * D_V
        + dv_off[None, :]
    )
    tl.store(
        acc_split_ptr + base_acc,
        acc,
        mask=head_mask[:, None] & dv_mask[None, :],
    )

    base_ms = pid_t * SPLIT_T * H + pid_split * H + head_off
    tl.store(max_split_ptr + base_ms, e_max, mask=head_mask)
    tl.store(sum_split_ptr + base_ms, e_sum, mask=head_mask)


@triton.jit
def _dsv4_sm80_sparse_attn_combine_kernel(
    acc_split_ptr,  # (B*S, SPLIT_T, H, D_V) fp32
    max_split_ptr,  # (B*S, SPLIT_T, H) fp32
    sum_split_ptr,  # (B*S, SPLIT_T, H) fp32
    attn_sink_ptr,  # (H,) fp32
    out_ptr,  # (B*S, H, D_V) bf16
    n_tokens,
    has_sink: tl.constexpr,
    H: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    SPLIT_T: tl.constexpr,
):
    """Cross-split LSE merge with sink correction.

    Reads `SPLIT_T` partial (acc, max, sum) tuples and emits the final
    softmax-normalised output. Sink contributes `exp2(sink_log2 - max)`
    to the global denominator only — it has no v term.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_t >= n_tokens:
        return

    head_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_off < H

    # Find global max over splits (and sink, if any).
    e_max_global = tl.zeros((BLOCK_H,), dtype=tl.float32) - 1.0e30
    for s in range(SPLIT_T):
        m = tl.load(
            max_split_ptr + pid_t * SPLIT_T * H + s * H + head_off,
            mask=head_mask,
            other=-1.0e30,
        )
        e_max_global = tl.maximum(e_max_global, m)

    sink_log2 = tl.zeros((BLOCK_H,), dtype=tl.float32)
    if has_sink:
        sink = tl.load(attn_sink_ptr + head_off, mask=head_mask, other=0.0)
        sink_log2 = sink * LOG2E
        e_max_global = tl.maximum(e_max_global, sink_log2)

    # Renormalise and reduce.
    dv_off = tl.arange(0, BLOCK_DV)
    dv_mask = dv_off < D_V
    acc_global = tl.zeros((BLOCK_H, BLOCK_DV), dtype=tl.float32)
    sum_global = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for s in range(SPLIT_T):
        m_s = tl.load(
            max_split_ptr + pid_t * SPLIT_T * H + s * H + head_off,
            mask=head_mask,
            other=-1.0e30,
        )
        sum_s = tl.load(
            sum_split_ptr + pid_t * SPLIT_T * H + s * H + head_off,
            mask=head_mask,
            other=0.0,
        )
        scale = tl.exp2(m_s - e_max_global)
        sum_global += scale * sum_s

        base_acc = (
            pid_t * SPLIT_T * H * D_V
            + s * H * D_V
            + head_off[:, None] * D_V
            + dv_off[None, :]
        )
        acc_s = tl.load(
            acc_split_ptr + base_acc,
            mask=head_mask[:, None] & dv_mask[None, :],
            other=0.0,
        )
        acc_global += scale[:, None] * acc_s

    if has_sink:
        sum_global += tl.exp2(sink_log2 - e_max_global)

    sum_safe = tl.where(sum_global > 0, sum_global, 1.0)
    out = (acc_global / sum_safe[:, None]).to(tl.bfloat16)
    tl.store(
        out_ptr + pid_t * H * D_V + head_off[:, None] * D_V + dv_off[None, :],
        out,
        mask=head_mask[:, None] & dv_mask[None, :],
    )


def _dsv4_sm80_sparse_attn_decode_triton(
    q: torch.Tensor,  # (B*S, H, D) bf16
    gathered_kv: torch.Tensor,  # (B*S, T, D) bf16
    invalid_mask: torch.Tensor,  # (B*S, T) bool
    attn_sink: torch.Tensor | None,  # (H,) fp32
    sm_scale: float,
    head_dim_v: int,
) -> torch.Tensor:
    """Split-K sparse-attention decode for V4 on SM80.

    Two-kernel pipeline: a split-K pass over the topk dimension followed
    by an LSE-merge combine. SPLIT_T is bounded by the BLOCK_N-tile count
    so each split has real work to do."""
    n_tokens, h, d = q.shape
    _, t, d_kv = gathered_kv.shape
    assert d_kv == d
    assert invalid_mask.shape == (n_tokens, t)

    block_d = triton.next_power_of_2(d)
    block_dv = triton.next_power_of_2(head_dim_v)
    # BLOCK_H capped at 16 (tl.dot's M-min) so the fp32 `acc` tile
    # (BLOCK_H × BLOCK_DV) stays under A100's 100 KB SMEM limit and so we
    # get more per-token head blocks for SM utilisation when h > 16.
    block_h = 16
    block_n = 32  # keeps q/kv/acc tiles within A100's 164KB SM

    # SPLIT_T heuristic: cap at 16 (combine overhead dominates beyond
    # that) but otherwise split as much as we have BLOCK_N tiles. Lifts
    # grid parallelism from (n_tokens, ceil(h/BLOCK_H)) to
    # (n_tokens, SPLIT_T, ceil(h/BLOCK_H)) — at batch=1 single-decode the
    # original kernel used 1 SM out of 108.
    n_tiles = (t + block_n - 1) // block_n
    split_t = max(1, min(16, n_tiles))

    out = torch.empty((n_tokens, h, head_dim_v), dtype=torch.bfloat16, device=q.device)
    invalid_u8 = invalid_mask.to(torch.uint8)

    acc_split = torch.empty(
        (n_tokens, split_t, h, head_dim_v),
        dtype=torch.float32,
        device=q.device,
    )
    max_split = torch.empty(
        (n_tokens, split_t, h), dtype=torch.float32, device=q.device
    )
    sum_split = torch.empty_like(max_split)

    grid_split = (n_tokens, split_t, triton.cdiv(h, block_h))
    _dsv4_sm80_sparse_attn_split_kernel[grid_split](
        q,
        gathered_kv,
        invalid_u8,
        acc_split,
        max_split,
        sum_split,
        n_tokens,
        t,
        sm_scale * LOG2E,
        H=h,
        D=d,
        D_V=head_dim_v,
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        BLOCK_DV=block_dv,
        SPLIT_T=split_t,
        num_warps=4,
    )

    grid_combine = (n_tokens, triton.cdiv(h, block_h))
    _dsv4_sm80_sparse_attn_combine_kernel[grid_combine](
        acc_split,
        max_split,
        sum_split,
        attn_sink if attn_sink is not None else q.new_zeros(h),
        out,
        n_tokens,
        has_sink=(attn_sink is not None),
        H=h,
        D_V=head_dim_v,
        BLOCK_H=block_h,
        BLOCK_DV=block_dv,
        SPLIT_T=split_t,
        num_warps=4,
    )
    return out


logger = init_logger(__name__)

# Prefill is processed in fixed-size chunks; this bounds the bf16 kv-gather
# workspace allocated at _forward_prefill (and the matching profile-time
# reservation in attention_impl's dummy-run branch).
PREFILL_CHUNK_SIZE = 4


@dataclass
class DeepseekV4MLAModules:
    """Modules used in DeepseekV4 MLA."""

    vllm_config: VllmConfig
    fused_wqa_wkv: torch.nn.Module
    q_norm: torch.nn.Module
    wq_b: torch.nn.Module
    kv_norm: torch.nn.Module
    wo_a: torch.nn.Module
    wo_b: torch.nn.Module
    attn_sink: torch.nn.Module
    rotary_emb: torch.nn.Module
    indexer: torch.nn.Module | None
    indexer_rotary_emb: torch.nn.Module
    topk_indices_buffer: torch.Tensor | None
    aux_stream: torch.cuda.Stream | None = None


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("deepseek_v4_multi_head_latent_attention")
class DeepseekV4MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        o_lora_rank: int | None,
        mla_modules: DeepseekV4MLAModules,
        window_size: int,
        compress_ratio: int | None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_local_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale

        # FlashMLA sparse kernel only supports 64 or 128 heads; pad up to the
        # next supported size. Must match DeepseekV4MLAAttention.padded_heads.
        if num_heads <= 64:
            self.padded_heads = 64
        elif num_heads <= 128:
            self.padded_heads = 128
        else:
            raise ValueError(
                f"DeepseekV4 attention does not support {num_heads} heads "
                "(must be <= 128)."
            )

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.window_size = window_size
        self.compress_ratio = compress_ratio if compress_ratio is not None else 1
        self.prefix = prefix

        # Extract config from vllm_config
        config = mla_modules.vllm_config.model_config.hf_config
        tp_size = get_tensor_model_parallel_world_size()

        # DeepseekV4-specific attributes (num_heads is already TP-adjusted)
        self.eps = config.rms_norm_eps
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = head_dim - self.rope_head_dim
        self.n_local_groups = config.o_groups // tp_size
        self.o_lora_rank = config.o_lora_rank

        # Store projection modules
        self.fused_wqa_wkv = mla_modules.fused_wqa_wkv
        self.q_norm = mla_modules.q_norm
        self.wq_b = mla_modules.wq_b

        self.kv_norm = mla_modules.kv_norm
        self.wo_a = mla_modules.wo_a

        # SM80 BMM cache for wo_a when n_local_groups > 1 (TP < o_groups).
        # Marlin-packed FP8 doesn't support per-group output slicing, so we
        # lazily dequantize to bf16 and run torch.bmm. At n_local_groups == 1
        # the flat Marlin path is mathematically a 1-batch BMM, so we keep it.
        self._wo_a_bmm_weight: torch.Tensor | None = None

        self._wo_a_act_quant = QuantFP8(
            static=False,
            group_shape=GroupShape(1, 128),
            use_ue8m0=True,
        )
        # Bypass packed-for-deepgemm path — we need FP32 scales (not packed
        # INT32) so fp8_einsum can handle layout transform internally.
        self._wo_a_act_quant.use_deep_gemm_supported = False
        self.wo_b = mla_modules.wo_b

        # Pick fp8_einsum recipe based on GPU arch:
        # SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128
        # SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1
        from vllm.platforms import current_platform

        cap = current_platform.get_device_capability()
        assert cap is not None, "DeepseekV4 attention requires a CUDA device"
        self._einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
        self._tma_aligned_scales = cap.major >= 10

        self.rotary_emb = mla_modules.rotary_emb
        self.indexer_rotary_emb = mla_modules.indexer_rotary_emb
        self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.indexer = mla_modules.indexer

        # Per-head RMS normalization for Q (no learnable weights)
        self.q_head_norm = RMSNorm(head_dim, eps=self.eps, has_weight=False)

        # TODO(yifan): currently hardcoded for FP8 sparse, make it more generic
        head_bytes = (
            self.nope_head_dim  # 448 fp8 NoPE
            + self.rope_head_dim * 2  # 64 bf16 RoPE
            + self.nope_head_dim // 64  # 7B scale factors
            + 1  # 1B pad
        )

        self.aux_stream = mla_modules.aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        assert cache_config is not None, "DeepseekV4 attention requires cache_config"
        self.swa_cache_layer = DeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=torch.uint8,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        self.mla_attn = DeepseekV4MLAAttention(
            num_heads=self.n_local_heads,
            head_dim=self.head_dim,
            scale=self.scale,
            qk_nope_head_dim=self.nope_head_dim,
            qk_rope_head_dim=self.rope_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            compress_ratio=self.compress_ratio,
            window_size=self.window_size,
            head_bytes=head_bytes,
            swa_cache_layer=self.swa_cache_layer,
            attn_sink=mla_modules.attn_sink,  # already padded with -inf
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            indexer=self.indexer,
            topk_indices_buffer=self.topk_indices_buffer,
        )
        # Register this layer in the compilation config's static forward context
        # This allows the custom op to retrieve the layer during execution
        compilation_config = mla_modules.vllm_config.compilation_config
        # HACK
        self.layer_name = prefix + ".deepseek_v4_multi_head_latent_attention"
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self

        # Create the compressor for layers with compress_ratio > 1; after
        # creating the DeepseekV4MLAAttention layer to get its cache.
        self.compressor = None
        if self.compress_ratio > 1:
            self.compressor = DeepseekCompressor(
                vllm_config=mla_modules.vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.mla_attn.prefix,
            )

    def _ensure_wo_a_bmm_weight(self, ref: torch.Tensor) -> None:
        """Lazily build a (n_local_groups, K, N_per_group) bf16 BMM weight.

        Marlin doesn't expose its packed FP8 weight; recover it by running
        an identity matrix through the linear so the bf16 output is W^T at
        full precision."""
        if self._wo_a_bmm_weight is not None:
            return
        K = self.wo_a.input_size_per_partition
        N_total = self.wo_a.output_size_per_partition
        n_groups = self.n_local_groups
        N_per_group = N_total // n_groups
        eye = torch.eye(K, dtype=ref.dtype, device=ref.device)
        with torch.no_grad():
            w_t = self.wo_a(eye)  # (K, N_total) bf16, dequantised
        if isinstance(w_t, tuple):
            w_t = w_t[0]
        # (K, n_groups, N_per_group) -> (n_groups, K, N_per_group)
        self._wo_a_bmm_weight = (
            w_t.view(K, n_groups, N_per_group).permute(1, 0, 2).contiguous()
        )

    def _apply_wo_a_bmm(self, o_rotated: torch.Tensor) -> torch.Tensor:
        """Per-group BMM dispatch for n_local_groups > 1 (TP < o_groups).

        Input is (T, n_local_heads, head_dim) where n_local_heads splits
        evenly across n_local_groups. A flat Marlin call would mix groups
        across the K axis; we use torch.bmm against the per-group bf16
        weight instead."""
        self._ensure_wo_a_bmm_weight(o_rotated)
        assert self._wo_a_bmm_weight is not None
        n_groups = self.n_local_groups
        T = o_rotated.shape[0]
        K_per_group = self._wo_a_bmm_weight.shape[1]
        N_per_group = self._wo_a_bmm_weight.shape[2]
        # (T, n_local_heads, head_dim) -> (T, n_groups, K_per_group)
        # -> (n_groups, T, K_per_group) for bmm.
        x = o_rotated.reshape(T, n_groups, K_per_group).transpose(0, 1).contiguous()
        out = torch.bmm(x, self._wo_a_bmm_weight)  # (n_groups, T, N_per_group)
        # (n_groups, T, N_per_group) -> (T, n_groups * N_per_group)
        return out.transpose(0, 1).reshape(T, n_groups * N_per_group)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)

        # Lift q/kv RMSNorm out of the attention custom op so the surrounding
        # residual / norm graph isn't cut by the opaque boundary. The op
        # now expects pre-normed inputs; Inductor can combo-fuse this norm
        # with adjacent ops in the wrapper-level graph.
        qr, kv = fused_q_kv_rmsnorm(
            qr,
            kv,
            self.q_norm.weight.data,
            self.kv_norm.weight.data,
            self.eps,
        )

        # Lift `wq_b` out as well so the projection-from-LoRA matmul is
        # graph-visible. The indexer still consumes the LoRA-rank `qr`,
        # so we pass both `qr` and the projected per-head `q` to the op.
        num_tokens = hidden_states.shape[0]
        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)

        # Pre-allocate attention output with FlashMLA-padded head count.
        # The op writes into `o_padded`; we slice to n_local_heads after.
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Attention (inside custom op for torch.compile boundary)
        torch.ops.vllm.deepseek_v4_attention(
            hidden_states,
            qr,
            kv,
            q,
            positions,
            o_padded,
            self.layer_name,
        )
        o = o_padded[:, : self.n_local_heads, :]

        if use_dsv4_reference_kernels():
            # SM80/ROCm reference path: bf16 inv-RoPE then wo_a. At
            # n_local_groups>1 wo_a is a per-group BMM and we route through
            # the bf16 dequant + torch.bmm path; the n_local_groups==1 case
            # is a 1-batch BMM and runs as a flat Marlin GEMM.
            o_rotated = _apply_inv_rope_to_o(
                o,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.rope_head_dim,
            )
            if self.n_local_groups > 1:
                return self.wo_b(self._apply_wo_a_bmm(o_rotated))
            return self.wo_b(self.wo_a(o_rotated.flatten(1)))

        # O projection: inverse RoPE + FP8 quant + einsum + wo_b
        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            tma_aligned_scales=self._tma_aligned_scales,
        )

        wo_a_fp8 = self.wo_a.weight
        wo_a_scale = self.wo_a.weight_scale_inv

        z = torch.empty(
            (num_tokens, self.n_local_groups, self.o_lora_rank),
            device=o.device,
            dtype=torch.bfloat16,
        )
        torch.ops.vllm.deepseek_v4_fp8_einsum(
            o_fp8,
            o_scale,
            wo_a_fp8,
            wo_a_scale,
            z,
            "bhr,hdr->bhd",
            list(self._einsum_recipe),
        )

        return self.wo_b(z.flatten(1))

    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        q: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, padded_heads, head_dim], written in place
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        # `qr`, `kv`, and `q` are all pre-computed in the calling `forward()`:
        # `fused_q_kv_rmsnorm` and `wq_b` were lifted out so the surrounding
        # residual/RMSNorm graph is no longer cut by the custom-op boundary.
        # Inductor can now combo-fuse those tails with adjacent ops.
        # The indexer (below) still consumes `qr` (the normed LoRA tensor)
        # while kv_insert and `mla_attn` consume `q`.

        # Overlap kv_insert with whichever of indexer/compressor is present.
        # Indexer implies compressor; when both exist, compressor rides on the
        # aux stream alongside kv_insert so the heavy indexer owns default.
        if self.indexer is not None:
            indexer = self.indexer
            # Local ref so the closure keeps a non-None type for mypy.
            assert self.compressor is not None
            compressor = self.compressor

            def kv_insert_and_compress() -> None:
                self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
                compressor(hidden_states, positions, self.rotary_emb)

            maybe_execute_in_parallel(
                lambda: indexer(hidden_states, qr, positions, self.indexer_rotary_emb),
                kv_insert_and_compress,
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
        elif self.compressor is not None:
            # Compressor on default, kv_insert on aux.
            compressor = self.compressor
            maybe_execute_in_parallel(
                lambda: compressor(hidden_states, positions, self.rotary_emb),
                lambda: self._fused_qnorm_rope_kv_insert(
                    q, kv, positions, attn_metadata
                ),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
        else:
            # SWA-only layer: no compressor, no overlap.
            self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

        # Handle dummy run (no metadata).
        if not isinstance(attn_metadata, dict):
            # Reserve _forward_prefill's bf16-gather workspace; the dummy
            # run returns before mla_attn runs, so without this the shared
            # workspace locks below the real prefill size.
            sub = self.mla_attn
            swa_only = sub.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (sub.max_model_len + sub.compress_ratio - 1) // sub.compress_ratio
            )
            M = N + sub.window_size + sub.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            out.zero_()
            return

        # Pad q to FlashMLA-required head count (64 or 128)
        if self.n_local_heads < self.padded_heads:
            pad_size = self.padded_heads - self.n_local_heads
            q = F.pad(q, (0, 0, 0, pad_size), value=0.0)

        # MLA attention writes into the pre-allocated `out` buffer
        # ([num_tokens, padded_heads, head_dim]).
        self.mla_attn(q, kv, positions, output=out)

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> None:
        if not isinstance(attn_metadata, dict):
            return

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_kv_cache = self.swa_cache_layer.kv_cache
        swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)

        # Horizontally fused:
        #   Q side:  q_head_norm (per-head RMSNorm, no weight) + GPT-J RoPE
        #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert
        # kv is unchanged; mla_attn reads kv solely via swa_kv_cache.
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q,
            kv,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping,
            positions.to(torch.int64),
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )


def deepseek_v4_attention(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    q: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.attention_impl(hidden_states, qr, kv, q, positions, out)


def deepseek_v4_attention_fake(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    q: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_attention",
    op_func=deepseek_v4_attention,
    mutates_args=["out"],
    fake_impl=deepseek_v4_attention_fake,
)


@triton.jit
def _inv_rope_bf16_kernel(
    o_ptr,  # (T, H, D) bf16, modified in place
    positions_ptr,  # (T,) int64
    cos_sin_cache_ptr,  # (max_pos, rope_dim) bf16
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope: tl.constexpr,
    nope_dim: tl.constexpr,
):
    """Single-launch inv-RoPE on bf16 for the SM80 reference path.

    One program per (token, head). Replaces the ~10-op PyTorch chain in
    `_apply_inv_rope_to_o` (index_select, clone, slice/stride pairs, mul,
    add, sub, copy_back) with one kernel.

    GPT-J interleaved: even/odd pairs at positions (2r, 2r+1) within the
    rope segment are rotated using the (cos, sin) at index r.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_t >= T:
        return

    pos = tl.load(positions_ptr + pid_t)
    base_cs = cos_sin_cache_ptr + pos * rope_dim
    r = tl.arange(0, half_rope)
    cos_v = tl.load(base_cs + r).to(tl.float32)
    sin_v = tl.load(base_cs + half_rope + r).to(tl.float32)

    base_row = o_ptr + (pid_t * H + pid_h) * D + nope_dim
    even = tl.load(base_row + 2 * r).to(tl.float32)
    odd = tl.load(base_row + 2 * r + 1).to(tl.float32)
    new_even = even * cos_v + odd * sin_v
    new_odd = odd * cos_v - even * sin_v
    tl.store(base_row + 2 * r, new_even.to(tl.bfloat16))
    tl.store(base_row + 2 * r + 1, new_odd.to(tl.bfloat16))


def _apply_inv_rope_to_o(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """Apply inverse GPT-J RoPE on the last `rope_dim` dims of each head.
    Used by the SM80/ROCm reference path that skips FP8 quantization.
    Matches the rotation in `_fused_inv_rope_fp8_quant_per_head` numerically."""
    if not o.is_contiguous():
        o = o.contiguous()
    out = o.clone()
    T, H, D = out.shape
    nope_dim = D - rope_dim
    half_rope = rope_dim // 2
    positions_i64 = positions.to(torch.int64).contiguous()
    cs = cos_sin_cache
    # cos_sin_cache is (max_pos, rope_dim) bf16 with the GPT-J layout
    # [cos | sin] along the last dim. We index by position and split inline.
    grid = (T, H)
    _inv_rope_bf16_kernel[grid](
        out,
        positions_i64,
        cs,
        T,
        H=H,
        D=D,
        rope_dim=rope_dim,
        half_rope=half_rope,
        nope_dim=nope_dim,
    )
    return out


def _decode_e8m0_scales(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        return _upcast_e8m0_to_fp32(scale).contiguous()
    return scale.to(torch.float32)


def _expand_last_dim_scales(scale: torch.Tensor, last_dim: int) -> torch.Tensor:
    scale = _decode_e8m0_scales(scale)
    block = math.ceil(last_dim / scale.shape[-1])
    return torch.repeat_interleave(scale, block, dim=-1)[..., :last_dim]


def _expand_2d_block_scales(
    scale: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    scale = _decode_e8m0_scales(scale)
    row_blocks, col_blocks = scale.shape[-2:]
    row_block = math.ceil(rows / row_blocks)
    col_block = math.ceil(cols / col_blocks)
    scale = torch.repeat_interleave(scale, row_block, dim=-2)[..., :rows, :]
    scale = torch.repeat_interleave(scale, col_block, dim=-1)[..., :, :cols]
    return scale


def _deepseek_v4_fp8_einsum_fallback(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
) -> None:
    """SM80/ROCm dequantize-and-einsum fallback for `bhr,hdr->bhd`.

    On SM80 with Marlin-packed FP8 weights the wrapper bypasses this and
    routes through `wo_a` as a regular Linear; this remains for ROCm and
    any non-Marlin SM80 layout."""
    if equation != "bhr,hdr->bhd":
        raise RuntimeError(f"Unsupported fallback equation: {equation}")

    num_groups = a.shape[1]
    hidden_dim = a.shape[2]
    output_dim = b.shape[0] // num_groups

    if b.shape[0] % num_groups != 0:
        raise RuntimeError(
            f"Cannot reshape weight of shape {tuple(b.shape)} into "
            f"({num_groups}, {output_dim}, {hidden_dim})."
        )

    a_deq = (a.to(torch.float32) * _expand_last_dim_scales(a_scale, hidden_dim)).to(
        torch.bfloat16
    )

    b_deq = b.view(num_groups, output_dim, hidden_dim).to(torch.float32)
    b_scale_deq = _expand_2d_block_scales(
        b_scale.view(num_groups, -1, b_scale.shape[-1]),
        output_dim,
        hidden_dim,
    )
    b_deq = (b_deq * b_scale_deq).to(torch.bfloat16)

    out.copy_(torch.einsum(equation, a_deq, b_deq).to(out.dtype))


def deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    if use_dsv4_reference_kernels():
        # SM80/ROCm: pre-gate to the dequant-and-einsum fallback. Catching
        # RuntimeError isn't enough because on SM80 DeepGEMM is importable,
        # so the C++ assert ("Unsupported architecture") fires from inside
        # the kernel call rather than from the wrapper's _missing() raise.
        _deepseek_v4_fp8_einsum_fallback(a, a_scale, b, b_scale, out, equation)
        return
    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))


def deepseek_v4_fp8_einsum_fake(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_fp8_einsum",
    op_func=deepseek_v4_fp8_einsum,
    mutates_args=["out"],
    fake_impl=deepseek_v4_fp8_einsum_fake,
)


class DeepseekV4MLAAttention(nn.Module, AttentionLayerBase):
    # FlashMLA FP8 sparse only supports 64 or 128 heads
    SUPPORTED_HEAD_COUNTS = (64, 128)

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        compress_ratio: int,
        window_size: int,
        head_bytes: int,
        swa_cache_layer: DeepseekV4SWACache,
        attn_sink: torch.Tensor,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        # Sparse MLA Args
        indexer: object | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream: torch.cuda.Stream | None = None,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = 1
        self.head_dim = head_dim
        self.scale = scale
        self.window_size = window_size
        self.head_bytes = head_bytes
        self.compress_ratio = compress_ratio
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.nope_head_dim = qk_nope_head_dim
        self.rope_head_dim = qk_rope_head_dim
        self.indexer = indexer
        self.topk_indices_buffer = topk_indices_buffer

        self.prefix = prefix  # Alias for compatibility with compressor

        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        # Cache for `torch.arange(0, topk)` used in invalid-mask construction.
        # Today the SM80 reference path constructs a fresh arange every gather
        # call (43 layers x 2 scopes per token). Each is small but the
        # allocations + kernel launches add up.
        self._arange_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

        # Determine padded head count for FlashMLA
        if num_heads not in self.SUPPORTED_HEAD_COUNTS:
            if num_heads < 64:
                self.padded_heads = 64
            elif num_heads < 128:
                self.padded_heads = 128
            else:
                raise ValueError(
                    f"DeepseekV4MLAAttention does not support {num_heads} heads. "
                    f"Supported: <= 128 (will be padded to 64 or 128)"
                )
        else:
            self.padded_heads = num_heads

        # Store attention sink
        assert attn_sink is not None
        self.attn_sink: torch.Tensor = attn_sink
        # Store SWA cache
        assert swa_cache_layer is not None
        self.swa_cache_layer: DeepseekV4SWACache = swa_cache_layer

        # Get vllm config for cache setup
        vllm_config = get_current_vllm_config()
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        # DeepseekV4 only supports fp8 kv-cache format for now
        kv_cache_dtype = cache_config.cache_dtype if cache_config is not None else "fp8"

        assert kv_cache_dtype.startswith("fp8"), (
            f"DeepseekV4 only supports fp8 kv-cache format for now, "
            f"got {kv_cache_dtype}"
        )
        assert issubclass(self.get_attn_backend(), FlashMLASparseBackend), (
            "Only FlashMLA Sparse Attention backend is supported for DeepseekV4 for now"
        )
        # FlashMLA Sparse Attention fp8 backend uses "fp8_ds_mla" kv-cache format
        # Automatically convert fp8 kv-cache format to "fp8_ds_mla"
        if (
            issubclass(self.get_attn_backend(), FlashMLASparseBackend)
            and kv_cache_dtype.startswith("fp8")
            and kv_cache_dtype != "fp8_ds_mla"
        ):
            assert cache_config is not None
            cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once("Using DeepSeek's fp8_ds_mla KV cache format.")

        self.kv_cache_dtype = kv_cache_dtype

        # Register with compilation context for metadata lookup
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self

        self.kv_cache = torch.tensor([])

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4FlashMLASparseBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if (
            self.compress_ratio <= 1
        ):  # SWA part. Allocated separately as DeepseekV4SWACache.
            return None
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576,  # NOTE: FlashMLA requires 576B alignment
            model_version="deepseek_v4",
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        # Get SWA and indexer metadata from forward context
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            FlashMLASparseMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation, so self.kv_cache may be empty after profiling cleanup.
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None

        # We treat queries in the same seq as different queries
        # and later we only attend by generated indices.
        # q arrives pre-padded to self.padded_heads by the outer wrapper.
        q = q.unsqueeze(1)

        # Prepare SWA cache (num_blocks, swa_block_size, 1, head_bytes)
        # Use unsqueeze to preserve strides (handles padded blocks correctly)
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        # Reshape KV cache to (num_blocks, block_size, 1, head_bytes)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        # One FlashMLASchedMeta per layer type, shared across all same-type
        # layers within this decode step. The first forward call per type
        # triggers the in-kernel planner (allocating tile_scheduler_metadata
        # and num_splits via PyTorch's graph-aware allocator so CUDA graph
        # capture reuses the same addresses on replay); subsequent same-type
        # layers see have_initialized=True and skip the planner.
        if self.compress_ratio <= 1:
            tile_metadata = swa_metadata.tile_sched_swaonly
        elif self.compress_ratio == 4:
            tile_metadata = swa_metadata.tile_sched_c4a
        elif self.compress_ratio == 128:
            tile_metadata = swa_metadata.tile_sched_c128a
        else:
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; "
                "expected 1, 4, or 128."
            )

        if use_dsv4_reference_kernels():
            # SM80/ROCm reference path: gather only the selected indices,
            # then dequantize. tile_metadata is unused here —
            # build_tile_scheduler short-circuits to None on these platforms.
            swa_block_size = self.swa_cache_layer.kv_cache.shape[1]
            has_extra = not swa_only and kv_cache is not None
            attn_out = self._ref_sparse_attn_decode_gather(
                q=q,
                swa_kv_cache=self.swa_cache_layer.kv_cache,
                swa_block_size=swa_block_size,
                swa_indices=swa_indices.unsqueeze(1),
                swa_topk_length=swa_lens,
                attn_sink=self.attn_sink[: q.shape[2]],
                extra_kv_cache=kv_cache.squeeze(-2) if has_extra else None,
                extra_block_size=kv_cache.shape[1] if has_extra else 0,
                extra_indices=topk_indices,
                extra_topk_length=topk_lens,
            )
            output.copy_(attn_out.to(output.dtype))
            return

        assert tile_metadata is not None, (
            "swa_metadata missing tile_sched entry for "
            f"compress_ratio={self.compress_ratio}; "
            "DeepseekSparseSWAMetadataBuilder.build_tile_scheduler did not "
            "allocate one for this layer type."
        )

        out, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=swa_cache,
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lens,
            softmax_scale=self.scale,
            attn_sink=self.attn_sink,
            extra_k_cache=kv_cache if not swa_only else None,
            extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens,
            out=output.unsqueeze(1),
        )

    def _dequantize_blocked_k_cache(self, quant_k_cache: torch.Tensor) -> torch.Tensor:
        """Dequantize a UE8M0-packed FP8 KV block cache to bf16."""
        from vllm.platforms import current_platform

        fp8_dtype = current_platform.fp8_dtype()
        d = self.head_dim
        d_nope = self.nope_head_dim
        d_rope = self.rope_head_dim
        tile_size = 64
        num_tiles = d_nope // tile_size

        num_blocks, block_size, _ = quant_k_cache.shape
        quant_k_cache = quant_k_cache.view(num_blocks, -1)
        input_nope_rope = quant_k_cache[:, : block_size * (d_nope + 2 * d_rope)].view(
            num_blocks, block_size, d_nope + 2 * d_rope
        )
        input_nope = input_nope_rope[:, :, :d_nope].view(fp8_dtype)
        input_rope = input_nope_rope[:, :, d_nope:].view(torch.bfloat16)
        input_scale = (
            quant_k_cache[:, block_size * (d_nope + 2 * d_rope) :]
            .view(num_blocks, block_size, 8)[:, :, :num_tiles]
            .view(torch.float8_e8m0fnu)
        )

        result = torch.empty(
            (num_blocks, block_size, 1, d),
            dtype=torch.bfloat16,
            device=quant_k_cache.device,
        )
        result[..., d_nope:] = input_rope.unsqueeze(2)
        for tile_idx in range(num_tiles):
            cur_nope = input_nope[
                ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
            ].to(torch.bfloat16)
            cur_scales = input_scale[:, :, tile_idx].to(torch.bfloat16).unsqueeze(-1)
            result[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
                cur_nope * cur_scales
            ).unsqueeze(2)
        return result

    def _gather_dequant_blocked_k_at_indices(
        self,
        kv_cache: torch.Tensor,
        flat_indices: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Gather and dequantize K vectors at the given flat indices
        (block_idx * block_size + pos_in_block) directly from the paged
        FP8 cache without dequantising the full cache. Returns (N, head_dim)
        bf16. Negative or out-of-range indices produce zero rows."""
        nope_dim = self.nope_head_dim
        rope_dim = self.rope_head_dim
        head_dim = self.head_dim
        quant_block = 64
        n_quant = nope_dim // quant_block
        scale_dim = n_quant + 1  # 7 real + 1 pad
        token_data_size = nope_dim + rope_dim * 2  # 448 + 128 = 576

        device = kv_cache.device
        flat_indices = flat_indices.to(torch.int64)
        valid_mask = flat_indices >= 0
        safe_idx = flat_indices.clamp_min(0)
        block_idx = safe_idx // block_size
        pos_in_block = safe_idx % block_size

        cache_u8 = (
            kv_cache.view(torch.uint8) if kv_cache.dtype != torch.uint8 else kv_cache
        )
        n_blocks = cache_u8.shape[0]
        block_stride = cache_u8[0].numel()
        flat = cache_u8.view(n_blocks, block_stride)

        block_idx = block_idx.clamp(max=n_blocks - 1)
        t_off = pos_in_block * token_data_size
        s_off = block_size * token_data_size + pos_in_block * scale_dim

        # NoPE (fp8) and RoPE (bf16-as-bytes) are contiguous within each
        # token's region of the cache, so a single fancy-index op covers
        # both: nope at [pos*token_data_size : +nope_dim], rope at
        # [+nope_dim : +nope_dim+2*rope_dim].
        nope_rope_arange = self._get_arange(nope_dim + rope_dim * 2, device)
        scale_arange = self._get_arange(n_quant, device)
        nope_rope_idx = t_off.unsqueeze(-1) + nope_rope_arange
        scale_idx = s_off.unsqueeze(-1) + scale_arange

        block_idx_b = block_idx.unsqueeze(-1)
        nope_rope_bytes = flat[block_idx_b, nope_rope_idx]  # (N, 576) uint8
        fp8_bytes = nope_rope_bytes[:, :nope_dim].contiguous().view(torch.float8_e4m3fn)
        bf16_raw = nope_rope_bytes[:, nope_dim:].contiguous()
        bf16_view = bf16_raw.view(torch.bfloat16).view(-1, rope_dim)
        scales_u8 = flat[block_idx_b, scale_idx]

        x_fp32 = fp8_bytes.to(torch.float32).view(-1, n_quant, quant_block)
        scale_factor = torch.pow(2.0, scales_u8.to(torch.float32) - 127.0).unsqueeze(-1)
        dequant_fp8 = (x_fp32 * scale_factor).view(-1, nope_dim).to(torch.bfloat16)

        full = torch.empty(
            (flat_indices.shape[0], head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        full[:, :nope_dim] = dequant_fp8
        full[:, nope_dim:] = bf16_view
        full.masked_fill_(~valid_mask.unsqueeze(-1), 0)
        return full

    def _get_arange(self, n: int, device: torch.device) -> torch.Tensor:
        """Cached `torch.arange(0, n)` keyed by (length, device).

        Each decode step calls this layer's gather many times with
        identical (length, device) tuples; the cache avoids the per-call
        allocation and kernel launch."""
        key = (n, device)
        cached = self._arange_cache.get(key)
        if cached is None:
            cached = torch.arange(0, n, device=device)
            self._arange_cache[key] = cached
        return cached

    def _ref_sparse_attn_decode_gather(
        self,
        q: torch.Tensor,
        swa_kv_cache: torch.Tensor,
        swa_block_size: int,
        swa_indices: torch.Tensor,
        swa_topk_length: torch.Tensor | None,
        attn_sink: torch.Tensor | None,
        extra_kv_cache: torch.Tensor | None,
        extra_block_size: int,
        extra_indices: torch.Tensor | None,
        extra_topk_length: torch.Tensor | None,
    ) -> torch.Tensor:
        """SM80 reference decode: gather-then-dequantise only the topk
        positions, then dispatch to the split-K attention kernel."""
        b, s_q, h_q, d_qk = q.shape
        d_v = self.head_dim

        def gather_scope(
            kv_cache: torch.Tensor,
            block_size: int,
            indices: torch.Tensor,
            topk_length: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            indices = indices.reshape(b, s_q, -1)
            topk = indices.size(-1)
            gathered = self._gather_dequant_blocked_k_at_indices(
                kv_cache, indices.reshape(-1), block_size
            ).view(b, s_q, topk, d_qk)
            invalid_mask = indices == -1
            if topk_length is not None:
                topk_length = topk_length.reshape(b)
                arange = self._get_arange(topk, invalid_mask.device)
                invalid_mask |= arange.view(1, 1, topk) >= topk_length.view(b, 1, 1)
            return gathered, invalid_mask

        gathered_kv, invalid_mask = gather_scope(
            swa_kv_cache, swa_block_size, swa_indices, swa_topk_length
        )
        if extra_kv_cache is not None and extra_indices is not None:
            extra_gathered, extra_invalid = gather_scope(
                extra_kv_cache, extra_block_size, extra_indices, extra_topk_length
            )
            gathered_kv = torch.cat([gathered_kv, extra_gathered], dim=2)
            invalid_mask = torch.cat([invalid_mask, extra_invalid], dim=2)

        # No NaN scrub: the FP8 quantiser clamps to +/-448 and the gather
        # zeroes invalid rows in-place, so the gathered buffer is NaN-free.
        bs = b * s_q
        gathered_kv_flat = gathered_kv.view(bs, -1, d_qk)
        invalid_flat = invalid_mask.view(bs, -1)
        # q may arrive non-contiguous from the upstream o_padded[...] slice.
        q_flat = q.view(bs, h_q, d_qk).to(torch.bfloat16).contiguous()

        out_flat = _dsv4_sm80_sparse_attn_decode_triton(
            q_flat,
            gathered_kv_flat,
            invalid_flat,
            attn_sink,
            self.scale,
            d_v,
        )
        # Match the prior PyTorch shape: (b, h_q, d_v) for s_q=1.
        return out_flat.view(b, h_q, d_v)

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
            # Compressed region must fit the full compressed pool (seq_len //
            # compress_ratio), not just top_k. top_k bounds how many indices
            # the indexer selects, not the pool size it indexes into.
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            # NOTE(woosuk): topk_indices will not be used for SWA-only layers.
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        num_chunks = (num_prefills + PREFILL_CHUNK_SIZE - 1) // PREFILL_CHUNK_SIZE

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )

            if use_dsv4_reference_kernels():
                # SM80/ROCm reference path. The reference returns the
                # attention output rather than writing to `out=`, so copy
                # into the output slice.
                output_chunk = self._ref_sparse_attn_prefill(
                    q=q[query_start:query_end],
                    kv=kv.view(-1, 1, q.shape[-1]),
                    indices=combined_indices.unsqueeze(1),
                    topk_length=combined_lens,
                )
                output[query_start:query_end].copy_(output_chunk.to(output.dtype))
            else:
                output_chunk, _, _ = flash_mla_sparse_fwd(
                    q=q[query_start:query_end],
                    kv=kv.view(-1, 1, q.shape[-1]),
                    indices=combined_indices.unsqueeze(1),
                    sm_scale=self.scale,
                    attn_sink=self.attn_sink,
                    topk_length=combined_lens,
                    out=output[query_start:query_end],
                )

    def _ref_sparse_attn_prefill(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        topk_length: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pure-PyTorch sparse MLA prefill reference."""
        indices = indices.clone().squeeze(1)
        s_q, h_q, d_qk = q.shape
        topk = indices.shape[-1]
        s_kv = kv.shape[0]
        if topk_length is not None:
            mask = torch.arange(topk, device=indices.device).unsqueeze(
                0
            ) >= topk_length.unsqueeze(1)
            indices[mask] = -1
        invalid_mask = (indices < 0) | (indices >= s_kv)
        indices[invalid_mask] = 0

        qf = q.float()
        gathered_kv = (
            kv.index_select(0, indices.flatten()).reshape(s_q, topk, d_qk).float()
        )
        scores = qf @ gathered_kv.transpose(1, 2)
        scores *= self.scale
        scores[invalid_mask.unsqueeze(1).expand_as(scores)] = float("-inf")

        orig_lse = torch.logsumexp(scores, dim=-1)
        lse_for_o = orig_lse
        if self.attn_sink is not None:
            lse_for_o = torch.logsumexp(
                torch.stack(
                    [
                        orig_lse,
                        self.attn_sink[:h_q].view(1, h_q).expand_as(orig_lse),
                    ],
                    dim=0,
                ),
                dim=0,
            )
        lse_for_o = lse_for_o.clone()
        lse_for_o[lse_for_o == float("-inf")] = float("+inf")
        probs = torch.exp(scores - lse_for_o.unsqueeze(-1))
        out = probs @ gathered_kv[..., : self.head_dim]
        lonely_q_mask = orig_lse == float("-inf")
        out[lonely_q_mask.unsqueeze(-1).expand_as(out)] = 0.0
        return out.to(torch.bfloat16)


class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # head_dim already carries the fp8 scale padding
        # compress_ratio=1 for V3.2, >1 for DeepseekV4; both use the same cache layout.
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
            # DeepseekV4 aligns indexer pages to FlashMLA's 576B so they can pack with
            # the indexer's compressor state cache. V3.2 keeps the legacy layout.
            alignment=576,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4IndexerBackend


class DeepseekV4Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        compress_ratio: int = 1,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        self.compress_ratio = compress_ratio
        self.use_fp4_kv = self.vllm_config.attention_config.use_fp4_indexer_cache
        logger.info_once(
            "Using %s indexer cache for Lighening Indexer.",
            "MXFP4" if self.use_fp4_kv else "FP8",
        )

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        self.max_model_len = (
            vllm_config.model_config.max_model_len // self.compress_ratio
        )
        self.prefix = prefix

        self.max_total_seq_len = (
            get_max_prefill_buffer_size(vllm_config) // self.compress_ratio
        )

        assert cache_config is not None, "Deepseek V4 indexer requires cache_config"
        # NOTE(yifan): FP8 indxer cache use the same layout as V3.2:
        # head_dim bytes = 128 fp8 + 4 fp32 scale = 132.
        # For FP4 indexer cache, we still allocate the same amount of memory as FP8,
        # but only use the first half of the memory.
        k_cache_head_dim = self.head_dim + self.head_dim // self.quant_block_size * 4
        self.k_cache = DeepseekV4IndexerCache(
            head_dim=k_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
        )
        self.compressor = DeepseekCompressor(
            vllm_config=vllm_config,
            compress_ratio=self.compress_ratio,
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
            k_cache_prefix=self.k_cache.prefix,
            use_fp4_cache=self.use_fp4_kv,
        )

        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            skip_k_cache_insert=True,
            use_fp4_cache=self.use_fp4_kv,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        k = self.compressor(hidden_states, positions, rotary_emb)
        weights, _ = self.weights_proj(hidden_states)
        q_quant, weights = fused_indexer_q_rope_quant(
            positions,
            q,
            rotary_emb.cos_sin_cache,
            weights,
            self.softmax_scale,
            self.n_head**-0.5,
            use_fp4=self.use_fp4_kv,
        )
        return self.indexer_op(hidden_states, q_quant, k, weights)
