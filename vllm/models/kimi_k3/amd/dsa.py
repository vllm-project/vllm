# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA (DeepSeek Sparse Attention) for the kimi_k3 stack on ROCm.

Ports ``nvidia/dsa.py``: the indexer runs through
``SparseAttnIndexer.forward_hip`` (the DeepSeek V3.2 ROCm pattern) and the
sparse MLA layers are served by ROCM_AITER_MLA_SPARSE over the fp8_ds_mla
cache. The DeepGEMM-only 32/64 index-head zero padding is dropped: the ROCm
indexer kernels take the native head count.
"""

import functools
import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_dcp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mla import (
    MLAModules,
    MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.sparse_attn_indexer import (
    SparseAttnIndexer,
    _merge_dcp_topk_global,
)
from vllm.models.kimi_k3.common.dsa_cache import KimiK3IndexerCache
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.mla import indexer as v32_indexer
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    indexer_k_quant_and_cache_triton,
    rocm_aiter_sparse_attn_indexer,
    rocm_fp8_paged_mqa_logits,
)

from . import akr_dsa
from .ops.fp4_indexer import (
    decode_layout,
    decode_tile_size,
    quantize_queries_and_weights,
    score_paged,
    store_keys,
    store_keys_and_reset,
)
from .ops.fp4_indexer_prefill import prefill_chunks, prefill_workspace_bytes

logger = init_logger(__name__)

# Largest batch for the fused q_a/kv_a RMSNorm (MI355X crossover vs. AITER is
# around 2K tokens).
_FUSED_QKV_NORM_MAX_TOKENS = 1024


@triton.jit
def _index_weights_kernel(
    values,
    scales,
    output,
    value_stride,
    weight_scale,
    softmax_scale,
    NUM_HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.arange(0, BLOCK)
    value = tl.load(values + row * value_stride + head, head < NUM_HEADS, 0)
    # Preserve the projection dtype's rounding before the FP32 scale products.
    value = (value.to(tl.float32) * weight_scale).to(values.dtype.element_ty)
    scale = tl.load(scales + row * NUM_HEADS + head, head < NUM_HEADS, 0)
    weight = value.to(tl.float32) * scale * softmax_scale
    tl.store(output + row * NUM_HEADS + head, weight, head < NUM_HEADS)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class KimiK3DSAIndexerKNorm(nn.Module):
    """Bias-full fp32 LayerNorm for the DSA indexer shared key.

    weight/bias load and stay in fp32; a copy in the activation dtype is cached
    so the fp32 round trip does not cost two casts per call. F.layer_norm still
    accumulates in fp32 for a bf16 input, so the only numeric change is that
    weight/bias multiply in the activation dtype.
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

        # Not persistent: derived from weight/bias rather than loaded from a
        # checkpoint, so state_dict() keeps exporting fp32.
        self.weight_bf16: torch.Tensor | None
        self.bias_bf16: torch.Tensor | None
        self.register_buffer("weight_bf16", None, persistent=False)
        self.register_buffer("bias_bf16", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight, bias = self.weight_bf16, self.bias_bf16
        if weight is None or weight.dtype != x.dtype:
            weight = self.weight.to(x.dtype)
            bias = self.bias.to(x.dtype)
            # nn.Module.__setattr__ is expensive, and mutating a buffer during
            # tracing would bake this branch into the captured graph.
            if not torch.compiler.is_compiling():
                self.weight_bf16, self.bias_bf16 = weight, bias
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)


@eager_break_during_capture
def _indexer(
    hidden: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cache: torch.Tensor,
    out: torch.Tensor,
    scores: torch.Tensor,
    prefix: str,
    max_model_len: int,
    max_keys: int,
    rank: int,
    world: int,
    interleave: int,
) -> None:
    metadata = get_forward_context().attn_metadata
    m = (
        cast(DeepseekV32IndexerMetadata, metadata[prefix])
        if isinstance(metadata, dict)
        else None
    )
    if m is None or m.num_prefills:
        rocm_aiter_sparse_attn_indexer(
            hidden,
            prefix,
            cache,
            q,
            k,
            weights,
            128,
            "ue8m0",
            out.shape[1],
            128,
            max_model_len,
            max_keys,
            out,
            cache_layout="NORMAL",
            dcp_rank=rank,
            dcp_world_size=world,
            cp_interleave=interleave,
        )
        return

    rows = m.num_decode_tokens
    assert rows <= scores.shape[0]
    indexer_k_quant_and_cache_triton(
        k[: m.slot_mapping.numel()],
        cache,
        m.slot_mapping,
        128,
        "ue8m0",
        cache_layout="NORMAL",
    )
    # Tracking already prepares causal token rows once in the shared builder.
    decode = m.decode
    assert decode is not None and not decode.requires_padding
    assert decode.seq_lens.numel() == rows
    assert decode.block_table.shape[0] == rows
    ends = decode.seq_lens.reshape(-1)
    width = (
        (max_model_len + world * interleave - 1) // (world * interleave)
    ) * interleave
    logits = rocm_fp8_paged_mqa_logits(
        q[:rows, None],
        cache.unsqueeze(-2),
        weights[:rows],
        decode.seq_lens,
        decode.block_table,
        decode.schedule_metadata,
        max_model_len=width,
        cache_layout="NORMAL",
    )
    rocm_aiter_ops.indexer_top_k_decode(
        logits,
        1,
        ends,
        out[:rows],
        out.shape[1],
        stable=True,
        values=scores[:rows],
    )
    out[rows : q.shape[0]].fill_(-1)


def _fake(
    hidden: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cache: torch.Tensor,
    out: torch.Tensor,
    scores: torch.Tensor,
    prefix: str,
    max_model_len: int,
    max_keys: int,
    rank: int,
    world: int,
    interleave: int,
) -> None:
    pass


direct_register_custom_op(
    op_name="kimi_dcp_score_indexer",
    op_func=_indexer,
    mutates_args=["cache", "out", "scores"],
    fake_impl=_fake,
)


@functools.cache
def _sampled_buffers(device, rows, width, k):
    """Zero row starts and scratch for AITER's sampled top-k, kept for every
    graph: Phase A clears what it shares before anything reads it. One per
    shape on a device: two selects of a shape must not run concurrently
    (indexers on two streams at once, e.g. overlapped micro-batches)."""
    from aiter.ops.topk import topk_sampled_workspace_size
    from aiter.utility.graph_alloc import persistent_alloc

    with persistent_alloc(device):
        starts = torch.zeros(rows, dtype=torch.int32, device=device)
        scratch = torch.empty(
            topk_sampled_workspace_size(rows, width, k),
            dtype=torch.uint8,
            device=device,
        )
    return starts, scratch


@functools.cache
def _sampled_serves(rows, width, k, device_index):
    from aiter.ops.topk import topk_sampled_supports

    # The shapes AITER measured `sampled` ahead on: few rows, wide pitch.
    return (
        rows <= 64
        and width >= 131072
        and 512 <= k <= 2048
        and bool(topk_sampled_supports(rows, width, k, device_index))
    )


def _select(logits, lengths, out, values=None):
    from vllm.models.kimi_k3.amd.specialized_decode.levers import (
        SAMPLED_TOPK,
        lever_on,
    )

    rows, width = logits.shape
    k = out.shape[1]
    if (
        values is None
        and lever_on(SAMPLED_TOPK)
        and logits.stride(1) == 1
        and _sampled_serves(rows, logits.stride(0), k, logits.get_device())
    ):
        from aiter.ops.topk import top_k_per_row_prefill_sampled

        starts, scratch = _sampled_buffers(logits.device, rows, logits.stride(0), k)
        top_k_per_row_prefill_sampled(
            logits,
            starts,
            lengths,
            out,
            None,
            rows,
            logits.stride(0),
            1,
            k,
            workspace=scratch,
        )
    elif values is not None:
        from aiter.ops.topk import top_k_per_row_decode

        top_k_per_row_decode(
            logits, 1, lengths, out, rows, *logits.stride(), k, values=values
        )
    elif rocm_aiter_ops.is_indexer_top_k_supported(
        is_prefill=False,
        compress_ratio=1,
        num_rows=rows,
        max_valid_seq_len=width,
        topk_tokens=k,
    ):
        rocm_aiter_ops.indexer_top_k_decode(logits, 1, lengths, out, k)
    else:
        torch.ops._C.top_k_per_row_decode(
            logits, 1, lengths, out, rows, logits.stride(0), logits.stride(1), k
        )


@eager_break_during_capture
def _fp4_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cache: torch.Tensor,
    out: torch.Tensor,
    prefix: str,
    max_model_len: int,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    interleave: int = 1,
    weight_scale: float = 1.0,
    scores: torch.Tensor | None = None,
    fused_prep: bool = False,
    k_norm_weight: torch.Tensor | None = None,
    k_norm_bias: torch.Tensor | None = None,
    k_norm_eps: float = 0.0,
    q_payload: torch.Tensor | None = None,
    q_scales: torch.Tensor | None = None,
    q_weights: torch.Tensor | None = None,
) -> None:
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        from vllm.config import get_current_vllm_config
        from vllm.v1.attention.backends.mla.indexer import (
            get_max_prefill_buffer_size,
        )

        config = get_current_vllm_config()
        _workspace = torch.empty(
            prefill_workspace_bytes(
                get_max_prefill_buffer_size(config),
                config.scheduler_config.max_num_seqs,
                dcp_world_size,
                interleave,
            ),
            device=q.device,
            dtype=torch.uint8,
        )
        return
    m = metadata[prefix]
    assert isinstance(m, DeepseekV32IndexerMetadata)
    # q_payload: the caller (the specialized decode's D1a) already stored the
    # keys, reset the top-k rows and quantized the query (MXFP4, with its FP32
    # head weights times weight_scale) for one tile of every decode row.
    prepped = q_payload is not None
    if prepped:
        assert q_scales is not None and q_weights is not None
        # pure decode, every key and decode row the caller's; padding rows
        # past them still get the reset
        # (graph-padded: the caller's rows may exceed the decode rows and the
        # slot mapping, and its prep already reset every row's top-k)
        rows = q_payload.shape[0]
        assert not m.num_prefills and m.num_decode_tokens <= rows <= q.shape[0]
        assert m.slot_mapping.numel() <= rows
        if q.shape[0] > rows:
            out[rows : q.shape[0]].fill_(-1)
    elif fused_prep:
        # k is the raw projection when k_norm_weight is set.
        store_keys_and_reset(
            k[: m.slot_mapping.numel()],
            cache,
            m.slot_mapping,
            out[: q.shape[0]],
            k_norm_weight,
            k_norm_bias,
            k_norm_eps,
        )
    else:
        store_keys(k[: m.slot_mapping.numel()], cache, m.slot_mapping)
        out[: q.shape[0]].fill_(-1)
    if m.num_prefills:
        assert m.prefill is not None
        prefill_chunks(
            q,
            weights,
            cache,
            out,
            m.prefill,
            dcp_world_size,
            interleave,
            weight_scale,
        )
    if m.num_decodes:
        decode = m.decode
        block_table = m.block_table
        assert decode is not None and block_table is not None
        if (
            decode.query_start_loc is not None
            and not decode.requires_padding
            and decode.seq_lens.numel() == m.num_decode_tokens
        ):
            # The builder already expands and DCP-localizes each causal bound.
            starts = decode.query_start_loc
            ends = decode.seq_lens.reshape(-1)
        else:
            starts, ends = decode_layout(
                m.seq_lens,
                decode.per_req_decode_lens,
                m.num_decode_tokens,
                dcp_rank,
                dcp_world_size,
                interleave,
            )
        tile_rows = decode_tile_size(
            max_model_len, block_table.shape[1] * 64, out.shape[1], q.shape[1]
        )
        score_rows = scores if not m.num_prefills else None
        for start in range(0, m.num_decode_tokens, tile_rows):
            end = min(start + tile_rows, m.num_decode_tokens)
            row_ends = ends[start:end]
            quantized = None
            # amd_kernels: scorer + select in one launch (a single tile, no prefill)
            akr_fused = (
                score_rows is not None
                and start == 0
                and end == m.num_decode_tokens
                and akr_dsa.score_topk_supported(
                    starts.numel() - 1, block_table.shape[1] * 64, cache, out
                )
            )
            if prepped and start == 0 and end <= q_payload.shape[0]:
                # the starts are decode rows' (0 .. num_decode_tokens): the clamp
                # is a no-op unless this first tile ends before the last row
                quantized = (q_payload[:end], q_scales[:end])
                tile_weights = q_weights[:end]
                rel_starts = (
                    starts if end == m.num_decode_tokens else starts.clamp(0, end)
                )
            elif fused_prep or akr_fused:
                q_payload, q_scales, tile_weights, rel_starts = (
                    quantize_queries_and_weights(
                        q[start:end],
                        weights[start:end],
                        weight_scale,
                        starts,
                        start,
                        end - start,
                    )
                )
                quantized = (q_payload, q_scales)
            else:
                tile_weights = weights[start:end].float() * weight_scale
                rel_starts = starts if start == 0 else starts - start
                rel_starts = rel_starts.clamp(0, end - start)
            max_qlen = min(end - start, decode.write_max_decode_len or end - start)
            selected = out[start:end]
            if akr_fused:
                assert quantized is not None
                if akr_dsa.score_topk(
                    quantized[0],
                    quantized[1],
                    tile_weights,
                    cache,
                    block_table[: starts.numel() - 1],
                    rel_starts,
                    row_ends,
                    max_qlen,
                    selected,
                    score_rows[start:end],
                ):
                    continue
            logits = score_paged(
                q[start:end],
                cache,
                tile_weights,
                block_table[: starts.numel() - 1],
                rel_starts,
                row_ends,
                max_qlen,
                quantized,
            )
            if score_rows is not None:
                # The sparse backend merges scores directly into local CSR slots.
                _select(logits, row_ends, selected, score_rows[start:end])
            else:
                _select(logits, row_ends, selected)
                _merge_dcp_topk_global(
                    logits, selected, out.shape[1], dcp_rank, dcp_world_size, interleave
                )


def _fp4_indexer_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cache: torch.Tensor,
    out: torch.Tensor,
    prefix: str,
    max_model_len: int,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    interleave: int = 1,
    weight_scale: float = 1.0,
    scores: torch.Tensor | None = None,
    fused_prep: bool = False,
    k_norm_weight: torch.Tensor | None = None,
    k_norm_bias: torch.Tensor | None = None,
    k_norm_eps: float = 0.0,
    q_payload: torch.Tensor | None = None,
    q_scales: torch.Tensor | None = None,
    q_weights: torch.Tensor | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="kimi_fp4_indexer",
    op_func=_fp4_indexer,
    mutates_args=["cache", "out", "scores"],
    fake_impl=_fp4_indexer_fake,
)


class KimiK3DSAIndexer(nn.Module):
    """wq/wk/wv indexer projections with FP8 or MXFP4 scoring on ROCm.

    Forward quantizes the indexer query, folds the scales into the per-head
    weights and drives ``torch.ops.vllm.rocm_aiter_sparse_attn_indexer`` via
    ``SparseAttnIndexer.forward_hip`` with ``cache_layout="NORMAL"``: the
    indexer pages are block-flat, which the Triton insert wrapper cannot infer
    from the cache shape alone. MXFP4 uses group scales and dedicated FlyDSL
    scoring with the same global token-index output contract.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        hidden_size: int,
        index_q_input_size: int,
        index_head_dim: int,
        index_num_heads: int,
        index_topk: int,
        rms_norm_eps: float,
        index_k_norm_type: str = "rmsnorm",
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.index_head_dim = index_head_dim
        self.index_num_heads = index_num_heads
        self.indexer_weight_scale = self.index_num_heads**-0.5
        self.use_fp4 = v32_indexer.dsa_indexer_uses_fp4(vllm_config)
        # MXFP4 decode prep in the indexer's Triton quantize launches.
        self.fused_prep = self.use_fp4 and envs.VLLM_K3_FUSED_INDEXER_PREP
        if self.use_fp4 and (
            index_head_dim != 128 or index_num_heads not in (16, 32, 64)
        ):
            raise ValueError("Kimi MXFP4 requires head_dim=128 and 16/32/64 heads")
        self.cp_interleave = vllm_config.parallel_config.cp_kv_cache_interleave_size
        # MLAModules contract: MultiHeadLatentAttention asserts the indexer
        # exposes topk_tokens.
        self.topk_tokens = index_topk
        self.wq_index = ReplicatedLinear(
            index_q_input_size,
            self.index_num_heads * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_index",
        )
        # The indexer query projection (wq_index) may be quantized in fp8
        # checkpoints, while the shared-key (wk_index) and per-head weights
        # (wv_index) projections are stored in bf16.
        self.wk_index = ReplicatedLinear(
            hidden_size,
            self.index_head_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.wk_index",
        )
        self.wv_index = ReplicatedLinear(
            hidden_size,
            self.index_num_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.wv_index",
        )
        self.k_index_norm: nn.Module
        if index_k_norm_type == "layernorm":
            self.k_index_norm = KimiK3DSAIndexerKNorm(
                self.index_head_dim, eps=rms_norm_eps
            )
        elif index_k_norm_type == "rmsnorm":
            self.k_index_norm = RMSNorm(self.index_head_dim, eps=rms_norm_eps)
        else:
            raise ValueError(
                f"dsa_config.index_k_norm_type must be 'rmsnorm' or "
                f"'layernorm', got {index_k_norm_type!r}"
            )

        # Index cache scales use ue8m0 (power-of-2 rounding at write), the
        # convention of upstream DeepSeek V3.2; the scoring kernels read the
        # scales as full fp32, which is exact for power-of-2 values.
        self.scale_fmt = "ue8m0"
        self.quant_block_size = self.index_head_dim
        self.index_softmax_scale = self.index_head_dim**-0.5
        # FP8: 128 values + one FP32 scale. MXFP4: 64 packed value bytes
        # + four UE8M0 group scales. Both formats add four scale bytes.
        self.k_cache = KimiK3IndexerCache(
            head_dim=(self.index_head_dim // 2 if self.use_fp4 else self.index_head_dim)
            + 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cast(CacheConfig, cache_config),
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_total_seq_len = v32_indexer.get_max_prefill_buffer_size(vllm_config)
        self.topk_indices_buffer = topk_indices_buffer
        self.dcp_scores = None
        if vllm_config.parallel_config.decode_context_parallel_size > 1:
            assert topk_indices_buffer is not None
            spec = vllm_config.speculative_config
            rows = max(
                vllm_config.scheduler_config.max_num_seqs
                * (1 + (spec.num_speculative_tokens if spec else 0)),
                vllm_config.compilation_config.max_cudagraph_capture_size or 0,
            )
            self.dcp_scores = torch.empty(
                (rows, self.topk_tokens),
                dtype=torch.float32,
                device=topk_indices_buffer.device,
            )
        self.indexer_op = (
            None
            if self.use_fp4
            else SparseAttnIndexer(
                self.k_cache,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.index_head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                topk_indices_buffer,
                cache_layout="NORMAL",
            )
        )

    def _quantize_index_q_and_weights(
        self,
        index_q: torch.Tensor,
        index_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FP8-quantize the indexer query and fold scales into the weights.

        Fold the query, head-count and softmax scales into FP32 weights,
        preserving the projection dtype's intermediate rounding.
        """
        q_flat = index_q.reshape(-1, self.index_head_dim).contiguous()
        q_fp8, q_scale = per_token_group_quant_fp8(
            q_flat,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=True,
        )
        q_fp8 = q_fp8.view(index_q.shape)
        weights = torch.empty(
            (index_q.shape[0], self.index_num_heads),
            dtype=torch.float32,
            device=index_q.device,
        )
        if index_q.shape[0]:
            _index_weights_kernel[(index_q.shape[0],)](
                index_weights,
                q_scale,
                weights,
                index_weights.stride(0),
                self.indexer_weight_scale,
                self.index_softmax_scale,
                NUM_HEADS=self.index_num_heads,
                BLOCK=triton.next_power_of_2(self.index_num_heads),
            )
        return q_fp8, weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        index_q_input: torch.Tensor | None,
        positions: torch.Tensor | None = None,
        rotary_emb: nn.Module | None = None,
    ) -> torch.Tensor:
        if index_q_input is None:
            # q_lora_rank is None: MultiHeadLatentAttention.forward_native
            # passes q_c=None, so the indexer query reads hidden_states.
            index_q_input = hidden_states
        index_q = self.wq_index(index_q_input)[0].view(
            -1, self.index_num_heads, self.index_head_dim
        )
        index_k = self.wk_index(hidden_states)[0]
        index_v = self.wv_index(hidden_states)[0]
        if self.use_fp4:
            # The fused key store applies the LayerNorm itself.
            k_norm = self.k_index_norm
            fuse_norm = self.fused_prep and isinstance(k_norm, KimiK3DSAIndexerKNorm)
            if not fuse_norm:
                index_k = k_norm(index_k)
            dcp = get_dcp_group()
            torch.ops.vllm.kimi_fp4_indexer(
                index_q,
                index_k,
                index_v,
                self.k_cache.kv_cache,
                self.topk_indices_buffer,
                self.k_cache.prefix,
                self.max_model_len,
                dcp.rank_in_group,
                dcp.world_size,
                self.cp_interleave,
                self.indexer_weight_scale * self.index_softmax_scale,
                self.dcp_scores,
                self.fused_prep,
                k_norm.weight if fuse_norm else None,
                k_norm.bias if fuse_norm else None,
                k_norm.eps if fuse_norm else 0.0,
            )
            return self.topk_indices_buffer
        index_k = self.k_index_norm(index_k)
        q_fp8, weights = self._quantize_index_q_and_weights(index_q, index_v)
        # Under a dummy run kv_cache is an empty placeholder and the op takes
        # its workspace-reservation path.
        indexer_op = self.indexer_op
        assert indexer_op is not None
        if self.dcp_scores is not None:
            torch.ops.vllm.kimi_dcp_score_indexer(
                hidden_states,
                q_fp8,
                index_k,
                weights,
                self.k_cache.kv_cache,
                self.topk_indices_buffer,
                self.dcp_scores,
                self.k_cache.prefix,
                self.max_model_len,
                self.max_total_seq_len,
                indexer_op.dcp_rank,
                indexer_op.dcp_world_size,
                indexer_op._parallel_config.cp_kv_cache_interleave_size,
            )
            return self.topk_indices_buffer
        return indexer_op.forward_hip(hidden_states, q_fp8, index_k, weights)


# ---------------------------------------------------------------------------
# Multi-stream DSA decode (VLLM_ROCM_DSA_MULTISTREAM=1).
#
# Layout per DSA layer (FULL-graph / eager decode only, DCP > 1, MXFP4):
#   main: [fork] fused_qkv_a -> q_a/kv_a norms -> [ev_q] q_b -> Q all-gather
#         -> [wait ev_merge] sparse MLA -> LSE combine -> [wait ev_done] gate*o
#   aux : wk+LN, wv | wait ev_q: wq -> FP4 indexer -> score AG (own comm)
#         -> top-k merge -> [ev_merge] g_proj -> [ev_done]
# One fork and one join per layer, on the shared vLLM aux stream.
# ---------------------------------------------------------------------------

_MS_LAYERS: dict[str, "KimiK3DSAMultiStreamMLAWrapper"] = {}
# None: not created yet; False: unavailable; else the aux AITER comm.
_MS_COMM: object = None


def _dsa_ms_comm(rows: int, topk: int):
    """One process-wide DCP-group AITER comm for the aux-stream score gather.

    It has its own IPC buffers and signals, so it can run concurrently with
    the main stream's DCP/TP collectives on the TP group's comm. It joins every
    ``graph_capture()`` window to register its captured buffers.
    """
    global _MS_COMM
    if _MS_COMM is not None:
        return _MS_COMM or None
    from vllm.distributed.parallel_state import register_graph_capture_comm
    from vllm.platforms import current_platform

    dcp = get_dcp_group()
    comm = None
    if current_platform.is_rocm() and dcp.world_size > 1:
        from vllm.distributed.device_communicators.aiter_custom_all_reduce import (
            AiterCustomAllreduce,
        )

        config = get_current_vllm_config()
        # Size for decode batches only (larger ones fall back to the serial
        # path via should_custom_ag): dcp_scores is sized by the max capture
        # size, which can be the max batched tokens.
        spec = config.speculative_config
        rows = min(
            rows,
            config.scheduler_config.max_num_seqs
            * (1 + (spec.num_speculative_tokens if spec else 0)),
        )
        # all_gather per-rank input must fit max_size / (2 * world).
        need = 2 * dcp.world_size * rows * topk * 4
        mib = 1 << 20
        max_size = max(16 * mib, -(-need // mib) * mib)
        comm = AiterCustomAllreduce(
            dcp.cpu_group,
            dcp.device,
            max_size=max_size,
            register_graph_buffers=not config.use_cumem_cudagraph_pool,
        )
        if comm.disabled:
            comm = None
    if comm is None:
        logger.warning(
            "VLLM_ROCM_DSA_MULTISTREAM: no DCP AITER comm available; the DSA "
            "layers keep the serial indexer path."
        )
        _MS_COMM = False
        return None
    register_graph_capture_comm(comm)
    logger.info(
        "VLLM_ROCM_DSA_MULTISTREAM: created the aux-stream DCP AITER comm "
        "(max_size=%d MiB).",
        max_size >> 20,
    )
    _MS_COMM = comm
    return comm


class _MSState:
    """Per-layer fork/join bookkeeping between the three custom ops."""

    def __init__(self) -> None:
        self.ev_fork = torch.cuda.Event()
        self.ev_q = torch.cuda.Event()
        self.ev_merge = torch.cuda.Event()
        self.ev_done = torch.cuda.Event()
        self.ev_qg = torch.cuda.Event()
        # 0: idle, 1: forked (indexer op pending), 2: merged (attn pending).
        self.stage = 0
        self.rows = 0
        self.kv: tuple[torch.Tensor, torch.Tensor] | None = None
        self.pending: tuple[torch.Tensor, torch.Tensor, int] | None = None
        self.joined = True
        self.reset_akr()

    def reset_akr(self) -> None:
        """amd_kernels hand-offs of one layer pass (see ``akr_dsa``)."""
        # the prologue kernels do this pass's KV / indexer-key inserts and prep
        self.akr_pro = False
        self.akr_qg = False
        self.ev_q_done = False
        self.skip_kv_update = False
        # gathered latent query for the sparse MLA; aux gate for the combine
        self.q_lat: torch.Tensor | None = None
        self.gate: torch.Tensor | None = None
        self.folded = False

    def take_pending(self, rows: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Called by the sparse MLA backend: hand over the merged CSR slots.

        Makes the current stream wait for the aux merge. ``None`` means the
        caller runs its serial score gather + merge (the scores are final).
        """
        if self.stage == 1:
            raise RuntimeError(
                "DSA multi-stream: attention ran before the indexer op; "
                "the op order was changed."
            )
        pending = self.pending
        if pending is None:
            return None
        self.pending = None
        self.stage = 0
        torch.cuda.current_stream().wait_event(self.ev_merge)
        packed, indptr, n = pending
        if n != rows:
            return None
        return packed, indptr


def _ms_active(layer: "KimiK3DSAMultiStreamMLAWrapper") -> bool:
    """Shape/mode-only decision: identical on every DCP rank."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import is_forward_context_available

    comm = layer.ms_comm
    ind = layer.indexer
    if comm is None or BreakableCUDAGraphCapture.is_active():
        return False
    if not is_forward_context_available():
        return False
    ctx = get_forward_context()
    if ctx.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
        return False
    md = ctx.attn_metadata
    if not isinstance(md, dict):
        return False
    im = md.get(ind.k_cache.prefix)
    mm = md.get(layer.mla_attn.layer_name)
    if im is None or mm is None:
        return False
    if im.num_prefills or not im.num_decodes:
        return False
    if mm.num_prefills or not mm.num_decode_tokens:
        return False
    rows = int(mm.num_decode_tokens)
    if rows > ind.dcp_scores.shape[0] or rows > im.num_decode_tokens:
        return False
    if not comm.should_custom_ag(ind.dcp_scores[:rows].view(torch.int32)):
        return False
    layer.ms_state.rows = rows
    return True


def _ms_fork(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    layer = _MS_LAYERS[layer_name]
    st = layer.ms_state
    st.stage = 0
    st.pending = None
    st.kv = None
    st.reset_akr()
    if _ms_active(layer):
        ind = layer.indexer
        aux = layer.aux_stream
        rows = x.shape[0]
        if layer.akr_pro:
            st.akr_pro = akr_dsa.prologue_runtime_ok(
                layer, *_akr_prologue_ctx(layer, rows)
            )
        if layer.akr_qg:
            st.akr_qg = akr_dsa.qgather_runtime_ok(layer, rows)
        if not st.akr_pro:
            # (the prologue kernel does the key side itself)
            st.ev_fork.record()
            with torch.cuda.stream(aux):
                aux.wait_event(st.ev_fork)
                ik = ind.k_index_norm(ind.wk_index(x)[0])
                iv = ind.wv_index(x)[0]
            st.kv = (ik, iv)
        st.stage = 1
    return x.new_empty(0)


def _akr_prologue_ctx(layer, rows: int):
    """``(rows, table_width, kv_cache, kv_slots, index cache, index slots)``."""
    ind = layer.indexer
    md = get_forward_context().attn_metadata
    im = md[ind.k_cache.prefix]
    _, _, kv_cache, kv_slots = get_attention_context(layer.mla_attn.layer_name)
    return (
        rows,
        im.block_table.shape[1] * 64 if im.block_table is not None else 0,
        kv_cache,
        kv_slots,
        ind.k_cache.kv_cache,
        im.slot_mapping,
    )


def _ms_prologue(
    qkv: torch.Tensor, token: torch.Tensor, layer_name: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """q_a / kv_a norms of the (extended) qkv_a output: the amd_kernels K_A
    (also the KV + indexer-key inserts and the head weights) when this pass is a
    multi-stream decode, else the stock norms."""
    layer = _MS_LAYERS[layer_name]
    st = layer.ms_state
    if st.stage == 1 and st.akr_pro:
        _, _, kv_cache, kv_slots = get_attention_context(layer.mla_attn.layer_name)
        im = get_forward_context().attn_metadata[layer.indexer.k_cache.prefix]
        q_c, kv_c, idx_w = akr_dsa.prologue_a(
            layer,
            qkv,
            kv_cache,
            kv_slots,
            layer.indexer.k_cache.kv_cache,
            im.slot_mapping,
        )
        st.skip_kv_update = True  # the MLA layer must not insert the KV again
        st.ev_q.record()  # q_c and the key are ready for the aux stream
        st.ev_q_done = True
        return q_c, kv_c, idx_w
    q_c = layer.q_a_layernorm(qkv[:, : layer.q_lora_rank])
    kv_c = layer.kv_a_layernorm(
        qkv[:, layer.q_lora_rank : layer.q_lora_rank + layer.kv_lora_rank]
    )
    return (
        q_c,
        kv_c,
        qkv.new_empty(
            (qkv.shape[0], layer.indexer.index_num_heads), dtype=torch.float32
        ),
    )


def _ms_prologue_fake(
    qkv: torch.Tensor, token: torch.Tensor, layer_name: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layer = _MS_LAYERS[layer_name]
    m = qkv.shape[0]
    return (
        qkv.new_empty((m, layer.q_lora_rank)),
        qkv.new_empty((m, layer.kv_lora_rank)),
        qkv.new_empty((m, layer.indexer.index_num_heads), dtype=torch.float32),
    )


def _ms_q(
    q_c: torch.Tensor, token: torch.Tensor, layer_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """The query projection: the amd_kernels q_b + W_UK + DCP gather (the gathered
    latent query goes to the sparse MLA through the layer state; the returned q is
    then unused) when this pass is a multi-stream decode, else the stock q_b."""
    layer = _MS_LAYERS[layer_name]
    st = layer.ms_state
    if st.stage == 1 and st.akr_qg:
        if not st.ev_q_done:
            # q_c is ready: the aux stream starts now, not behind the gather
            st.ev_q.record()
            st.ev_q_done = True
        st.q_lat = akr_dsa.qgather(layer, q_c)
        # the aux-stream score gather may start only once this kernel is done
        # (two spinning kernels must not wait for each other's residency)
        st.ev_qg.record()
        q = q_c.new_empty((q_c.shape[0], layer.num_heads, layer.qk_head_dim))
    else:
        q = layer.q_b_proj(q_c)[0].view(-1, layer.num_heads, layer.qk_head_dim)
    return q, q_c.new_empty(0)


def _ms_q_fake(
    q_c: torch.Tensor, token: torch.Tensor, layer_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = _MS_LAYERS[layer_name]
    return (
        q_c.new_empty((q_c.shape[0], layer.num_heads, layer.qk_head_dim)),
        q_c.new_empty(0),
    )


def _ms_fork_fake(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    return x.new_empty(0)


def _ms_indexer(
    x: torch.Tensor,
    q_c: torch.Tensor,
    token: torch.Tensor,
    layer_name: str,
    idx_w: torch.Tensor,
    q_token: torch.Tensor,
) -> torch.Tensor:
    layer = _MS_LAYERS[layer_name]
    st = layer.ms_state
    ind = layer.indexer
    if st.stage != 1:
        # Serial reference path (prefill, mixed, PIECEWISE, profile runs).
        ind(x, q_c)
        st.joined = True
        return layer.g_proj(x)[0]
    from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as be

    if st.akr_pro:
        ik = iv = None
    else:
        assert st.kv is not None
        ik, iv = st.kv
    st.kv = None
    rows = st.rows
    dcp = get_dcp_group()
    topk = ind.topk_indices_buffer
    scores = ind.dcp_scores
    _, mla_layer, kv_cache, _ = get_attention_context(layer.mla_attn.layer_name)
    mm = get_forward_context().attn_metadata[layer.mla_attn.layer_name]
    _, block_stride_rows = flat_kv_row_view(kv_cache, mm.block_size)
    width = dcp.world_size * topk.shape[1]

    def merge_inputs():
        # The merge's per-row page table and key-width vector.
        ends = be._score_widths.get((scores.device, width))
        if ends is None or ends.numel() < rows:
            be._score_widths[(scores.device, width)] = torch.full(
                (max(rows, 1024),), width, dtype=torch.int32, device=scores.device
            )
        token_pages = getattr(mm, "_dcp_token_pages", None)
        if token_pages is None or token_pages.shape[0] != rows:
            token_pages = mm.block_table[mm.req_id_per_token[:rows]]
            mm._dcp_token_pages = token_pages
        return token_pages

    aux = layer.aux_stream
    token_pages = None
    if not st.ev_q_done:
        # Created on main before the fork so both streams see it filled. (After the
        # prologue the fork event is already recorded and the aux stream must not
        # wait for the query gather behind it: it builds them itself.)
        token_pages = merge_inputs()
        st.ev_q.record()
    with torch.cuda.stream(aux):
        aux.wait_event(st.ev_q)
        if token_pages is None:
            token_pages = merge_inputs()
        index_q = ind.wq_index(q_c)[0].view(-1, ind.index_num_heads, ind.index_head_dim)
        if st.akr_pro:
            # K_B (query MXFP4 + top-k reset); the key and weights came from K_A
            q_payload, q_scales = akr_dsa.prologue_b(
                layer, index_q, topk, ind.k_cache.kv_cache
            )
            prepped = dict(q_payload=q_payload, q_scales=q_scales, q_weights=idx_w)
            ik = iv = index_q  # (unused by the prepped path)
        else:
            prepped = {}
        _fp4_indexer(
            index_q,
            ik,
            iv,
            ind.k_cache.kv_cache,
            topk,
            ind.k_cache.prefix,
            ind.max_model_len,
            dcp.rank_in_group,
            dcp.world_size,
            ind.cp_interleave,
            ind.indexer_weight_scale * ind.index_softmax_scale,
            scores,
            **prepped,
        )
        if st.q_lat is not None:
            # the query gather (main) and the score gather spin on their peers:
            # keep them apart
            aux.wait_event(st.ev_qg)
        merged = akr_dsa.gather_merge(
            scores[:rows], topk[:rows], token_pages, mm.block_size, block_stride_rows
        )
        if merged is not None:
            packed, indptr = merged
        else:
            gathered = layer.ms_comm.custom_all_gather(
                scores[:rows].view(torch.int32), dim=1
            ).view(torch.float32)
            packed, indptr = be.merge_dcp_scores(
                gathered,
                topk[:rows],
                token_pages,
                dcp.rank_in_group,
                dcp.world_size,
                mm.block_size,
                block_stride_rows,
            )
        st.ev_merge.record()
        gate = layer.g_proj(x)[0]
        st.ev_done.record()
    st.gate = gate
    st.pending = (packed, indptr, rows)
    st.stage = 2
    st.joined = False
    logger.info_once(
        "VLLM_ROCM_DSA_MULTISTREAM: DSA indexer + DCP score all-gather/merge "
        "+ gate run on the aux stream (decode, %d rows).",
        rows,
    )
    return gate


def _ms_indexer_fake(
    x: torch.Tensor,
    q_c: torch.Tensor,
    token: torch.Tensor,
    layer_name: str,
    idx_w: torch.Tensor,
    q_token: torch.Tensor,
) -> torch.Tensor:
    g_proj = _MS_LAYERS[layer_name].g_proj
    return x.new_empty((x.shape[0], g_proj.output_size_per_partition))


@triton.jit
def _gate_mul_kernel(a_ptr, g_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask, other=0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask, other=0).to(tl.float32)
    out = a * tl.sigmoid(g)
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask)


def _ms_join(
    attn_out: torch.Tensor,
    gate: torch.Tensor,
    x: torch.Tensor,
    q_c: torch.Tensor,
    idx_w: torch.Tensor,
    layer_name: str,
) -> None:
    """Join the aux stream and gate `attn_out` in place (the amd_kernels combine
    may have done both already). x, q_c and idx_w are inputs only to keep their
    main-stream buffers alive (and unreused) until the aux reads of them are joined
    here."""
    st = _MS_LAYERS[layer_name].ms_state
    if not st.joined:
        torch.cuda.current_stream().wait_event(st.ev_done)
        st.joined = True
    if st.stage == 2:
        # Attention did not consume the merge (no DCP decode rows).
        st.pending = None
        st.stage = 0
    st.gate = None
    if st.folded:
        st.folded = False
        return
    n = attn_out.numel()
    if n:
        if attn_out.is_contiguous() and gate.is_contiguous():
            _gate_mul_kernel[(triton.cdiv(n, 1024),)](
                attn_out, gate, attn_out, n, BLOCK=1024
            )
        else:
            attn_out.copy_(
                (attn_out.float() * gate.float().sigmoid()).to(attn_out.dtype)
            )


def _ms_join_fake(
    attn_out: torch.Tensor,
    gate: torch.Tensor,
    x: torch.Tensor,
    q_c: torch.Tensor,
    idx_w: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="kimi_dsa_ms_fork", op_func=_ms_fork, fake_impl=_ms_fork_fake
)
direct_register_custom_op(
    op_name="kimi_dsa_ms_prologue",
    op_func=_ms_prologue,
    fake_impl=_ms_prologue_fake,
)
direct_register_custom_op(op_name="kimi_dsa_ms_q", op_func=_ms_q, fake_impl=_ms_q_fake)
direct_register_custom_op(
    op_name="kimi_dsa_ms_indexer", op_func=_ms_indexer, fake_impl=_ms_indexer_fake
)
direct_register_custom_op(
    op_name="kimi_dsa_ms_join",
    op_func=_ms_join,
    mutates_args=["attn_out"],
    fake_impl=_ms_join_fake,
)


class KimiK3DSAMultiStreamMLAWrapper(MultiHeadLatentAttentionWrapper):
    """MLA wrapper whose decode runs the DSA indexer chain on the aux stream.

    Same modules and weights as the plain wrapper. Outside multi-stream decode
    the indexer op runs the serial chain; only the gate multiply differs (one
    fused fp32 kernel, as Inductor emits it).
    """

    def __init__(self, *args, ms_comm=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.q_lora_rank is not None and self.g_proj is not None
        assert self.rotary_emb is None and not self.dcp_q_replicate
        assert not self.fuse_qkv_rmsnorm and not self.skip_topk
        self.ms_comm = ms_comm
        self.ms_state = _MSState()
        self._aux_stream = None
        name = self.mla_attn.layer_name
        _MS_LAYERS[name] = self
        self.ms_name = name
        # amd_kernels: the prologue (extended qkv_a GEMM + K_A / K_B) and the
        # fused q_b + W_UK + query gather; the sparse MLA, the combine and the
        # score gather/merge live in their own call sites.
        self.akr_w: akr_dsa.LayerWeights | None = None
        self.akr_pro = akr_dsa.prologue_static_ok(self)
        self.akr_qg = akr_dsa.qgather_static_ok(self)
        self.mla_attn._akr_ms_state = self.ms_state

    @property
    def aux_stream(self) -> torch.cuda.Stream:
        if self._aux_stream is None:
            from vllm.utils.torch_utils import aux_stream

            self._aux_stream = aux_stream()
        return self._aux_stream

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
        skip_o_proj: bool = False,
    ) -> torch.Tensor:
        assert llama_4_scaling is None
        name = self.ms_name
        token = torch.ops.vllm.kimi_dsa_ms_fork(hidden_states, name)
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_end = self.q_lora_rank
        kv_end = q_end + self.kv_lora_rank
        if self.akr_pro:
            q_c, kv_c_normed, idx_w = torch.ops.vllm.kimi_dsa_ms_prologue(
                qkv_lora, token, name
            )
            k_pe = qkv_lora[:, kv_end : kv_end + self.qk_rope_head_dim]
        else:
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q_c = self.q_a_layernorm(q_c)
            kv_c, k_pe = kv_lora.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_c_normed = self.kv_a_layernorm(kv_c)
            idx_w = token
        k_pe = k_pe.unsqueeze(1)
        q_token = token
        if self.akr_qg:
            # before the indexer op: its aux-stream score gather waits for this one
            q, q_token = torch.ops.vllm.kimi_dsa_ms_q(q_c, token, name)
        gate = torch.ops.vllm.kimi_dsa_ms_indexer(
            hidden_states, q_c, token, name, idx_w, q_token
        )
        self.mla_attn.impl.record_logical_topk_ready()  # type: ignore[attr-defined]
        if not self.akr_qg:
            q = self.q_b_proj(q_c)[0].view(-1, self.num_heads, self.qk_head_dim)
        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
        )
        torch.ops.vllm.kimi_dsa_ms_join(attn_out, gate, hidden_states, q_c, idx_w, name)
        if skip_o_proj:
            return attn_out
        return self.o_proj(attn_out)[0]


def _akr_post_load(w: MultiHeadLatentAttentionWrapper) -> None:
    """amd_kernels weights of one DSA layer and the compile-only warm-up (called at
    the end of ``MLAAttention.process_weights_after_loading``)."""
    w.akr_w = akr_dsa.build_weights(w)  # type: ignore[attr-defined]
    if w.akr_w.w_uv_t is not None:  # type: ignore[attr-defined]
        w.mla_attn.akr_w_uv_t = w.akr_w.w_uv_t  # type: ignore[attr-defined]
    akr_dsa.warm_up(w, w.akr_w)  # type: ignore[attr-defined]


class KimiK3DSAAttention(nn.Module):
    """Kimi K3.1 DSA (DeepSeek Sparse Attention) inference path on ROCm.

    Same MLA projections as the dense Kimi-K3 MLA path, plus the DSA indexer
    wired through ``MLAModules(is_sparse=True)`` so the ROCM_AITER_MLA_SPARSE
    backend consumes the top-k indices.
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        reduce_results: bool = True,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if config.dsa_config is None:
            raise ValueError("dsa_config is required for KimiK3DSAAttention")
        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        if parallel_config.prefill_context_parallel_size > 1:
            raise NotImplementedError(
                "Kimi-K3.1 DSA on ROCm does not support prefill context parallelism."
            )
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size > 0:
            num_padded_heads = (num_heads + tp_size - 1) // tp_size * tp_size
            logger.warning_once(
                f"DSA num_heads {num_heads} is not divisible by tp_size "
                f"{tp_size}, will pad to {num_padded_heads} heads"
            )
            self.num_heads = num_padded_heads
        self.num_local_heads = self.num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # NoPE-only: the production caller asserts mla_use_nope before
        # building this class; re-assert for direct construction.
        if not config.mla_use_nope:
            raise NotImplementedError(
                "Kimi-K3.1 DSA on ROCm is NoPE-only (requires mla_use_nope)"
            )
        self.rotary_emb = None
        # The YaRN mscale adjustment also runs under NoPE (it rescales the
        # attention softmax, independent of rotary application); nvidia/dsa.py
        # applies it identically.
        rope_parameters: dict = dict(getattr(config, "rope_parameters", None) or {})
        rope_scaling = (
            rope_parameters
            if rope_parameters.get("rope_type") not in (None, "default")
            else None
        )
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        if config.mla_use_output_gate:
            self.g_proj = ColumnParallelLinear(
                self.hidden_size,
                self.v_head_dim * self.num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )

        dsa_config = config.dsa_config
        self.index_topk = dsa_config.index_topk
        self.index_head_dim = dsa_config.index_head_dim
        self.index_num_heads = dsa_config.index_num_heads
        if dsa_config.index_rope_dim != 0:
            raise NotImplementedError(
                "Kimi-K3.1 DSA on ROCm requires index_rope_dim == 0 "
                f"(got {dsa_config.index_rope_dim}); the index_rope path is not "
                "ported yet."
            )

        if topk_indices_buffer is None:
            raise ValueError(
                "KimiK3DSAAttention requires the model-level "
                "topk_indices_buffer created by KimiLinearModel"
            )
        self.topk_indices_buffer = topk_indices_buffer

        # Kimi-K3.1 has no index-share mechanism: every DSA layer builds
        # its own indexer; fail loudly on configs that ask for it.
        if getattr(dsa_config, "index_topk_freq", 1) not in (None, 1):
            raise NotImplementedError(
                "Kimi-K3.1 DSA does not support index_topk_freq != 1 "
                f"(index share), got {dsa_config.index_topk_freq}."
            )

        index_q_input_size = (
            self.q_lora_rank if self.q_lora_rank is not None else self.hidden_size
        )
        self.dsa_indexer = KimiK3DSAIndexer(
            vllm_config=vllm_config,
            hidden_size=self.hidden_size,
            index_q_input_size=index_q_input_size,
            index_head_dim=self.index_head_dim,
            index_num_heads=self.index_num_heads,
            index_topk=self.index_topk,
            rms_norm_eps=config.rms_norm_eps,
            index_k_norm_type=dsa_config.index_k_norm_type,
            cache_config=cache_config,
            quant_config=quant_config,
            topk_indices_buffer=self.topk_indices_buffer,
            prefix=f"{prefix}.dsa_indexer",
        )

        # The fp8_ds_mla normalization happens at config-verification time
        # (see the KimiK3 verifier in vllm/model_executor/models/config.py):
        # the runner resolves kv_cache_dtype from cache_config.cache_dtype
        # before the model is built; changing it inside the model would make
        # the spec dtype (bf16) inconsistent with the fp8_ds_mla page size.

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj
            if self.q_lora_rank is not None
            else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa
            if self.q_lora_rank is None
            else None,
            q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            g_proj=self.g_proj if (hasattr(self, "g_proj")) else None,
            indexer=self.dsa_indexer,
            indexer_rotary_emb=self.rotary_emb,
            is_sparse=True,
            topk_indices_buffer=self.topk_indices_buffer,
        )
        # The plain wrapper, not the AMD KimiK3MultiHeadLatentAttentionWrapper:
        # its fused decode Q-prep path writes dense 576-wide rows, which the
        # 656-byte fp8_ds_mla records cannot take.
        # VLLM_K3_DSA_FUSED_NORM_GATE only fuses the q_a/kv_a RMSNorms into one
        # launch and the output gate into one sigmoid-multiply; kv_c_normed and
        # the KV cache write are unchanged. The fused norm (one row per program)
        # loses to the AITER norms on large prefill chunks, so those keep them.
        # The multi-stream wrapper has its own forward (and fused gate), so it
        # takes neither.
        fuse_norm_gate = envs.VLLM_K3_DSA_FUSED_NORM_GATE
        wrapper_cls: type[MultiHeadLatentAttentionWrapper] = (
            MultiHeadLatentAttentionWrapper
        )
        wrapper_kwargs: dict = dict(
            fuse_qkv_rmsnorm=fuse_norm_gate and self.q_lora_rank is not None,
            fuse_output_gate=fuse_norm_gate,
            fuse_qkv_rmsnorm_max_tokens=_FUSED_QKV_NORM_MAX_TOKENS,
        )
        if (
            envs.VLLM_ROCM_DSA_MULTISTREAM
            and self.dsa_indexer.use_fp4
            and self.dsa_indexer.dcp_scores is not None
            and self.q_lora_rank is not None
            and hasattr(self, "g_proj")
        ):
            ms_comm = _dsa_ms_comm(*self.dsa_indexer.dcp_scores.shape)
            if ms_comm is not None:
                wrapper_cls = KimiK3DSAMultiStreamMLAWrapper
                wrapper_kwargs = {"ms_comm": ms_comm}
        akr_dsa.init_comms(self)
        self.mla_attn = wrapper_cls(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
            **wrapper_kwargs,
        )
        if isinstance(self.mla_attn, KimiK3DSAMultiStreamMLAWrapper):
            # The sparse MLA backend takes the aux-stream merge from here.
            self.mla_attn.mla_attn.impl._dsa_ms_state = self.mla_attn.ms_state
        # amd_kernels weights and warm-up, after MLAAttention's own post-load
        self.mla_attn.mla_attn._akr_post_load = functools.partial(
            _akr_post_load, self.mla_attn
        )

    def get_input_quant_key(self) -> None:
        # The generic MLA wrapper also feeds these activations to the gate.
        return None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states)
