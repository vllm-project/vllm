# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA (DeepSeek Sparse Attention) for the kimi_k3 stack on ROCm.

Ports `nvidia/dsa.py`: the indexer runs through
`SparseAttnIndexer.forward_hip` (the DeepSeek V3.2 ROCm pattern) and the
sparse MLA layers are served by ROCM_AITER_MLA_SPARSE over the fp8_ds_mla
cache. The DeepGEMM-only 32/64 index-head zero padding is dropped: the ROCm
indexer kernels take the native head count.
"""

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_dcp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
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
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    indexer_k_quant_and_cache_triton,
    rocm_aiter_sparse_attn_indexer,
    rocm_fp8_paged_mqa_logits,
)

from .ops.fp4_indexer import decode_layout, decode_tile_size, score_paged, store_keys
from .ops.fp4_indexer_prefill import prefill_chunks, prefill_workspace_bytes

logger = init_logger(__name__)


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


def _select(logits, lengths, out, values=None):
    rows, width = logits.shape
    k = out.shape[1]
    if values is not None:
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
            rel_starts = starts if start == 0 else starts - start
            logits = score_paged(
                q[start:end],
                cache,
                weights[start:end].float() * weight_scale,
                block_table[: starts.numel() - 1],
                rel_starts.clamp(0, end - start),
                row_ends,
                min(end - start, decode.write_max_decode_len or end - start),
            )
            selected = out[start:end]
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
    weights and drives `torch.ops.vllm.rocm_aiter_sparse_attn_indexer` via
    `SparseAttnIndexer.forward_hip` with `cache_layout="NORMAL"`: the
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
        index_k = self.k_index_norm(self.wk_index(hidden_states)[0])
        index_v = self.wv_index(hidden_states)[0]
        if self.use_fp4:
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
            )
            return self.topk_indices_buffer
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


class KimiK3DSAAttention(nn.Module):
    """Kimi K3.1 DSA (DeepSeek Sparse Attention) inference path on ROCm.

    Same MLA projections as the dense Kimi-K3 MLA path, plus the DSA indexer
    wired through `MLAModules(is_sparse=True)` so the ROCM_AITER_MLA_SPARSE
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
        self.mla_attn = MultiHeadLatentAttentionWrapper(
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
