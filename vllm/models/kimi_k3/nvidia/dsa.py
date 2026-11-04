# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA (DeepSeek Sparse Attention) for the kimi_k3 stack.

The DSA indexer uses the native upstream facilities
(``DeepseekV32IndexerCache`` + ``torch.ops.vllm.sparse_attn_indexer``); the
sparse MLA layers go through ``MultiHeadLatentAttentionWrapper`` with
``MLAModules(is_sparse=True)``. Checkpoints name indexer weights
``self_attn.indexer.{wq_b,wk,weights_proj,k_norm}``; the module tree keeps
``self_attn.dsa_indexer.*``; load_weights renames into it and stacks ``wk``
and ``weights_proj`` into the merged ``wkv_index``.
"""

import math
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import CacheConfig, CUDAGraphMode, VllmConfig, get_current_vllm_config
from vllm.distributed import get_dcp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.mla import (
    MLAModules,
    MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.utils import is_weights_pre_processed
from vllm.models.kimi_k3.common.dsa_cache import (
    KimiK3IndexerCache,
    indexer_cache_entry_bytes,
    kimi_k3_indexer_uses_fp4,
)
from vllm.models.kimi_k3.nvidia.index_rope import index_rope
from vllm.models.kimi_k3.nvidia.low_latency_gemm import (
    low_latency_linear,
    request_low_latency_warmup,
)
from vllm.models.kimi_k3.nvidia.mla import (
    _GATE_MULTI_STREAM_TOKEN_THRESHOLD,
    _gate_sigmoid_mul,
)
from vllm.models.kimi_k3.nvidia.ops.dsa_indexer import (
    dsa_fused_prologue,
    indexer_k_norm_quant_insert,
    indexer_q_quant_fold_weights,
)
from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import (
    mxfp4_fused_prologue,
    mxfp4_k_norm_quant_insert,
    mxfp4_q_quant,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backends.mla import indexer as v32_indexer
from vllm.v1.attention.ops.dcp_topk_merge import init_dcp_topk_merge

logger = init_logger(__name__)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class KimiK3DSAIndexerKNorm(nn.Module):
    """Bias-full fp32 LayerNorm for the DSA indexer shared key.

    ``F.layer_norm(x.float(), (dim,), weight.float(), bias.float(),
    eps).type_as(x)`` -- fp32 compute with a cast back to the input dtype.
    weight/bias are stored in fp32 directly (numerically equivalent to
    casting per forward), keeping only the input/output dtype casts.
    Selected via ``dsa_config.index_k_norm_type == "layernorm"``; legacy
    checkpoints keep the RMSNorm path untouched.
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.float(), self.normalized_shape, self.weight, self.bias, self.eps
        ).type_as(x)


class KimiK3DSAIndexer(nn.Module):
    """wq / wkv indexer projections + native ``DeepseekV32IndexerCache``.

    Forward quantizes the indexer query, folds the scales into the per-head
    weights and drives ``torch.ops.vllm.sparse_attn_indexer`` directly (the
    upstream deepseek_v2.Indexer pattern).
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
        index_rope_dim: int = 0,
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
        # MLAModules contract: MultiHeadLatentAttention asserts the indexer
        # exposes topk_tokens.
        self.topk_tokens = index_topk
        # Leading dims of every indexer head that receive RoPE. Weights are
        # pair-permuted at checkpoint conversion (interleaved -> half-split),
        # so serving applies neox-style rope with the main-attention
        # cos_sin_cache (YaRN scaling therefore carries over automatically).
        # 0 = pre-rope checkpoints; the forward path is then unchanged.
        if index_rope_dim < 0 or index_rope_dim % 2 != 0:
            raise ValueError(
                f"dsa_config.index_rope_dim must be even and >= 0, got {index_rope_dim}"
            )
        if index_rope_dim > index_head_dim:
            raise ValueError(
                f"dsa_config.index_rope_dim ({index_rope_dim}) must not "
                f"exceed index_head_dim ({index_head_dim})"
            )
        self.index_rope_dim = index_rope_dim
        self.wq_index = ReplicatedLinear(
            index_q_input_size,
            self.index_num_heads * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_index",
        )
        # The indexer query projection (wq_index) may be quantized in fp8
        # checkpoints; wk and weights_proj are bf16 and share one GEMM.
        self.wkv_index = MergedColumnParallelLinear(
            hidden_size,
            [self.index_head_dim, self.index_num_heads],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wkv_index",
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
        # convention of upstream DeepSeek V3.2; the DeepGEMM scoring kernels
        # read the scales as full fp32, which is exact for power-of-2 values.
        self.scale_fmt = "ue8m0"
        self.quant_block_size = self.index_head_dim
        self.index_softmax_scale = self.index_head_dim**-0.5
        # DeepGEMM indexer kernels support at most 64 index heads; smaller
        # head counts are zero-padded to the next supported width.
        if self.index_num_heads <= 32:
            self.padded_index_num_heads = 32
        elif self.index_num_heads <= 64:
            self.padded_index_num_heads = 64
        else:
            raise ValueError(
                "Official DeepGEMM DSA indexer supports at most 64 index "
                f"heads, got {self.index_num_heads}."
            )
        # FP8 keys by default; MXFP4 keys with MXFP4 queries (DeepGEMM's FP4
        # MQA logits) only on an explicit indexer_kv_dtype='mxfp4'.
        self.use_fp4_cache = kimi_k3_indexer_uses_fp4(vllm_config)
        self.k_cache = KimiK3IndexerCache(
            head_dim=indexer_cache_entry_bytes(self.index_head_dim, self.use_fp4_cache),
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cast(CacheConfig, cache_config),
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_total_seq_len = v32_indexer.get_max_prefill_buffer_size(vllm_config)
        self.topk_indices_buffer = topk_indices_buffer
        # Same derivation as the upstream SparseAttnIndexer
        # (sparse_attn_indexer.py): enable the allgather-k path when prefill
        # context parallelism is configured.
        self.use_pcp = vllm_config.parallel_config.prefill_context_parallel_size > 1
        # Set by KimiK3DSAAttention to skip scoring dense-MHA prefills.
        self.dense_mha_metadata_layer_name = ""
        # Set by KimiK3DSAAttention when its MLA layer takes ShardQ steps: the
        # op then follows that layer's token split.
        self.shardq_layer_name = ""
        self.topk_backend = vllm_config.kernel_config.sparse_indexer_topk_backend
        # PCP gathers the unquantized key inside the op.
        self.use_fused_k_insert = (
            not self.use_pcp
            and index_k_norm_type == "layernorm"
            and index_rope_dim == 0
        )
        if self.use_fp4_cache and not self.use_fused_k_insert:
            raise NotImplementedError("MXFP4 indexer keys need the fused key insert")
        # DCP wiring, same as the upstream SparseAttnIndexer: the op merges
        # each rank's local top-K into the global top-K only when these are
        # passed through (sparse_attn_indexer.py:_merge_dcp_topk_global).
        self._parallel_config = vllm_config.parallel_config
        self.dcp_world_size = self._parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self._cp_kv_cache_interleave_size: int | None = None
        # Same as the upstream SparseAttnIndexer: decode merges each rank's
        # packed local top-N in one exchange (0 keeps the generic merge).
        self.dcp_local_topk = init_dcp_topk_merge(vllm_config, index_topk)

    @property
    def cp_kv_cache_interleave_size(self) -> int:
        """Same lazy resolution as upstream SparseAttnIndexer: the real value
        is finalized after block_size setup, ahead of warmup/serving."""
        if self._cp_kv_cache_interleave_size is None:
            value = self._parallel_config.cp_kv_cache_interleave_size
            if isinstance(get_forward_context().attn_metadata, dict):
                self._cp_kv_cache_interleave_size = value
            return value
        return self._cp_kv_cache_interleave_size

    def _apply_index_rope(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE on the leading index_rope_dim dims of each head.

        Slices the leading index_rope_dim//2 frequencies from the main
        attention's cos_sin_cache (matching training, which reuses the
        leading freqs of the main freqs_cis for the indexer), computes in
        fp32 and casts back. index_q: [T, H, D]; index_k: [T, D] (already
        k_index_norm-ed, matching the training order norm -> rope).

        Uses the neox/half-split pairing (dim i pairs with dim i+half),
        matching the pair-permuted checkpoint conversion.
        """
        index_rope(
            index_q,
            index_k,
            positions.view(-1),
            rotary_emb.cos_sin_cache,
            self.index_rope_dim,
            False,  # interleave: neox/half-split
        )
        return index_q, index_k

    def _shardq_indexer_rows(self) -> torch.Tensor | None:
        """The rows a ShardQ step scores on this rank (decode rows, then this
        rank's prefill tokens), or None outside ShardQ steps and in cudagraph
        capture or replay: a graph would bake in the capture batch's row count
        and the address of its rows tensor."""
        if not self.shardq_layer_name:
            return None
        forward_context = get_forward_context()
        if (
            forward_context.cudagraph_runtime_mode != CUDAGraphMode.NONE
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            return None
        shardq = getattr(attn_metadata.get(self.shardq_layer_name), "shardq", None)
        return None if shardq is None else shardq.indexer_rows

    def _fused_k_insert(self, index_k: torch.Tensor, num_tokens: int) -> None:
        attn_metadata = get_forward_context().attn_metadata
        kv_cache = self.k_cache.kv_cache
        # Dummy / profiling run: the op leaves the cache and top-k buffer
        # alone too.
        if not isinstance(attn_metadata, dict) or kv_cache.numel() == 0:
            return
        metadata = attn_metadata[self.k_cache.prefix]
        assert isinstance(metadata, v32_indexer.DeepseekV32IndexerMetadata)
        assert self.topk_indices_buffer is not None
        norm = cast(KimiK3DSAIndexerKNorm, self.k_index_norm)
        insert = indexer_k_norm_quant_insert
        if self.use_fp4_cache:
            insert = mxfp4_k_norm_quant_insert
        insert(
            index_k,
            norm.weight,
            norm.bias,
            norm.eps,
            metadata.slot_mapping,
            kv_cache,
            self.topk_indices_buffer,
            num_tokens,
        )

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
        # A ShardQ step scores this rank's rows only: project and quantize the
        # query of just those (the op detects the compact query).
        rows = self._shardq_indexer_rows() if self.use_fused_k_insert else None
        if rows is not None:
            index_q_input = index_q_input[rows]
        index_q = self.wq_index(index_q_input)[0].view(
            -1, self.index_num_heads, self.index_head_dim
        )
        index_k, index_weights = self.wkv_index(hidden_states)[0].split(
            [self.index_head_dim, self.index_num_heads], dim=-1
        )
        if rows is not None:
            index_weights = index_weights[rows]
        if self.use_fused_k_insert:
            self._fused_k_insert(index_k, hidden_states.shape[0])
            index_k = None
        else:
            index_k = self.k_index_norm(index_k)
            if self.index_rope_dim > 0:
                assert positions is not None and rotary_emb is not None, (
                    "index_rope_dim > 0 requires positions and the main "
                    "attention rotary_emb (mla_use_nope models cannot enable "
                    "indexer rope)"
                )
                index_q, index_k = self._apply_index_rope(
                    index_q, index_k, positions, rotary_emb
                )
        return self.score(hidden_states, index_q, index_weights, index_k)

    def score(
        self,
        hidden_states: torch.Tensor,
        index_q: torch.Tensor,
        index_weights: torch.Tensor,
        index_k: torch.Tensor | None,
    ) -> torch.Tensor:
        """Top-k of the current rows into ``topk_indices_buffer``; ``index_k`` is
        None when the key was already inserted (``use_fused_k_insert``)."""
        weight_scale = self.index_softmax_scale * self.indexer_weight_scale
        q_scale: torch.Tensor | None = None
        if self.use_fp4_cache:
            q_quant, q_scale, weights = mxfp4_q_quant(
                index_q, index_weights, weight_scale, self.padded_index_num_heads
            )
        else:
            q_quant, weights = indexer_q_quant_fold_weights(
                index_q, index_weights, weight_scale, self.padded_index_num_heads
            )
        # Writes the index-k cache and the top-k indices into the shared
        # model-level buffer (same call as SparseAttnIndexer.forward_cuda:
        # under a dummy run kv_cache is an empty placeholder and the op takes
        # the workspace-reservation path internally, so kv_cache[0] indexing
        # is not allowed here).
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_quant,
            q_scale,  # None for fp8 Q: its scale is folded into the weights
            index_k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.index_head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.use_fused_k_insert,  # skip_k_cache_insert
            self.use_pcp,
            self._parallel_config.pcp_shard_decode_requests,
            _encode_layer_name(self.dense_mha_metadata_layer_name),
            self.use_fp4_cache,
            self.dcp_rank,
            self.dcp_world_size,
            self.cp_kv_cache_interleave_size,
            skip_topk_buffer_clear=self.use_fused_k_insert,
            topk_backend=self.topk_backend,
            shardq_layer_name=self.shardq_layer_name,
            dcp_local_topk=self.dcp_local_topk,
        )


class KimiK3DSAAttention(nn.Module):
    """Kimi K3.1 DSA (DeepSeek Sparse Attention) inference path.

    Same MLA projections as the dense Kimi-K3 MLA path, plus the DSA indexer
    wired through ``MLAModules(is_sparse=True)`` so the FlashMLA sparse
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
        aux_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        if config.dsa_config is None:
            raise ValueError("dsa_config is required for KimiK3DSAAttention")
        if current_platform.is_rocm():
            raise NotImplementedError(
                "Kimi-K3.1 DSA requires the CUDA FlashMLA sparse / DeepGEMM "
                "indexer kernels; ROCm is not supported."
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
        self.max_position_embeddings = config.max_position_embeddings

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

        # Full-attention layers are NoPE; the rope path below only runs for
        # checkpoints whose MLA uses positional encoding (GPT-J interleaved
        # rope dims; freqs are identical, only the dim pairing differs).
        rope_parameters: dict = dict(getattr(config, "rope_parameters", None) or {})
        if "rope_theta" not in rope_parameters:
            rope_parameters["rope_theta"] = 10000.0
        rope_scaling = (
            rope_parameters
            if rope_parameters.get("rope_type") not in (None, "default")
            else None
        )
        self.use_nope = bool(config.mla_use_nope)
        if self.use_nope:
            self.rotary_emb = None
        else:
            if rope_scaling:
                rope_parameters["rope_type"] = "deepseek_yarn"
            self.rotary_emb = get_rope(
                qk_rope_head_dim,
                max_position=self.max_position_embeddings,
                is_neox_style=False,
                rope_parameters=rope_parameters,
            )
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # The output gate is full-rank (v_head_dim * num_heads), same shape
        # as the dense Kimi-K3 MLA's g_proj.
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
        self.index_rope_dim = dsa_config.index_rope_dim
        if self.index_rope_dim > 0 and self.use_nope:
            raise ValueError(
                "dsa_config.index_rope_dim > 0 is incompatible with "
                "mla_use_nope (there is no rotary cache to share)"
            )
        if self.index_rope_dim > min(self.qk_rope_head_dim, self.index_head_dim):
            raise ValueError(
                f"dsa_config.index_rope_dim ({self.index_rope_dim}) must "
                f"not exceed min(qk_rope_head_dim={self.qk_rope_head_dim}, "
                f"index_head_dim={self.index_head_dim})"
            )

        vllm_config = get_current_vllm_config()
        if topk_indices_buffer is None:
            # Fallback for hosts that build KimiDecoderLayer directly:
            # allocate a private buffer. The main model path always passes
            # the single model-level buffer created in KimiLinearModel.
            logger.warning_once(
                "KimiK3DSAAttention did not receive a model-level "
                "topk_indices_buffer; allocating a per-layer fallback."
            )
            topk_buffer_device = "cuda" if torch.cuda.is_available() else "cpu"
            # Same guard as KimiLinearModel's model-level buffer: in
            # allgather_k mode prefill rows are padded to ceil(P/W) per rank,
            # so the all-gathered output takes num_decode_tokens +
            # W*ceil(P/W) rows, at most W-1 more than num_actual_tokens.
            dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
            dcp_kg_slack = dcp_world_size - 1 if dcp_world_size > 1 else 0
            topk_indices_buffer = torch.empty(
                (
                    vllm_config.scheduler_config.max_num_batched_tokens + dcp_kg_slack,
                    self.index_topk,
                ),
                dtype=torch.int32,
                device=topk_buffer_device,
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
            index_rope_dim=self.index_rope_dim,
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

        # Dense-MHA prefills never read the top-k indices (PCP excluded, as
        # upstream).
        indexer = self.dsa_indexer
        if not indexer.use_pcp:
            indexer.dense_mha_metadata_layer_name = self.mla_attn.mla_attn.layer_name
        # ShardQ (sparse_mla_shard_q): the indexer is not an indexer_op, so the
        # layer opts in itself, as DeepseekV32Attention does.
        mla_layer = cast(Any, self.mla_attn.mla_attn)
        if getattr(mla_layer.impl, "use_shardq", False):
            mla_layer.use_shardq = True
            indexer.shardq_layer_name = mla_layer.layer_name

        offload = vllm_config.offload_config
        self.weights_offloaded = (
            offload.uva.cpu_offload_gb > 0 or offload.prefetch.offload_group_size > 0
        )
        self.aux_stream = aux_stream
        self._gate_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
        # Set by pack_projections(): the packed input and query weights.
        self._packed: list[torch.Tensor] | None = None

    def pack_projections(self) -> None:
        """Enable the fused forward (as DeepseekV32Attention): pack
        ``[q_a | kv_a | wk | weights_proj]`` and ``[q_b | wq_index]`` into one
        weight each, the linears becoming views. Other configurations keep the
        wrapper forward."""
        attn = cast(Any, self.mla_attn.mla_attn)
        indexer = cast(Any, self.dsa_indexer)
        if not (
            indexer.use_fused_k_insert
            and indexer.dcp_world_size == 1
            and self.q_lora_rank is not None
            and self.use_nope
            and hasattr(self, "g_proj")
            and attn.kv_cache_dtype == "fp8_ds_mla"
            and attn.hisparse_cache is None
            and not is_weights_pre_processed()
            and not self.weights_offloaded
            # The fused prologue takes one eps for all three norms.
            and self.q_a_layernorm.variance_epsilon
            == self.kv_a_layernorm.variance_epsilon
            == indexer.k_index_norm.eps
        ):
            return
        groups = (
            (self.fused_qkv_a_proj, indexer.wkv_index),
            (self.q_b_proj, indexer.wq_index),
        )
        if not all(
            isinstance(linear.quant_method, UnquantizedLinearMethod)
            for linears in groups
            for linear in linears
        ):
            return
        self._packed = []
        for linears in groups:
            packed = torch.cat([linear.weight.data for linear in linears])
            rows = [linear.weight.shape[0] for linear in linears]
            for linear, view in zip(linears, packed.split(rows)):
                linear.weight.data = view
            request_low_latency_warmup(packed)
            self._packed.append(packed)
        if self.aux_stream is not None:
            self._gate_events = (torch.cuda.Event(), torch.cuda.Event())
        logger.info_once("Kimi K3.1 DSA layers use the fused forward.")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self._packed is None:
            return self.mla_attn(positions, hidden_states)
        # As in the dense Kimi-K3 MLA: the output-gate GEMM overlaps the
        # attention front-end on the aux stream for small batches.
        if (
            self._gate_events is not None
            and hidden_states.shape[0] < _GATE_MULTI_STREAM_TOKEN_THRESHOLD
        ):
            attn_out, gate = maybe_execute_in_parallel(
                lambda: self._fused_attention(hidden_states),
                lambda: self.g_proj(hidden_states)[0],
                *self._gate_events,
                self.aux_stream,
            )
        else:
            attn_out = self._fused_attention(hidden_states)
            gate = self.g_proj(hidden_states)[0]
        return self.o_proj(_gate_sigmoid_mul(attn_out, gate))[0]

    def _fused_attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.q_lora_rank is not None and self._packed is not None
        indexer = self.dsa_indexer
        attn = cast(Any, self.mla_attn.mla_attn)
        input_weight, query_weight = self._packed
        q_c, kv_c, k_pe, index_k, index_weights = low_latency_linear(
            hidden_states, input_weight
        ).split(
            [
                self.q_lora_rank,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                indexer.index_head_dim,
                indexer.index_num_heads,
            ],
            dim=-1,
        )

        # Profile / dummy runs have no metadata and leave both caches alone.
        caches: tuple = (None, None, None, None)
        has_prefills = False
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict) and attn.kv_cache.numel() > 0:
            assert isinstance(forward_context.slot_mapping, dict)
            has_prefills = cast(Any, attn_metadata[attn.layer_name]).num_prefills > 0
            index_metadata = attn_metadata[indexer.k_cache.prefix]
            assert isinstance(index_metadata, v32_indexer.DeepseekV32IndexerMetadata)
            caches = (
                attn.kv_cache,
                forward_context.slot_mapping[attn.layer_name],
                indexer.k_cache.kv_cache,
                index_metadata.slot_mapping,
            )
        assert indexer.topk_indices_buffer is not None
        k_norm = cast(KimiK3DSAIndexerKNorm, indexer.k_index_norm)
        # One eps for the three norms (equal, checked in pack_projections).
        prologue = dsa_fused_prologue
        if indexer.use_fp4_cache:
            prologue = mxfp4_fused_prologue
        q_c, kv_c_normed = prologue(
            q_c,
            kv_c,
            k_pe,
            index_k,
            self.q_a_layernorm.weight,
            self.kv_a_layernorm.weight,
            k_norm.weight,
            k_norm.bias,
            k_norm.eps,
            indexer.topk_indices_buffer,
            *caches,
        )

        q, index_q = low_latency_linear(q_c, query_weight).split(
            [
                self.num_local_heads * self.qk_head_dim,
                indexer.index_num_heads * indexer.index_head_dim,
            ],
            dim=-1,
        )
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        if has_prefills:
            # q is a column slice; some MHA prefill backends need it contiguous.
            q = q.contiguous()
        index_q = index_q.view(-1, indexer.index_num_heads, indexer.index_head_dim)
        indexer.score(hidden_states, index_q, index_weights, None)
        attn.impl.record_logical_topk_ready()

        output = torch.empty(
            (hidden_states.shape[0], self.num_local_heads * self.v_head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.unified_mla_attention_with_output(
            q,
            kv_c_normed,
            k_pe.unsqueeze(1),
            output,
            _encode_layer_name(attn.layer_name),
        )
        return output
