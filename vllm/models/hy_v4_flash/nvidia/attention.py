# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer MLA projections with Flash's SWA/IndexPool selection."""

from copy import deepcopy

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.kernels.triton.fused_sigmoid_gate_mul import fused_sigmoid_gate_mul
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.hpc import hpc_gated_mla_gemm, hpc_gated_mla_supported
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import make_attention_sink
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm
from vllm.models.hy_v4.common.mrope import fused_mrope_norm, rope_into
from vllm.models.hy_v4.nvidia.attention import HYV4MLAAttention
from vllm.models.hy_v4_flash.configs import SLIDING_ATTENTION
from vllm.platforms import current_platform

from .compressed_indexer import FlashCompressedIndexer
from .swa import FlashSparseBackend, FlashSWABackend, FlashWindowIndexer


class HYV4FlashMLAAttention(MLAAttention):
    # FlashSWABackend indexes live pages and forces the paged FP8 kernel.
    supports_fp8_ds_mla_sliding_window = True


def _get_fused_projection_sizes(config, is_window_layer: bool) -> list[int]:
    sizes = [config.q_lora_rank, config.kv_lora_rank + config.qk_rope_head_dim]
    if not is_window_layer:
        sizes.extend(
            [
                config.compressed_indexer_head_dim,
                config.compressed_indexer_num_attention_heads,
            ]
        )
    return sizes


def _can_fuse_unquantized_projections(*layers: LinearBase) -> bool:
    return (
        all(type(layer.quant_method) is UnquantizedLinearMethod for layer in layers)
        and len({layer.params_dtype for layer in layers}) == 1
    )


class HYV4FlashFusedQIndexerQueryLinear(MergedColumnParallelLinear):
    """Fuse TP-sharded attention Q with replicated Indexer query output."""

    def __init__(
        self,
        input_size: int,
        q_output_size: int,
        indexer_output_size: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        self.q_tp_rank = get_tensor_model_parallel_rank()
        self.q_tp_size = get_tensor_model_parallel_world_size()
        assert q_output_size % self.q_tp_size == 0
        self.q_output_size = q_output_size
        self.q_output_size_per_partition = q_output_size // self.q_tp_size
        super().__init__(
            input_size,
            [self.q_output_size_per_partition, indexer_output_size],
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            disable_tp=True,
            prefix=prefix,
        )

    def _slice_q_weight(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None,
    ) -> torch.Tensor:
        output_dim = getattr(param, "output_dim", None)
        if loaded_shard_id == 0 and output_dim is not None:
            loaded_size = loaded_weight.shape[output_dim]
            if loaded_size == self.q_output_size:
                loaded_weight = loaded_weight.narrow(
                    output_dim,
                    self.q_tp_rank * self.q_output_size_per_partition,
                    self.q_output_size_per_partition,
                )
            else:
                assert loaded_size == self.q_output_size_per_partition
        return loaded_weight

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        loaded_weight = self._slice_q_weight(param, loaded_weight, loaded_shard_id)
        super().weight_loader(param, loaded_weight, loaded_shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        loaded_weight = self._slice_q_weight(param, loaded_weight, loaded_shard_id)
        super().weight_loader_v2(param, loaded_weight, loaded_shard_id)


class HYV4FlashAttention(HYV4MLAAttention):
    q_a_layernorm: RMSNorm
    q_b_proj: ColumnParallelLinear | HYV4FlashFusedQIndexerQueryLinear
    fused_qkv_a_proj: MergedColumnParallelLinear
    linear_gate: ColumnParallelLinear

    def __init__(self, vllm_config, config, layer_idx, topk_indices_buffer, prefix):
        nn.Module.__init__(self)
        self.is_window_layer = config.layer_types[layer_idx] == SLIDING_ATTENTION
        self.config = deepcopy(config)
        self.config.rope_parameters = config.rope_for_layer(layer_idx)
        self.hidden_size = config.hidden_size
        self.layer_idx = self.layer_id = layer_idx
        self.prefix = prefix
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.max_position_embeddings = vllm_config.model_config.max_model_len
        self.num_heads = (
            config.swa_num_attention_heads
            if self.is_window_layer
            else config.num_attention_heads
        )
        self.num_local_heads = self.num_heads // get_tensor_model_parallel_world_size()
        self.qk_nope_head_dim = (
            config.swa_qk_nope_head_dim
            if self.is_window_layer
            else config.qk_nope_head_dim
        )
        self.qk_rope_head_dim = (
            config.swa_qk_rope_head_dim
            if self.is_window_layer
            else config.qk_rope_head_dim
        )
        self.v_head_dim = (
            config.swa_v_head_dim if self.is_window_layer else config.v_head_dim
        )
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.qk_head_dim**-0.5
        self.is_sparse = self.create_indexer = True
        self.skip_topk = self.external_topk = False
        self.topk_snapshot = None
        self.gated_mla = config.gated_mla
        self.learnable_sink = config.learnable_sink
        capability = current_platform.get_device_capability()
        if capability is None or capability.major not in (9, 10):
            raise NotImplementedError(
                "Flash MLA requires Hopper or Blackwell data-center GPUs"
            )
        if self.kv_lora_rank != 512 or self.qk_rope_head_dim != 64:
            raise ValueError("Flash sparse MLA requires latent512 + rotary64 KV")
        if (
            not config.learnable_sink
            or not config.gated_mla
            or config.gating_type != "elementwise"
        ):
            raise ValueError("Flash requires learnable sinks and elementwise gated MLA")
        if any(
            getattr(config, name, None) is not None
            for name in ("gate_lora_rank", "o_lora_rank")
        ):
            raise ValueError("Flash requires direct gate/output projections")
        if (
            getattr(config, "gate_groups", 1) != 1
            or getattr(config, "o_groups", 1) != 1
            or getattr(config, "full_nope", False)
            or not getattr(config, "qk_norm", True)
        ):
            raise ValueError("Flash requires normalized ungrouped RoPE MLA")
        quant = vllm_config.quant_config
        self.fused_qkv_a_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.fused_qkv_a_proj",
            disable_tp=True,
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, config.rms_norm_eps)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.linear_gate = ColumnParallelLinear(
            config.hidden_size,
            self.num_heads * self.v_head_dim,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.linear_gate",
        )
        self.use_hpc_gated_mla = hpc_gated_mla_supported(
            config.gating_type, self.linear_gate
        )
        has_post_proj_norm = config.sandwich_norm and (
            self.is_window_layer or not config.dsa_skip_attn_post_norm
        )
        # With a post-projection norm, the all-reduce is fused into it.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant,
            reduce_results=not has_post_proj_norm,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=self.config.rope_parameters,
            is_neox_style=False,
        )
        self.indexer_rope_emb = self.rotary_emb
        self.indexer = (
            FlashWindowIndexer(
                config.sliding_window, topk_indices_buffer, f"{prefix}.attn"
            )
            if self.is_window_layer
            else FlashCompressedIndexer(
                vllm_config, self.config, topk_indices_buffer, f"{prefix}.indexer"
            )
        )
        self._fuse_qkv_a_indexer = False
        self._fuse_q_b_indexer = False
        if not self.is_window_layer:
            assert isinstance(self.indexer, FlashCompressedIndexer)
            token_key_projection = self.indexer.token_key_projection
            head_weight_projection = self.indexer.head_weight_projection
            query_projection = self.indexer.query_projection
            assert token_key_projection is not None
            assert head_weight_projection is not None
            assert query_projection is not None
            if _can_fuse_unquantized_projections(
                self.fused_qkv_a_proj,
                token_key_projection,
                head_weight_projection,
            ):
                self.fused_qkv_a_proj = MergedColumnParallelLinear(
                    config.hidden_size,
                    _get_fused_projection_sizes(config, self.is_window_layer),
                    bias=False,
                    params_dtype=self.fused_qkv_a_proj.params_dtype,
                    quant_config=None,
                    prefix=f"{prefix}.fused_qkv_a_proj",
                    disable_tp=True,
                )
                self.indexer.token_key_projection = None
                self.indexer.head_weight_projection = None
                self._fuse_qkv_a_indexer = True

            if _can_fuse_unquantized_projections(self.q_b_proj, query_projection):
                self.q_b_proj = HYV4FlashFusedQIndexerQueryLinear(
                    self.q_lora_rank,
                    self.num_heads * self.qk_head_dim,
                    self.indexer.num_heads * self.config.compressed_indexer_head_dim,
                    self.q_b_proj.params_dtype,
                    prefix=f"{prefix}.q_b_proj",
                )
                self.indexer.query_projection = None
                self._fuse_q_b_indexer = True
        self.learnable_sink_param = make_attention_sink(
            self.num_local_heads, torch.float32
        )
        self._force_sparse_mqa()
        self.mla_attn = HYV4FlashMLAAttention(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=vllm_config.cache_config,
            quant_config=quant,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=True,
            indexer=self.indexer,
            topk_indices_buffer=self.indexer.topk_indices_buffer,
            attn_backend=FlashSWABackend
            if self.is_window_layer
            else FlashSparseBackend,
            sliding_window=config.sliding_window if self.is_window_layer else None,
            sinks=self.learnable_sink_param,
        )
        self.post_proj_norm = (
            RMSNorm(config.hidden_size, config.rms_norm_eps)
            if has_post_proj_norm
            else None
        )

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        projection = self.fused_qkv_a_proj(hidden_states)[0]
        indexer_token_keys = None
        indexer_head_weights = None
        if self._fuse_qkv_a_indexer:
            (
                q_c,
                kv,
                indexer_token_keys,
                indexer_head_weights,
            ) = projection.split(
                [
                    self.q_lora_rank,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                    self.config.compressed_indexer_head_dim,
                    self.config.compressed_indexer_num_attention_heads,
                ],
                dim=-1,
            )
        else:
            q_c, kv = projection.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
        indexer_query = None
        if positions.ndim == 2:
            q_c, kv_c, k_pe, rope_pos, cos_sin = fused_mrope_norm(
                self.rotary_emb,
                positions,
                q_c,
                kv,
                self.q_a_layernorm,
                self.kv_a_layernorm,
            )
            q_proj = self.q_b_proj(q_c)[0]
            if self._fuse_q_b_indexer:
                assert isinstance(self.indexer, FlashCompressedIndexer)
                q_proj, indexer_query = q_proj.split(
                    [
                        self.num_local_heads * self.qk_head_dim,
                        self.indexer.num_heads
                        * self.config.compressed_indexer_head_dim,
                    ],
                    dim=-1,
                )
            q = q_proj.view(-1, self.num_local_heads, self.qk_head_dim)
            q_pe = q[..., self.qk_nope_head_dim :]
            rope_into(rope_pos, q_pe, cos_sin, q_pe)
        else:
            q_c = self.q_a_layernorm(q_c)
            q_proj = self.q_b_proj(q_c)[0]
            kv_c, k_pe = kv.split((self.kv_lora_rank, self.qk_rope_head_dim), -1)
            kv_c = self.kv_a_layernorm(kv_c)
            if self._fuse_q_b_indexer:
                assert isinstance(self.indexer, FlashCompressedIndexer)
                q_proj, indexer_query = q_proj.split(
                    [
                        self.num_local_heads * self.qk_head_dim,
                        self.indexer.num_heads
                        * self.config.compressed_indexer_head_dim,
                    ],
                    dim=-1,
                )
            else:
                indexer_query = None
            q = q_proj.view(-1, self.num_local_heads, self.qk_head_dim)
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe.unsqueeze(1)
            )
        if llama_4_scaling is not None:
            q *= llama_4_scaling
        out = torch.empty(
            (hidden_states.shape[0], self.num_local_heads * self.v_head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._indexer_and_attn(
            hidden_states,
            q_c,
            positions,
            q,
            kv_c,
            k_pe,
            out,
            indexer_token_keys=indexer_token_keys,
            indexer_head_weights=indexer_head_weights,
            indexer_query=indexer_query,
        )
        if self.use_hpc_gated_mla:
            assert hidden_states.is_contiguous() and out.is_contiguous()
            out = hpc_gated_mla_gemm(hidden_states, self.linear_gate.weight, out)
        else:
            out = fused_sigmoid_gate_mul(out, self.linear_gate(hidden_states)[0])
        out = self.o_proj(out)[0]
        if self.post_proj_norm is None:
            return out
        return fused_allreduce_rms_norm(out, None, self.post_proj_norm)[0]

    @eager_break_during_capture
    def _indexer_and_attn(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        out: torch.Tensor,
        indexer_token_keys: torch.Tensor | None = None,
        indexer_head_weights: torch.Tensor | None = None,
        indexer_query: torch.Tensor | None = None,
    ) -> None:
        """Use the combined DSA projections before running sparse attention."""
        if self.indexer is not None and self.is_sparse and not self.skip_topk:
            if (
                indexer_token_keys is None
                and indexer_head_weights is None
                and indexer_query is None
            ):
                self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
            else:
                assert isinstance(self.indexer, FlashCompressedIndexer)
                self.indexer(
                    hidden_states,
                    q_c,
                    positions,
                    self.indexer_rope_emb,
                    indexer_token_keys=indexer_token_keys,
                    indexer_head_weights=indexer_head_weights,
                    indexer_query=indexer_query,
                )
            if self.topk_snapshot is not None:
                assert self.topk_indices_buffer is not None
                num_rows = hidden_states.shape[0]
                self.topk_snapshot[:num_rows].copy_(self.topk_indices_buffer[:num_rows])
        if self.is_sparse:
            self.mla_attn.impl.record_logical_topk_ready()  # type: ignore[attr-defined]
        out.copy_(
            self.mla_attn(
                q,
                kv_c_normed,
                k_pe,
                output_shape=out.shape,
            )
        )
