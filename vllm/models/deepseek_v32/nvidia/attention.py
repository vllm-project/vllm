# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn
from torch.profiler import record_function
from transformers import DeepseekV2Config, DeepseekV3Config

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_dcp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sparse_attn_indexer import (
    SparseAttnIndexer,
    sparse_attn_indexer,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepSeekV2FusedQkvAProjLinear,
    DeepseekV32IndexerCache,
    yarn_get_mscale,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.ops.common import (
    cp_lse_ag_out_rs,
    cp_lse_vmm_out_gather,
)
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce

from .kernels import fused_norm_rope, fused_q

logger = init_logger(__name__)
_logged_query_routes: set[tuple[str, int]] = set()


class DeepseekV32Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        super().__init__()
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.q_lora_rank = q_lora_rank

        # No tensor parallel, just replicated.
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
        # FP8 wk weights are upcasted to BF16 during loading to keep this fused.
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128
        self.topk_indices_buffer = topk_indices_buffer

        # fp8 naive cache: value in fp8 + fp32 scale per quant_block_size element.
        assert cache_config is not None, "DeepSeek V3.2 indexer requires cache_config"
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix

        from vllm.v1.attention.backends.mla.indexer import (
            get_max_prefill_buffer_size,
        )

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
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

        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        # Fused wk + weights_proj: one GEMM, then split.
        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # RoPE (NeoX) can introduce extra leading dims; reshape back to flat.
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)

        # Only quant q here; k quant is fused with cache insertion.
        q = q.view(-1, self.head_dim)
        q_fp8, q_scale = per_token_group_quant_fp8(
            q,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=self.scale_fmt is not None,
        )
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)

        weights = (
            weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
        )
        weights = weights.squeeze(-1)

        return self.indexer_op(hidden_states, q_fp8, k, weights)


class DeepseekV32Attention(MLAAttention):
    # Narrow the base's broadly-typed `indexer` to the concrete type so the
    # `if self.indexer is not None` guards below type-check its attributes.
    indexer: "DeepseekV32Indexer | None"

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        hidden_size = config.hidden_size
        qk_nope_head_dim = config.qk_nope_head_dim
        qk_rope_head_dim = config.qk_rope_head_dim
        v_head_dim = config.v_head_dim
        q_lora_rank = config.q_lora_rank
        kv_lora_rank = config.kv_lora_rank
        num_heads = config.num_attention_heads

        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        num_local_heads = num_heads // tp_size
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        scaling = qk_head_dim**-0.5
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        # DSA checkpoints may use plain ("default") or yarn-scaled RoPE.
        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        if config.rope_parameters["rope_type"] == "deepseek_yarn":
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            scaling = scaling * mscale * mscale

        # DSA "shared indexer" pattern: only some layers carry an indexer; the
        # rest reuse the top-k written by the previous indexer layer into the
        # shared topk_indices_buffer. DeepSeek-V3.2 builds it on every layer
        # (index_topk_freq defaults to 1); GLM-5.2 uses index_topk_freq=4 so
        # only layers [0,1,2,6,10,...] (+ MTP) carry one.
        layer_id = extract_layer_index(prefix)
        index_topk_freq = getattr(config, "index_topk_freq", 1)
        index_topk_pattern = getattr(config, "index_topk_pattern", None)
        index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)
        if index_topk_pattern is None:
            skip_topk = (
                max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0
            )
        elif 0 <= layer_id < len(index_topk_pattern):
            skip_topk = index_topk_pattern[layer_id] == "S"
        else:
            skip_topk = False
        # MTP/nextn layers always build a full indexer (they toggle at runtime).
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        is_mtp_layer = num_hidden_layers is not None and layer_id >= num_hidden_layers

        # Build kv_b_proj + indexer first; they are passed to MLAAttention.__init__
        # (which runs nn.Module.__init__ and registers them).
        kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        indexer = None
        if not skip_topk or is_mtp_layer:
            indexer = DeepseekV32Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                prefix=f"{prefix}.indexer",
            )

        # Set up the MLA engine (impl, KV cache, scales, backend, registration,
        # and process_weights_after_loading) via the MLAAttention base.
        super().__init__(
            num_heads=num_local_heads,
            scale=scaling,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            kv_b_proj=kv_b_proj,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_sparse=True,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
        )

        self.num_local_heads = num_local_heads
        self.qk_head_dim = qk_head_dim
        self.indexer = indexer
        self.topk_indices_buffer = topk_indices_buffer
        # Runtime toggle for index_share_for_mtp_iteration: MTP draft step 0
        # computes the top-k, steps 1+ set this True to reuse it.
        self.skip_topk = False
        # Fused fp8 paths: Triton fused norm/rope/cache + fused-q. Two layouts,
        # picked by the sparse MLA backend's query support:
        #   * supports_quant_query_input (FlashInfer sparse, SM100): per-tensor
        #     fp8 cache + a single packed fp8 MQA query.
        #   * not supported (FlashMLA sparse, SM90/SM100): fp8_ds_mla cache
        #     (per-128 block-scaled fp8 NoPE + unquantized bf16 RoPE) + a bf16
        #     (ql_nope, q_pe) query tuple. FA3 cannot mix a bf16 query with an
        #     fp8 KV cache, so FlashMLA (which dequantizes internally) is used.
        #     FlashMLA sparse runs on both Hopper and Blackwell, so this is the
        #     only DSA path on SM90 and an opt-in alternative on SM100.
        assert is_quantized_kv_cache(self.kv_cache_dtype), (
            "deepseek_v32 (nvidia) requires an fp8 KV cache served by a sparse "
            "MLA backend. Launch with --kv-cache-dtype fp8 (FlashInfer sparse) "
            "or --kv-cache-dtype fp8_ds_mla (FlashMLA sparse)."
        )
        self._fp8_query = self.impl.supports_quant_query_input
        if not self._fp8_query:
            assert self.kv_cache_dtype == "fp8_ds_mla", (
                "deepseek_v32 (nvidia) on a bf16-query sparse MLA backend "
                "(FlashMLA sparse) requires the fp8_ds_mla KV cache layout. "
                "Launch with --kv-cache-dtype fp8_ds_mla."
            )
        # The paged KV cache is stored as uint8 and viewed as fp8 for the decode
        # (per-tensor fp8). The fp8_ds_mla layout is consumed as raw bytes.
        self._fp8_kv_needs_view = self.kv_cache_dtype != "fp8_ds_mla"
        # GLM-5.2 uses interleaved indexer RoPE; DeepSeek-V3.2 uses NeoX.
        self._index_rope_interleave = getattr(config, "indexer_rope_interleave", False)

        # Remaining MLA projections (registered on this module).
        self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            hidden_size,
            [q_lora_rank, kv_lora_rank + qk_rope_head_dim],
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            q_lora_rank,
            num_heads * qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=config.rms_norm_eps)
        # reduce_results=False: the attention all-reduce is fused with the
        # following post_attention_layernorm in the decoder layer via
        # fused_allreduce_rms_norm.
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        # Lightning indexer uses its own RoPE; interleave maps to non-NeoX.
        self.indexer_rope_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=not getattr(config, "indexer_rope_interleave", False),
        )
        self._dcp_query_vmm_max_rows = 0
        if envs.VLLM_DCP_QUERY_VMM:
            dcp_group = get_dcp_group()
            if dcp_group.world_size > 1:
                if not self._fp8_query:
                    raise NotImplementedError(
                        "The bounded DCP query VMM experiment requires the "
                        "FlashInfer sparse FP8-query backend."
                    )
                from vllm.v1.attention.ops.dcp_query_vmm import (
                    DEFAULT_MAX_ROWS,
                    get_dcp_query_vmm_workspace,
                )

                get_dcp_query_vmm_workspace(
                    DEFAULT_MAX_ROWS,
                    num_local_heads,
                    kv_lora_rank + qk_rope_head_dim,
                    dcp_group.cpu_group,
                    dcp_group.device,
                )
                self._dcp_query_vmm_max_rows = DEFAULT_MAX_ROWS
        self._dcp_output_vmm_max_rows = 0
        if envs.VLLM_DCP_OUTPUT_VMM:
            dcp_group = get_dcp_group()
            if dcp_group.world_size > 1:
                from vllm.v1.attention.ops.dcp_output_vmm import (
                    DEFAULT_MAX_ROWS,
                    get_dcp_output_vmm_workspace,
                )

                # Initialize collectively during model construction, before
                # memory profiling or CUDA-graph warmup. All layers reuse this
                # fail-closed singleton.
                get_dcp_output_vmm_workspace(
                    DEFAULT_MAX_ROWS,
                    num_local_heads * dcp_group.world_size,
                    kv_lora_rank,
                    dcp_group.cpu_group,
                    dcp_group.device,
                )
                self._dcp_output_vmm_max_rows = DEFAULT_MAX_ROWS

    def forward(  # type: ignore[override]
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Captured: A-projections (+ indexer A-GEMM on indexer layers).
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_c, k_pe = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        if self.indexer is not None and not self.skip_topk:
            kw = self.indexer.wk_weights_proj(hidden_states)[0]
            index_k = kw[:, : self.indexer.head_dim]
            index_weights = kw[:, self.indexer.head_dim :]
        else:
            index_k = None
            index_weights = None

        num_tokens = hidden_states.shape[0]
        output = torch.empty(
            (num_tokens, self.num_local_heads * self.v_head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._fused_attention(
            positions, q_c, kv_c, k_pe, index_k, index_weights, output
        )
        return self.o_proj(output)[0]

    @eager_break_during_capture
    def _fused_attention(
        self,
        positions: torch.Tensor,
        q_c: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        index_k: torch.Tensor | None,
        index_weights: torch.Tensor | None,
        output: torch.Tensor,
    ) -> None:
        # One eager break for the whole attention. In FULL cudagraph mode (pure
        # decode) this decorator is a no-op, so everything here is captured; in
        # PIECEWISE (prefill) it runs eagerly. The cache writes, sparse indexer,
        # and forward_mqa all depend on per-step metadata and must not be split
        # out (PIECEWISE capture would otherwise miss them).
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw.get(self.layer_name)
        elif isinstance(attn_metadata_raw, list):
            attn_metadata = attn_metadata_raw[0].get(self.layer_name)
        else:
            attn_metadata = attn_metadata_raw

        slot_mapping = forward_context.slot_mapping
        assert isinstance(slot_mapping, dict)
        mla_slot = slot_mapping.get(self.layer_name)

        if self.indexer is not None:
            has_indexer = True
            indexer_k_norm_w = self.indexer.k_norm.weight
            indexer_k_norm_bias = self.indexer.k_norm.bias
            indexer_k_norm_eps = self.indexer.k_norm.eps
            indexer_k_rope_cos_sin_cache = self.indexer_rope_emb.cos_sin_cache
            indexer_k_cache = self.indexer.k_cache.kv_cache
            indexer_softmax_scale = self.indexer.softmax_scale
            indexer_n_head_scale = self.indexer.n_head**-0.5
        else:
            has_indexer = False
            indexer_k_norm_w = None
            indexer_k_norm_bias = None
            indexer_k_norm_eps = 1e-6
            indexer_k_rope_cos_sin_cache = None
            indexer_k_cache = None
            indexer_softmax_scale = 0.0
            indexer_n_head_scale = 0.0

        if attn_metadata is None:
            mla_kv_cache = None
            mla_k_scale = None
            indexer_k_cache = None
            mla_slot = None
        else:
            mla_kv_cache = self.kv_cache
            mla_k_scale = self._k_scale

        q_c = fused_norm_rope(
            positions,
            q_c,
            self.q_a_layernorm.weight,
            self.q_a_layernorm.variance_epsilon,
            kv_c,
            self.kv_a_layernorm.weight,
            self.kv_a_layernorm.variance_epsilon,
            k_pe,
            self.rotary_emb.cos_sin_cache,
            index_k,
            indexer_k_norm_w,
            indexer_k_norm_bias,
            indexer_k_norm_eps,
            indexer_k_rope_cos_sin_cache,
            self.topk_indices_buffer,
            slot_mapping=mla_slot,
            indexer_k_cache=indexer_k_cache,
            mla_kv_cache=mla_kv_cache,
            mla_kv_cache_dtype=self.kv_cache_dtype,
            mla_k_scale=mla_k_scale,
            has_indexer=has_indexer,
            index_rope_interleave=self._index_rope_interleave,
        )

        query_projection_heads = self.num_local_heads
        q = self.q_b_proj(q_c)[0].view(
            -1,
            query_projection_heads,
            self.qk_head_dim,
        )
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_nope = q_nope.transpose(0, 1)
        W_UK_T = self.W_UK_T
        if W_UK_T is None or W_UK_T.shape[0] != query_projection_heads:
            raise RuntimeError(
                "NVIDIA DCP query projection and W_UK_T head geometry "
                f"disagree: projection_heads={query_projection_heads}, "
                f"W_UK_T_shape={None if W_UK_T is None else tuple(W_UK_T.shape)}."
            )
        ql_nope = torch.bmm(q_nope, W_UK_T).transpose(0, 1)

        if self.indexer is not None:
            index_q = self.indexer.wq_b(q_c)[0]
            index_q = index_q.view(-1, self.indexer.n_head, self.indexer.head_dim)
        else:
            index_q = None

        query_workspace = None
        use_bounded_query_vmm = False
        num_actual_for_query = 0
        if attn_metadata is not None:
            num_actual_for_query = attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
            is_decode_only_for_query = (
                attn_metadata.num_prefills == 0  # type: ignore[attr-defined]
                and attn_metadata.num_decode_tokens  # type: ignore[attr-defined]
                == num_actual_for_query
            )
            use_bounded_query_vmm = (
                self.impl.dcp_world_size > 1
                and envs.VLLM_DCP_QUERY_VMM
                and is_decode_only_for_query
                and q_pe.shape[0] <= self._dcp_query_vmm_max_rows
                and num_actual_for_query <= self._dcp_query_vmm_max_rows
            )

        mqa_q_out = None
        if use_bounded_query_vmm:
            from vllm.v1.attention.ops.dcp_query_vmm import (
                DEFAULT_MAX_ROWS,
                get_dcp_query_vmm_workspace,
            )

            dcp_group = get_dcp_group()
            query_workspace = get_dcp_query_vmm_workspace(
                DEFAULT_MAX_ROWS,
                self.num_local_heads,
                self.kv_lora_rank + self.qk_rope_head_dim,
                dcp_group.cpu_group,
                dcp_group.device,
            )
            mqa_q_out = query_workspace.begin_publish(q_pe.shape[0])

        with record_function(
            "dcp.query_vmm.producer_fused_q"
            if use_bounded_query_vmm
            else "dcp.query_explicit.producer_fused_q"
        ):
            index_q_fp8, index_weights_out, mqa_q = fused_q(
                positions,
                q_pe,
                self.rotary_emb.cos_sin_cache,
                index_q,
                self.indexer_rope_emb.cos_sin_cache if has_indexer else None,
                ql_nope,
                self._q_scale,
                index_weights,
                indexer_softmax_scale,
                indexer_n_head_scale,
                has_indexer=has_indexer,
                index_rope_interleave=self._index_rope_interleave,
                quantize_mqa=self._fp8_query,
                mqa_q_out=mqa_q_out,
            )
        if use_bounded_query_vmm:
            if query_workspace is None:
                raise RuntimeError(
                    "DCP query VMM producer completed without an initialized workspace."
                )
            query_workspace.finish_publish()

        if self.indexer is not None:
            sparse_attn_indexer(
                q_c,
                self.indexer.k_cache.prefix,
                self.indexer.k_cache.kv_cache,
                index_q_fp8,
                None,  # q_scale folded into weights on the fp8 path
                None,  # k unused when skip_k_cache_insert=True
                index_weights_out,
                self.indexer.quant_block_size,
                self.indexer.scale_fmt,
                self.indexer.topk_tokens,
                self.indexer.head_dim,
                self.indexer.max_model_len,
                self.indexer.max_total_seq_len,
                self.topk_indices_buffer,
                True,  # skip_k_cache_insert
                False,  # use_fp4_cache
                # fused_norm_rope already cleared the topk buffer this forward.
                skip_topk_buffer_clear=True,
                # The fused NVIDIA path bypasses SparseAttnIndexer.forward_cuda,
                # so forward the run-constant DCP geometry explicitly.
                dcp_rank=self.indexer.indexer_op.dcp_rank,
                dcp_world_size=self.indexer.indexer_op.dcp_world_size,
                cp_kv_cache_interleave_size=(
                    self.indexer.indexer_op.cp_kv_cache_interleave_size
                ),
            )

        if attn_metadata is None:
            output.zero_()
            return

        num_actual = attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
        kv_cache = self.kv_cache
        if self._fp8_kv_needs_view:
            kv_cache = kv_cache.view(torch.float8_e4m3fn)
        if self._fp8_query:
            # FlashInfer sparse: single packed fp8 query.
            mqa_q_arg: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = mqa_q[
                :num_actual
            ]
        else:
            # FlashMLA sparse: bf16 (ql_nope, q_pe) tuple. mqa_q is the RoPE'd
            # q_pe; ql_nope is consumed directly.
            mqa_q_arg = (ql_nope[:num_actual], mqa_q[:num_actual])
        if self.impl.dcp_world_size > 1:
            if self.use_pcp:
                raise NotImplementedError(
                    "The NVIDIA DeepSeek-v3.2/GLM-5.2 override does not yet "
                    "support combined PCP and DCP attention."
                )
            if isinstance(mqa_q_arg, tuple):
                mqa_q_arg = torch.cat(mqa_q_arg, dim=-1)
            if use_bounded_query_vmm:
                if query_workspace is None:
                    raise RuntimeError(
                        "DCP query VMM route selected without an initialized workspace."
                    )
                with record_function(
                    f"dcp.query.route.vmm.decode.rows.{num_actual_for_query}"
                ):
                    mqa_q_arg = query_workspace.acquire_local_query(
                        num_actual_for_query
                    )
            else:
                # Each TP/DCP rank projects only its local query-head shard.
                # Every DCP rank must evaluate all query heads against its
                # owner-local KV shard before the output/LSE merge below.
                route_key = ("explicit_all_gather", num_actual_for_query)
                if route_key not in _logged_query_routes:
                    _logged_query_routes.add(route_key)
                    logger.info(
                        "Executing DCP query AllGather for decode rows=%d "
                        "(local_heads=%d, total_heads=%d).",
                        num_actual_for_query,
                        self.num_local_heads,
                        self.num_local_heads * self.impl.dcp_world_size,
                    )
                with record_function("dcp.query.route.explicit_all_gather"):
                    mqa_q_arg = get_dcp_group().all_gather(mqa_q_arg, dim=1)

        attn_out, lse = self.impl.forward_mqa(  # type: ignore[attr-defined]
            mqa_q_arg, kv_cache, attn_metadata, self
        )
        if use_bounded_query_vmm:
            if query_workspace is None:
                raise RuntimeError(
                    "DCP query VMM consumer completed without an initialized workspace."
                )
            query_workspace.acknowledge()
        if self.impl.dcp_world_size > 1:
            if lse is None:
                raise RuntimeError(
                    "The NVIDIA DCP attention path requires per-head LSE from "
                    "the sparse MLA backend."
                )
            dcp_group = get_dcp_group()
            if self.dcp_a2a:
                attn_out = dcp_a2a_lse_reduce(
                    attn_out,
                    lse,
                    dcp_group,
                    is_lse_base_on_e=self.impl.lse_base_on_e,
                )
            else:
                is_decode_only = (
                    attn_metadata.num_prefills == 0  # type: ignore[attr-defined]
                    and attn_metadata.num_decode_tokens  # type: ignore[attr-defined]
                    == num_actual
                )
                use_bounded_vmm = (
                    envs.VLLM_DCP_OUTPUT_VMM
                    and is_decode_only
                    and num_actual <= self._dcp_output_vmm_max_rows
                )
                if use_bounded_vmm:
                    with record_function(
                        f"dcp.output_lse.route.vmm.decode.rows.{num_actual}"
                    ):
                        attn_out = cp_lse_vmm_out_gather(
                            attn_out,
                            lse,
                            dcp_group,
                            is_lse_base_on_e=self.impl.lse_base_on_e,
                        )
                else:
                    route = (
                        "explicit_ag_rs.non_bounded_shape"
                        if envs.VLLM_DCP_OUTPUT_VMM
                        else "explicit_ag_rs.baseline"
                    )
                    with record_function(f"dcp.output_lse.route.{route}"):
                        attn_out = cp_lse_ag_out_rs(
                            attn_out,
                            lse,
                            dcp_group,
                            is_lse_base_on_e=self.impl.lse_base_on_e,
                        )
        x = attn_out.view(
            num_actual, self.num_local_heads, self.kv_lora_rank
        ).transpose(0, 1)
        out = (
            output[:num_actual]
            .view(num_actual, self.num_local_heads, self.v_head_dim)
            .transpose(0, 1)
        )
        torch.bmm(x, self.W_UV, out=out)
