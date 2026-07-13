# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import (
    CacheConfig,
    VllmConfig,
)
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.model_executor.layers.sparse_attn_indexer_kpool import SparseAttnIndexerKpool
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.models.deepseek_v2 import (
    DeepSeekV2FusedQkvAProjLinear,
    DeepseekV32IndexerCache,
    yarn_get_mscale,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig
from vllm.v1.kv_cache_interface import MLAAttentionSpec

logger = init_logger(__name__)


def naive_kda_lowerbound_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float = -5.0,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    g = g.float()
    g = g + dt_bias.view(1, -1)
    g = g.view(g.shape[0], A_log.numel(), -1)
    g = lower_bound * torch.nn.functional.sigmoid(A_log.view(-1, 1).exp() * g)
    return g.to(output_dtype)


class Glm5NextIndexerCache(DeepseekV32IndexerCache):
    """Indexer K cache that stores kpool-compressed entries.

    Setting ``compress_ratio = index_kpool`` on the kv_cache_spec makes vLLM's
    indexer metadata builder emit pool-granular ``slot_mapping`` /
    ``seq_lens`` / ``cu_seq_lens`` / ``page_table`` for free, and shrinks the
    cache allocation to ``storage_block_size = block_size // kpool``. The pool
    *content* (softmax-weighted sum vs keep-every-Nth) is computed by the
    kpool compress kernel inside the indexer op — the cache only provides the
    addressing, which is identical for both schemes.

    The indexer shares one block with the co-located MLA (a single
    ``MLAAttentionSpec`` / block_table), so ``block_size`` is the model-wide
    ``cache_config.block_size``. DeepGEMM's paged-MQA kernel
    (``csrc/apis/attention.hpp``) requires ``block_kv`` — which equals
    ``storage_block_size`` here — to be exactly 32 or 64. With
    ``index_kpool = 16`` that means ``--block-size 1024`` (sm90) or ``512``
    (sm100). A smaller block (e.g. the default 64) silently collapses
    ``storage_block_size`` (64 // 16 = 4) and only fails later at the opaque
    C++ assert; ``get_kv_cache_spec`` guards this up front instead.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
        index_kpool: int,
    ):
        super().__init__(
            head_dim=head_dim, dtype=dtype, prefix=prefix, cache_config=cache_config
        )
        assert index_kpool > 1, "Glm5NextIndexerCache expects index_kpool > 1"
        self._index_kpool = index_kpool

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        from dataclasses import replace

        spec = super().get_kv_cache_spec(vllm_config)
        # compress_ratio lives on MLAAttentionSpec, but the base
        # DeepseekV32IndexerCache.get_kv_cache_spec is typed to return the
        # KVCacheSpec base; narrow so dataclass.replace sees the field.
        assert isinstance(spec, MLAAttentionSpec)
        spec = replace(spec, compress_ratio=self._index_kpool)

        # storage_block_size (= block_size // index_kpool) is forwarded to
        # DeepGEMM paged-MQA as block_kv, which must be 32 or 64. Since the
        # indexer shares the MLA block (one block_table), block_size comes from
        # cache_config.block_size and must be sized so the division lands on
        # 32/64 -- i.e. --block-size = index_kpool * 64 (sm90) / * 32 (sm100).
        storage_block_size = spec.block_size // self._index_kpool
        assert spec.block_size % self._index_kpool == 0 and storage_block_size in (
            32,
            64,
        ), (
            "Glm5NextIndexerCache: kpool indexer requires cache block_size to be "
            f"a multiple of index_kpool ({self._index_kpool}) and yield "
            "storage_block_size (block_size // index_kpool) of 32 or 64 for "
            f"DeepGEMM paged-MQA, got block_size={spec.block_size} -> "
            f"storage_block_size={storage_block_size}. Set --block-size "
            f"{self._index_kpool * 64} on sm90 (Hopper) or "
            f"{self._index_kpool * 32} on sm100 (Blackwell)."
        )
        return spec


class Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        # Indexer is only constructed for v32 configs, where these sparse-indexer
        # fields are guaranteed populated; narrow away the `int | None` declared
        # on Glm5NextConfig for the optional-indexer case.
        assert config.index_topk is not None
        assert config.index_n_heads is not None
        assert config.index_head_dim is not None
        assert config.index_kpool is not None
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.index_kpool = config.index_kpool
        self.q_lora_rank = q_lora_rank  # 1536

        # kpool
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.index_kpool, self.head_dim, dtype=torch.float32)
        )
        # NOTE: kept as a bare nn.Parameter (not ReplicatedLinear) so the weight
        # name matches the checkpoint verbatim ("index_kpool_compress_gate",
        # no ".weight" suffix) — the trained checkpoint stores it the sglang way.
        # Shape [head_dim, hidden_size]; consumed via F.linear(x, gate) = x @ gate.T.
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(self.head_dim, hidden_size, dtype=torch.bfloat16)
        )

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
        # FP8 wk weights are upcasted to BF16 during loading to maintain fusion.
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
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        # NOTE: (zyongye) we use fp8 naive cache,
        #       where we store value in fp8 and scale in fp32
        #       per self.quant_block_size element
        self.k_cache = Glm5NextIndexerCache(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            index_kpool=self.index_kpool,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix
        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.indexer_op = SparseAttnIndexerKpool(
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
        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)

        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        # Fused wk + weights_proj: one GEMM, then split
        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # Note: RoPE (NeoX) can introduce extra leading dimensions during
        # compilation so we need to reshape back to token-flattened shapes
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        # `rotary_emb` is shape-preserving; `q_pe` is already
        # [num_tokens, n_head, rope_dim].
        q = torch.cat([q_pe, q_nope], dim=-1)
        # `k_pe` is [num_tokens, 1, rope_dim] (MQA).
        k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)

        # we only quant q here since k quant is fused with cache insertion
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

        # kpool: per-token gate score driving the softmax-weighted pool. Computed
        # from the same hidden_states that produced `k`, so it stays token-aligned.
        # F.linear(x, gate) = x @ gate.T  with gate [head_dim, hidden_size].
        gate_score = F.linear(hidden_states, self.index_kpool_compress_gate)

        # DeepGEMM's MQA-logits kernels (fp8_mqa_logits /
        # fp8_fp4_paged_mqa_logits) require num_heads in {32, 64}; this
        # checkpoint uses index_n_heads=16. Zero-pad q and the per-head
        # weights: logits are a weights-weighted sum over heads, so
        # zero-weight padded heads contribute exactly nothing.
        if self.n_head < 32:
            pad = 32 - self.n_head
            q_fp8 = torch.cat(
                [q_fp8, q_fp8.new_zeros(q_fp8.shape[0], pad, self.head_dim)], dim=1
            )
            weights = torch.cat(
                [weights, weights.new_zeros(weights.shape[0], pad)], dim=1
            )

        return self.indexer_op(
            hidden_states,
            q_fp8,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=self.index_kpool_compress_ape,
            index_kpool=self.index_kpool,
            positions=positions,
        )


class Glm5NextMLAAttention(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        input_size: int | None = None,
        skip_rope: bool | None = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        # Use input_size for projection input dimensions if provided,
        # otherwise default to hidden_size (used in Eagle3 Deepseek with MLA)
        proj_input_size = input_size if input_size is not None else self.hidden_size

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
                proj_input_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                proj_input_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                proj_input_size,
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
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if not skip_rope:
            assert config.rope_parameters is not None
            if config.rope_parameters["rope_type"] != "default":
                config.rope_parameters["rope_type"] = (
                    "deepseek_yarn"
                    if config.rope_parameters.get("apply_yarn_scaling", True)
                    else "deepseek_llama_scaling"
                )

            self.rotary_emb: RotaryEmbedding | None = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=False,
            )

            if (
                config.rope_parameters["rope_type"] != "default"
                and config.rope_parameters["rope_type"] == "deepseek_yarn"
            ):
                mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
                scaling_factor = config.rope_parameters["factor"]
                mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                self.scaling = self.scaling * mscale * mscale
        else:
            self.rotary_emb = None

        # `index_topk` is declared on Glm5NextTextConfig with a default of None,
        # so hasattr() is True even for full-MLA configs (no kpool indexer).
        self.is_v32 = getattr(config, "index_topk", None) is not None
        # self.is_v32 = False

        _skip_topk = False
        if self.is_v32:
            self.indexer_rope_emb: RotaryEmbedding | None = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=not getattr(config, "indexer_rope_interleave", False),
            )
            # The sparse indexer projects from the MLA q-lora rank, which is
            # always set for v32 MLA configs; narrow away the `int | None`.
            assert q_lora_rank is not None
            self.indexer: Indexer | None = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )

            # Enable IndexCache for DeepSeek models to reduce redundant top-k
            # token selection computations in sparse attention.
            use_index_cache = getattr(config, "use_index_cache", False)
            if use_index_cache:
                # IndexCache config
                # Refer: https://arxiv.org/abs/2603.12201 for more details.
                _index_topk_freq = getattr(config, "index_topk_freq", 1)
                _index_topk_pattern = getattr(config, "index_topk_pattern", None)
                layer_id = extract_layer_index(prefix)
                if _index_topk_pattern is None:
                    _skip_topk = max(layer_id - 1, 0) % _index_topk_freq != 0
                elif 0 <= layer_id < len(_index_topk_pattern):
                    _skip_topk = _index_topk_pattern[layer_id] == "S"

        else:
            self.indexer_rope_emb = None
            self.indexer = None

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
            indexer=self.indexer,
            indexer_rotary_emb=self.indexer_rope_emb,
            is_sparse=self.is_v32,
            topk_indices_buffer=topk_indices_buffer,
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
            skip_topk=_skip_topk,
        )

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, output: torch.Tensor
    ) -> None:
        # Delegate to the MultiHeadLatentAttentionWrapper, which performs the
        # q/kv projection, RoPE, the sparse-indexer top-k selection
        # (``self.indexer``), the inner MLA attention, and the output
        # projection. Re-implementing the projection here and calling the inner
        # ``mla_attn`` directly would skip the indexer call, leaving the topk
        # buffer empty and silently corrupting attention. Mirrors
        # DeepseekV2MLAAttention.forward.
        output[:] = self.mla_attn(positions, hidden_states)


class Glm5NextLinearAttention(KimiGatedDeltaNetAttention):
    """GLM5-Next KDA layer.

    GLM5-Next KDA checkpoints ship ``linear_attn_config["safe_gate"]``: the 70B
    sets it ``False`` (gate = ``-exp(A)*softplus(g+g_bias)``, the fused-kernel
    default), but the 300B VLM sets it ``True`` and requires the bounded gate
    ``y = lower_bound * sigmoid(exp(A)*(g+g_bias))`` (default ``lower_bound``
    -5.0 when absent). Without the bounded gate the KDA forget factor is wrong,
    which corrupts the linear-attention state and degrades generation as the
    sequence grows. We read ``safe_gate``/``lower_bound`` off the config and
    expose them as ``self.kda_safe_gate``/``self.kda_lower_bound``, consumed by
    ``KimiGatedDeltaNetAttention._forward`` to select the gate branch in
    ``fused_kda_gate`` / ``chunk_kda_with_fused_gate``. Other customizations:

    - KDA projections are kept BF16 even in FP8 checkpoints (no
      ``weight_scale_inv`` is stored for them), so the quant config is stripped
      while building the projection modules -- mirroring the MLA path, which
      also passes ``quant_config=None``.
    - The checkpoint stores ``A_log`` as a 1-D ``(num_heads,)`` tensor; the
      upstream loader assumes the 4-D param shape, so a reshape-aware loader
      is attached.
    """

    def __init__(
        self,
        config: Glm5NextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        # KDA projections are BF16 in the checkpoint (no weight_scale_inv),
        # even for FP8 checkpoints. The GDN base reads quant_config off
        # vllm_config wholesale, so swap it to None just for this layer's
        # construction and restore it afterwards (single-threaded init).
        saved_quant = vllm_config.quant_config
        vllm_config.quant_config = None
        try:
            super().__init__(config, vllm_config, prefix)
        finally:
            vllm_config.quant_config = saved_quant

        # KDA gate variant: read safe_gate/lower_bound (matching SGlang's
        # `self.attn.lower_bound = linear_attn_config.get("lower_bound", -5.0)`
        # gated on safe_gate). Consumed by KimiGatedDeltaNetAttention._forward.
        linear_attn_config = getattr(config, "linear_attn_config", None) or {}
        if linear_attn_config.get("safe_gate", True):
            self.kda_safe_gate = True
            self.kda_lower_bound = linear_attn_config.get("lower_bound", -5.0)
        else:
            self.kda_safe_gate = False
            self.kda_lower_bound = -5.0

        # checkpoint A_log is 1-D (num_heads,); upstream loader assumes 4-D.
        def a_log_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor):
            if loaded_weight.dim() == 1:
                loaded_weight = loaded_weight.view([1, 1, -1, 1])
            return sharded_weight_loader(2)(param, loaded_weight)

        self.A_log.weight_loader = a_log_weight_loader
