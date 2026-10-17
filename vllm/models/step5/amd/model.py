# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-5: Step-3.5 with compressed sparse attention (CSA) and an FP32 residual."""

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import unified_kv_cache_update
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.step3p5 import (
    Step3p5Attention,
    Step3p5DecoderLayer,
    Step3p5ForCausalLM,
    Step3p5Model,
)
from vllm.model_executor.models.step3p7 import Step3p7ForConditionalGeneration
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.step5.common.config import Step5Config
from vllm.models.step5.common.csa_attention import step5_csa_attention
from vllm.models.step5.common.csa_cache import (
    Step5CSAIndexCache,
    Step5CSASummaryCache,
)
from vllm.models.step5.common.ops.csa import indexer_norm
from vllm.platforms import current_platform
from vllm.utils.torch_utils import _encode_layer_name, kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
    RocmAiterUnifiedAttentionBackend,
)

logger = init_logger(__name__)


class Step5SparseIndexer(nn.Module):
    """MQA indexer with ``heads_per_group`` query heads per KV group.

    Q and W are sharded like the attention KV heads, so each rank computes only
    its own groups; the shared single-head K and Z stay replicated.
    """

    def __init__(self, config: Step5Config, layer_idx: int, prefix: str) -> None:
        super().__init__()
        sparse = config.sparse_config
        assert sparse is not None
        hidden = config.hidden_size
        self.proxy_dim = sparse.proxy_dim
        self.region_size = sparse.region_block_size
        self.topk = sparse.topk
        self.num_groups = sparse.num_provider_groups
        self.heads_per_group = sparse.indexer_heads_per_group
        self.eps = config.rms_norm_eps

        tp_size = get_tensor_model_parallel_world_size()
        group_shards = min(tp_size, self.num_groups)
        group_rank = get_tensor_model_parallel_rank() * group_shards // tp_size
        self.num_local_groups = self.num_groups // group_shards

        def linear(name: str, out: int, dtype: torch.dtype | None = None):
            return ReplicatedLinear(
                hidden,
                out,
                bias=False,
                params_dtype=dtype,
                quant_config=None,
                prefix=f"{prefix}.{name}",
            )

        def group_linear(name: str, out: int, dtype: torch.dtype | None = None):
            return ColumnParallelLinear(
                hidden,
                out,
                bias=False,
                params_dtype=dtype,
                quant_config=None,
                prefix=f"{prefix}.{name}",
                tp_rank=group_rank,
                tp_size=group_shards,
            )

        heads = self.num_groups * self.heads_per_group
        self.sparse_indexer_q = group_linear("sparse_indexer_q", heads * self.proxy_dim)
        self.sparse_indexer_k = linear("sparse_indexer_k", self.proxy_dim)
        self.sparse_indexer_z = linear("sparse_indexer_z", self.proxy_dim)
        self.sparse_indexer_w = group_linear("sparse_indexer_w", heads, torch.float32)
        self.sparse_indexer_q_norm = nn.Module()
        self.sparse_indexer_q_norm.weight = nn.Parameter(
            torch.zeros(self.proxy_dim, dtype=torch.float32), requires_grad=False
        )
        self.sparse_indexer_k_norm = nn.LayerNorm(
            self.proxy_dim, eps=self.eps, dtype=torch.float32
        )

        theta = config.rope_theta
        rope_parameters = dict(_rope_scaling(config, layer_idx) or {})
        rope_parameters.setdefault("rope_type", "default")
        rope_parameters["rope_theta"] = (
            theta[layer_idx] if isinstance(theta, list) else theta
        )
        rope_parameters["partial_rotary_factor"] = (
            sparse.sparse_indexer_rope_dim / self.proxy_dim
        )
        assert config.max_position_embeddings is not None
        self.rotary_emb = get_rope(
            head_size=self.proxy_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
        )

        vllm_config = get_current_vllm_config()
        self.index_cache = Step5CSAIndexCache(
            proxy_dim=self.proxy_dim,
            cache_config=vllm_config.cache_config,
            vllm_config=vllm_config,
            prefix=f"{prefix}.index_cache",
        )
        self.register_buffer(
            "index_cache_scale", torch.ones((), dtype=torch.float32), persistent=False
        )
        self.summary_cache = Step5CSASummaryCache(
            proxy_dim=self.proxy_dim,
            region_size=self.region_size,
            cache_config=vllm_config.cache_config,
            vllm_config=vllm_config,
            prefix=f"{prefix}.summary_cache",
        )

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(q [T, G, Hg, P], k [T, P], z [T, P], w [T, G, Hg])`` for
        this rank's ``G`` local groups."""
        tokens = hidden_states.shape[0]
        q, _ = self.sparse_indexer_q(hidden_states)
        k, _ = self.sparse_indexer_k(hidden_states)
        z, _ = self.sparse_indexer_z(hidden_states)
        w, _ = self.sparse_indexer_w(hidden_states.float())
        q, k = indexer_norm(
            q.view(tokens, -1, self.proxy_dim),
            k.view(tokens, 1, self.proxy_dim),
            self.sparse_indexer_q_norm.weight,
            self.sparse_indexer_k_norm.weight,
            self.sparse_indexer_k_norm.bias,
            self.eps,
        )
        q, k = self.rotary_emb(positions, q.flatten(1), k.flatten(1))
        weights = w.view(tokens, self.num_local_groups, self.heads_per_group)
        return (
            q.view(tokens, self.num_local_groups, self.heads_per_group, self.proxy_dim),
            k.view(tokens, self.proxy_dim),
            z.view(tokens, self.proxy_dim),
            weights * self.heads_per_group**-0.5,
        )


def _rope_scaling(config: Step5Config, layer_idx: int) -> dict | None:
    yarn_only = getattr(config, "yarn_only_types", None)
    if (
        yarn_only
        and config.layer_types
        and config.layer_types[layer_idx] not in yarn_only
    ):
        return None
    return getattr(config, "rope_scaling", None)


class Step5Attention(Step3p5Attention):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        vllm_config = get_current_vllm_config()
        config = vllm_config.model_config.hf_config.get_text_config()
        self.indexer = None
        if not config.is_sparse_layer(self.layer_idx):
            return
        cache_dtype = kv_cache_dtype_str_to_dtype(
            vllm_config.cache_config.cache_dtype, vllm_config.model_config
        )
        if cache_dtype != torch.bfloat16:
            raise NotImplementedError(
                f"Step-5 sparse attention reads a BF16 KV cache; got {cache_dtype}"
            )
        if not self.use_rope:
            raise NotImplementedError("Step-5 sparse layers require RoPE")
        self.indexer = Step5SparseIndexer(
            config, self.layer_idx, prefix=kwargs["prefix"]
        )
        self.csa_layer_name = f"{kwargs['prefix']}.csa"
        static_context = vllm_config.compilation_config.static_forward_context
        if self.csa_layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.csa_layer_name}")
        static_context[self.csa_layer_name] = self

    def _attn_backend(self) -> type[AttentionBackend] | None:
        # The CSA kernels read this backend's paged K/V layout.
        config = get_current_vllm_config().model_config.hf_config.get_text_config()
        if not config.is_sparse_layer(self.layer_idx):
            return None
        logger.info_once("Step-5 sparse attention layers use ROCM_AITER_UNIFIED_ATTN.")
        return RocmAiterUnifiedAttentionBackend

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.indexer is None:
            return super().forward(positions, hidden_states)
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        query = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        out = torch.empty_like(query)
        attn_name = _encode_layer_name(self.attn.layer_name)
        csa_name = _encode_layer_name(self.csa_layer_name)
        if current_platform.opaque_attention_op():
            kv_dep = torch.ops.vllm.unified_kv_cache_update(k, v, attn_name)
            torch.ops.vllm.step5_csa_attention(
                hidden_states, positions, query, out, csa_name, kv_dep
            )
        else:
            kv_dep = unified_kv_cache_update(k, v, attn_name)
            step5_csa_attention(hidden_states, positions, query, out, csa_name, kv_dep)
        if self.use_head_wise_attn_gate:
            gate, _ = self.g_proj(hidden_states)
            out = out * gate.unsqueeze(-1).sigmoid().to(out.dtype)
        output, _ = self.o_proj(out.view(-1, self.q_size))
        return output


class Step5DecoderLayer(Step3p5DecoderLayer):
    attention_cls = Step5Attention

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        dtype = self.self_attn.o_proj.weight.dtype
        residual = hidden_states
        hidden = self.input_layernorm(residual.to(dtype))
        residual = residual + self.self_attn(positions, hidden).float()
        hidden = self.post_attention_layernorm(residual.to(dtype))
        ffn = self.moe(hidden) if self.use_moe else self.mlp(hidden)
        return residual + ffn.float()


class Step5Model(Step3p5Model):
    decoder_layer_cls = Step5DecoderLayer

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        make_empty = self.make_empty_intermediate_tensors

        def make_empty_fp32(batch_size, dtype, device):
            return make_empty(batch_size, torch.float32, device)

        self.make_empty_intermediate_tensors = make_empty_fp32

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids).float()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.float()
        hidden = super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        if isinstance(hidden, torch.Tensor):
            return hidden.to(self.norm.weight.dtype)
        return hidden


class Step5ForCausalLM(Step3p5ForCausalLM):
    model_cls = Step5Model
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".share_expert.": ".moe.share_expert.",
            ".self_attn.sparse_indexer_": ".self_attn.indexer.sparse_indexer_",
            ".self_attn.ssmax_s": None,
        },
    )


class Step5ForConditionalGeneration(Step3p7ForConditionalGeneration):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        },
        orig_to_new_substr={
            ".attn.in_proj_weight": ".attn.qkv_proj.weight",
            ".attn.in_proj_bias": ".attn.qkv_proj.bias",
            ".mlp.c_fc": ".mlp.fc1",
            ".mlp.c_proj": ".mlp.fc2",
            ".self_attn.sparse_indexer_": ".self_attn.indexer.sparse_indexer_",
            ".self_attn.ssmax_s": None,
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        vllm_config.model_config.hf_config.text_config.architectures = [
            "Step5ForCausalLM"
        ]
        super().__init__(vllm_config=vllm_config, prefix=prefix)
