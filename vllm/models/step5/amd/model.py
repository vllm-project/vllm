# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-5: Step-3.5 with compressed sparse attention (CSA) and an FP32 residual."""

import torch
from torch import nn

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import unified_kv_cache_update
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
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
from vllm.models.step5.common.config import get_step5_sparse_config
from vllm.models.step5.common.csa_attention import step5_csa_attention
from vllm.models.step5.common.csa_cache import Step5CSASummaryCacheLayer
from vllm.models.step5.nvidia.csa_tail_ring import Step5CSATailRing
from vllm.models.step5.nvidia.layernorm import get_step5_norm_dtype
from vllm.models.step5.nvidia.model import (
    _validate_step5_dsa_parallel_geometry,
    _verify_step5_dsa_kv_transfer_compatibility,
)
from vllm.models.step5.nvidia.ops.indexer_norm_rope import indexer_norm_rope
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx1250
from vllm.transformers_utils.configs.step5 import Step5Config
from vllm.utils.math_utils import cdiv, round_up
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
        sparse = get_step5_sparse_config(config)
        assert sparse is not None
        hidden = config.hidden_size
        self.proxy_dim = sparse.proxy_dim
        self.region_size = sparse.region_block_size
        self.topk = sparse.topk
        self.num_groups = sparse.index_tp_size
        self.heads_per_group = sparse.sparse_indexer_num_heads // self.num_groups
        self.eps = config.rms_norm_eps
        self.zero_centered = config.zero_centered

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
        self.sparse_indexer_w = group_linear("sparse_indexer_w", heads)
        self.sparse_indexer_q_norm = nn.Module()
        self.sparse_indexer_q_norm.weight = nn.Parameter(
            torch.zeros(self.proxy_dim), requires_grad=False
        )
        self.sparse_indexer_k_norm = nn.LayerNorm(self.proxy_dim, eps=self.eps)

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

        self.native_rope = not self.rotary_emb.enabled()
        vllm_config = get_current_vllm_config()
        self.topk_backend = vllm_config.kernel_config.sparse_indexer_topk_backend
        spec = vllm_config.speculative_config
        self.num_speculative_tokens = spec.num_speculative_tokens if spec else 0
        self.max_regions = cdiv(
            vllm_config.model_config.max_model_len, self.region_size
        )
        self.num_sms = current_platform.num_compute_units()
        # Peak scoring workspace: decode scores or bounded prefill logits, plus
        # the per-step prefill page table.
        sched = vllm_config.scheduler_config
        decode_rows = (
            sched.max_num_seqs
            * (1 + self.num_speculative_tokens)
            * self.num_local_groups
        )
        decode_scores = decode_rows * round_up(self.max_regions, 256) * 4
        prefill_table = sched.max_num_batched_tokens * cdiv(self.max_regions, 128) * 16
        self.workspace_bytes = (
            max(envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB << 20, decode_scores)
            + prefill_table
        )

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, decode=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(q [T, G, 8, P] fp8, k [T, 1, P], z [T, 1, P],
        w [T, G, 8])`` for this rank's ``G`` local groups, heads padded to 8.

        With ``decode=(summary cache, ring, slots)`` it also stores each
        token's open-region row and summarizes the regions it closes.
        """
        q, _ = self.sparse_indexer_q(hidden_states)
        k, _ = self.sparse_indexer_k(hidden_states)
        z, _ = self.sparse_indexer_z(hidden_states)
        w, _ = self.sparse_indexer_w(hidden_states)
        fused = {}
        if decode is not None:
            cache, ring, slots = decode
            fused = dict(
                decode_cache=cache, decode_ring=ring, decode_z=z, decode_slots=slots
            )
        q, k, w = indexer_norm_rope(
            q,
            k,
            w,
            self.sparse_indexer_q_norm.weight,
            self.sparse_indexer_k_norm.weight,
            self.sparse_indexer_k_norm.bias,
            self.rotary_emb.cos_sin_cache,
            positions,
            self.eps,
            self.zero_centered,
            native_rope=self.native_rope,
            num_groups=self.num_local_groups,
            **fused,
        )
        return q, k, z.view(-1, 1, self.proxy_dim), w


def _layer_types(config: Step5Config) -> list[str]:
    # `layer_types` covers only the dense stack; MTP layers index past it.
    return config.layer_types_with_mtp or config.layer_types or []


def _is_sparse_layer(config: Step5Config, layer_idx: int) -> bool:
    sparse = get_step5_sparse_config(config)
    layer_types = _layer_types(config)
    return (
        sparse is not None
        and layer_idx < len(layer_types)
        and layer_types[layer_idx] in sparse.apply_to_layer_types
    )


def _rope_scaling(config: Step5Config, layer_idx: int) -> dict | None:
    yarn_only = getattr(config, "yarn_only_types", None)
    layer_types = _layer_types(config)
    if yarn_only and layer_types and layer_types[layer_idx] not in yarn_only:
        return None
    return getattr(config, "rope_scaling", None)


class Step5Attention(Step3p5Attention):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        vllm_config = get_current_vllm_config()
        config = vllm_config.model_config.hf_config.get_text_config()
        self.indexer = None
        if not _is_sparse_layer(config, self.layer_idx):
            return
        cache_dtype = vllm_config.cache_config.cache_dtype
        if (
            cache_dtype not in ("fp8", "fp8_e4m3")
            and kv_cache_dtype_str_to_dtype(cache_dtype, vllm_config.model_config)
            != torch.bfloat16
        ):
            raise NotImplementedError(
                "Step-5 sparse attention reads a BF16 or FP8 E4M3 KV cache; "
                f"got {cache_dtype}"
            )
        if not self.use_rope:
            raise NotImplementedError("Step-5 sparse layers require RoPE")
        if self.head_dim != 192:
            raise ValueError(
                f"Step-5 sparse attention requires head_dim 192, got {self.head_dim}"
            )
        if vllm_config.parallel_config.use_ubatching:
            raise ValueError(
                "Step-5 sparse attention does not support dual-batch overlap or "
                "microbatching yet. Disable --enable-dbo and set --ubatch-size 1."
            )
        _verify_step5_dsa_kv_transfer_compatibility(vllm_config)
        if rocm_aiter_ops.is_shuffle_kv_cache_enabled() and on_gfx1250():
            raise NotImplementedError(
                "Step-5 sparse attention reads the unshuffled KV cache layout."
            )
        sparse = get_step5_sparse_config(config)
        assert sparse is not None
        _validate_step5_dsa_parallel_geometry(
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            indexer_num_heads=sparse.sparse_indexer_num_heads,
            index_tp_size=sparse.index_tp_size,
            tp_size=get_tensor_model_parallel_world_size(),
        )
        self.indexer = Step5SparseIndexer(
            config, self.layer_idx, prefix=kwargs["prefix"]
        )
        static_context = vllm_config.compilation_config.static_forward_context
        self.sparse_summary_cache = Step5CSASummaryCacheLayer(
            prefix=f"{kwargs['prefix']}.attn.sparse_summary_cache",
            target_impl=self.attn.impl,
            static_forward_context=static_context,
        )
        self.csa_tail_ring = Step5CSATailRing(
            prefix=f"{kwargs['prefix']}.attn.csa_tail_ring",
            main_layer_name=self.attn.layer_name,
            num_speculative_tokens=self.indexer.num_speculative_tokens,
            static_forward_context=static_context,
        )
        self.csa_layer_name = f"{kwargs['prefix']}.csa"
        if self.csa_layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.csa_layer_name}")
        static_context[self.csa_layer_name] = self

    def _attn_backend(self) -> type[AttentionBackend] | None:
        # The CSA kernels read this backend's paged K/V layout.
        config = get_current_vllm_config().model_config.hf_config.get_text_config()
        if not _is_sparse_layer(config, self.layer_idx):
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
        gate = None
        if self.use_head_wise_attn_gate:
            gate, _ = self.g_proj(hidden_states)
        attn_name = _encode_layer_name(self.attn.layer_name)
        csa_name = _encode_layer_name(self.csa_layer_name)
        if current_platform.opaque_attention_op():
            kv_dep = torch.ops.vllm.unified_kv_cache_update(k, v, attn_name)
            torch.ops.vllm.step5_csa_attention(
                hidden_states, positions, query, out, csa_name, gate, kv_dep
            )
        else:
            kv_dep = unified_kv_cache_update(k, v, attn_name)
            step5_csa_attention(
                hidden_states, positions, query, out, csa_name, gate, kv_dep
            )
        output, _ = self.o_proj(out.view(-1, self.q_size))
        return output


def _use_fp32_norm_weights(module: nn.Module, config: Step5Config) -> None:
    """The checkpoint keeps every RMSNorm weight in ``norm_dtype``."""
    if getattr(config, "norm_dtype", None) is None:
        return
    if get_step5_norm_dtype(config) != torch.float32:
        return
    for norm in module.modules():
        if isinstance(norm, GemmaRMSNorm):
            norm.weight.data = norm.weight.data.float()


class Step5DecoderLayer(Step3p5DecoderLayer):
    attention_cls = Step5Attention

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config, prefix=prefix)
        self.act_dtype = vllm_config.model_config.dtype

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden = self.input_layernorm(residual).to(self.act_dtype)
        residual = residual + self.self_attn(positions, hidden).float()
        hidden = self.post_attention_layernorm(residual).to(self.act_dtype)
        ffn = self.moe(hidden) if self.use_moe else self.mlp(hidden)
        return residual + ffn.float()


class Step5Model(Step3p5Model):
    decoder_layer_cls = Step5DecoderLayer

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _use_fp32_norm_weights(self, vllm_config.model_config.hf_text_config)
        self.act_dtype = vllm_config.model_config.dtype
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
            return hidden.to(self.act_dtype)
        return hidden


class Step5ForCausalLM(Step3p5ForCausalLM):
    model_cls = Step5Model
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".share_expert.": ".moe.share_expert.",
            ".self_attn.sparse_indexer_": ".self_attn.indexer.sparse_indexer_",
            ".self_attn.ssmax_s": None,
        },
        orig_to_new_prefix={"vision_model.": None, "vit_large_projector.": None},
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
