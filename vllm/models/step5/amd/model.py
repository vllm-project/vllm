# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 on ROCm: compressed sparse attention (CSA) and an FP32 residual."""

from collections.abc import Iterable

import torch
from torch import nn

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import SiluAndMul, SwigluStepAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import unified_kv_cache_update
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
    skip_spec_layers,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx1250
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.step5 import Step5Config
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import _encode_layer_name, kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
    RocmAiterUnifiedAttentionBackend,
)

from ..common.config import get_step5_sparse_config
from ..common.csa_tail_ring import Step5CSATailRing
from ..common.ops.indexer_norm_rope import indexer_norm_rope
from .checks import validate_parallel_geometry, verify_kv_transfer_compatibility
from .csa_attention import step5_csa_attention
from .csa_cache import Step5CSASummaryCacheLayer


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

        def linear(name: str, out: int):
            return ReplicatedLinear(
                hidden, out, bias=False, quant_config=None, prefix=f"{prefix}.{name}"
            )

        def group_linear(name: str, out: int):
            return ColumnParallelLinear(
                hidden,
                out,
                bias=False,
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

        self.rotary_emb = get_rope(
            head_size=self.proxy_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=_rope_parameters(config, layer_idx)
            | {
                "partial_rotary_factor": sparse.sparse_indexer_rope_dim / self.proxy_dim
            },
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


def _num_attention_heads(config: Step5Config, layer_type: str) -> int:
    if layer_type == "sliding_attention":
        if num_heads := getattr(config, "num_sliding_attention_heads", None):
            return num_heads
        other = config.attention_other_setting or {}
        if other.get("attention_type") == layer_type:
            return other["num_attention_heads"]
    return config.num_attention_heads


def _rope_parameters(config: Step5Config, layer_idx: int) -> dict:
    rope_parameters = dict(_rope_scaling(config, layer_idx) or {})
    rope_parameters.setdefault("rope_type", "default")
    theta = config.rope_theta
    rope_parameters["rope_theta"] = (
        theta[layer_idx] if isinstance(theta, list) else theta
    )
    factors = config.partial_rotary_factors
    rope_parameters["partial_rotary_factor"] = factors[layer_idx] if factors else 1.0
    return rope_parameters


def _is_moe_layer(config: Step5Config, layer_idx: int) -> bool:
    if (moe_layers := getattr(config, "moe_layers_enum", None)) is not None:
        return layer_idx in {int(i) for i in moe_layers.strip().split(",")}
    return 0 < layer_idx < config.num_hidden_layers


def _swiglu_limit(limits: list | None, layer_idx: int) -> float | None:
    limit = limits[layer_idx] if limits and layer_idx < len(limits) else None
    return float(limit) if limit else None


class Step5MLP(nn.Module):
    def __init__(
        self,
        config: Step5Config,
        intermediate_size: int,
        quant_config,
        reduce_results: bool,
        prefix: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        limit = _swiglu_limit(config.swiglu_limits_shared, extract_layer_index(prefix))
        self.act_fn = SwigluStepAndMul(limit=limit) if limit else SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        output, _ = self.down_proj(self.act_fn(gate_up))
        return output


class Step5MoE(nn.Module):
    """Routed experts with the shared expert fused into one FusedMoE call, so
    the combined output is all-reduced once in the model dtype."""

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config
        layer_idx = extract_layer_index(prefix)
        self.n_routed_experts = config.moe_num_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        if self.n_redundant_experts:
            raise NotImplementedError(
                "Step5 on ROCm does not support redundant experts"
            )
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = (
            self.n_physical_experts // get_ep_group().device_group.size()
        )
        if not (config.use_moe_router_bias and config.need_fp32_gate):
            raise ValueError("Step5 MoE requires an FP32 gate with a router bias.")
        limit = _swiglu_limit(config.swiglu_limits, layer_idx)
        if limit not in (None, 7.0):
            raise ValueError(f"Step5 fused MoE supports swiglu_limit 7.0, got {limit}")

        self.gate = GateLinear(
            config.hidden_size,
            config.moe_num_experts,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.router_bias = nn.Parameter(
            torch.zeros(config.moe_num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.share_expert = Step5MLP(
            config,
            config.share_expert_dim,
            vllm_config.quant_config,
            reduce_results=False,
            prefix=f"{prefix}.share_expert",
        )
        self.experts = FusedMoEFactory(
            shared_experts=self.share_expert,
            gate=self.gate,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_expert_weight,
            quant_config=vllm_config.quant_config,
            activation="swiglustep" if limit else "silu",
            prefix=f"{prefix}.experts",
            scoring_func=config.moe_router_activation,
            e_score_correction_bias=self.router_bias,
            routed_scaling_factor=config.moe_router_scaling_factor,
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            router_logits_dtype=torch.float32,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.experts(hidden_states=hidden_states, router_logits=hidden_states)


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


class Step5Attention(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.layer_idx = extract_layer_index(prefix)
        layer_type = _layer_types(config)[self.layer_idx]
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = _num_attention_heads(config, layer_type)
        self.total_num_kv_heads = config.num_attention_groups
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        use_rope_layers = getattr(config, "use_rope_layers", None)
        self.use_rope = use_rope_layers[self.layer_idx] if use_rope_layers else True
        self.use_head_wise_attn_gate = getattr(config, "use_head_wise_attn_gate", False)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        if self.use_head_wise_attn_gate:
            self.g_proj = ColumnParallelLinear(
                config.hidden_size,
                self.total_num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
        self.q_norm = GemmaRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, config.rms_norm_eps)
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=_rope_parameters(config, self.layer_idx),
        )

        sparse = _is_sparse_layer(config, self.layer_idx)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            per_layer_sliding_window=getattr(config, "sliding_window", None)
            if layer_type == "sliding_attention"
            else None,
            # The CSA kernels read this backend's paged K/V layout.
            attn_backend=RocmAiterUnifiedAttentionBackend if sparse else None,
        )

        self.indexer = None
        if not sparse:
            return
        self._validate_sparse(vllm_config)
        self.indexer = Step5SparseIndexer(config, self.layer_idx, prefix=prefix)
        static_context = vllm_config.compilation_config.static_forward_context
        self.sparse_summary_cache = Step5CSASummaryCacheLayer(
            prefix=f"{prefix}.attn.sparse_summary_cache",
            target_impl=self.attn.impl,
            static_forward_context=static_context,
        )
        self.csa_tail_ring = Step5CSATailRing(
            prefix=f"{prefix}.attn.csa_tail_ring",
            main_layer_name=self.attn.layer_name,
            num_speculative_tokens=self.indexer.num_speculative_tokens,
            static_forward_context=static_context,
        )
        self.csa_layer_name = f"{prefix}.csa"
        if self.csa_layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.csa_layer_name}")
        static_context[self.csa_layer_name] = self

    def _validate_sparse(self, vllm_config: VllmConfig) -> None:
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
        verify_kv_transfer_compatibility(vllm_config)
        if rocm_aiter_ops.is_shuffle_kv_cache_enabled() and on_gfx1250():
            raise NotImplementedError(
                "Step-5 sparse attention reads the unshuffled KV cache layout."
            )
        sparse = get_step5_sparse_config(vllm_config.model_config.hf_config)
        assert sparse is not None
        validate_parallel_geometry(
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            indexer_num_heads=sparse.sparse_indexer_num_heads,
            index_tp_size=sparse.index_tp_size,
            tp_size=get_tensor_model_parallel_world_size(),
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)
        if self.use_rope:
            q, k = self.rotary_emb(positions, q, k)
        gate = None
        if self.use_head_wise_attn_gate:
            gate, _ = self.g_proj(hidden_states)
        if self.indexer is None:
            out = self.attn(q, k, v)
            if gate is not None:
                out = (
                    out.view(-1, self.num_heads, self.head_dim)
                    * gate.unsqueeze(-1).sigmoid()
                ).view(out.shape)
            output, _ = self.o_proj(out)
            return output

        query = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        out = torch.empty_like(query)
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


def use_fp32_norm_weights(module: nn.Module, config: Step5Config) -> None:
    """The checkpoint keeps every RMSNorm weight in ``norm_dtype``."""
    norm_dtype = (getattr(config, "norm_dtype", None) or "").lower()
    if norm_dtype not in ("fp32", "float32", "float"):
        return
    for norm in module.modules():
        if isinstance(norm, GemmaRMSNorm):
            norm.weight.data = norm.weight.data.float()


class Step5DecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        layer_idx = extract_layer_index(prefix)
        if vllm_config.cache_config is not None:
            vllm_config.cache_config.sliding_window = None
        self.self_attn = Step5Attention(vllm_config, prefix=f"{prefix}.self_attn")
        self.use_moe = _is_moe_layer(config, layer_idx)
        if self.use_moe:
            self.moe = Step5MoE(vllm_config, prefix=f"{prefix}.moe")
        else:
            self.mlp = Step5MLP(
                config,
                config.intermediate_size,
                vllm_config.quant_config,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
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


_STACKED_PARAMS: tuple[tuple[str, str, str | int], ...] = (
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)
# The checkpoint packs each MoE projection as one [experts, out, in] tensor.
_EXPERT_PARAMS = tuple(
    (f"moe.experts.routed_experts.{param}{suffix}", f"moe.{proj}.{suffix}", shard)
    for proj, param, shard in (
        ("gate_proj", "w13_", "w1"),
        ("up_proj", "w13_", "w3"),
        ("down_proj", "w2_", "w2"),
    )
    for suffix in ("weight_scale_inv", "weight")
)


def load_step5_weights(
    model: nn.Module, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    params = dict(model.named_parameters())
    loaded: set[str] = set()
    for name, weight in weights:
        for param_name, ckpt_name, shard in _EXPERT_PARAMS:
            if ckpt_name in name:
                name = name.replace(ckpt_name, param_name)
                if is_pp_missing_parameter(name, model):
                    break
                param = params[name]
                for expert_id in range(weight.shape[0]):
                    param.weight_loader(
                        param,
                        weight[expert_id],
                        name,
                        shard_id=shard,
                        expert_id=expert_id,
                    )
                loaded.add(name)
                break
        else:
            for param_name, ckpt_name, shard_id in _STACKED_PARAMS:
                if ckpt_name not in name or ".moe.experts." in name:
                    continue
                name = name.replace(ckpt_name, param_name)
                if is_pp_missing_parameter(name, model):
                    break
                if name not in params:
                    raise ValueError(f"Unsupported Step5 checkpoint tensor: {name}")
                params[name].weight_loader(params[name], weight, shard_id)
                loaded.add(name)
                break
            else:
                if is_pp_missing_parameter(name, model):
                    continue
                if name not in params:
                    raise ValueError(f"Unsupported Step5 checkpoint tensor: {name}")
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
                loaded.add(name)
    return loaded


@support_torch_compile
class Step5Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.moe_num_experts = config.moe_num_experts
        self.act_dtype = vllm_config.model_config.dtype
        if config.tie_word_embeddings:
            raise NotImplementedError("Step5 does not tie word embeddings")
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Step5DecoderLayer(vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        use_fp32_norm_weights(self, config)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        # The residual stream stays FP32 across pipeline stages.
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size),
                    dtype=torch.float32,
                    device=device,
                )
            }
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids).float()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds.float()
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states = layer(positions, hidden_states)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        return hidden_states.to(self.act_dtype)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return load_step5_weights(self, skip_spec_layers(weights, self.config))


class Step5ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".share_expert.": ".moe.share_expert.",
            ".self_attn.sparse_indexer_": ".self_attn.indexer.sparse_indexer_",
            ".self_attn.ssmax_s": None,
        },
        orig_to_new_prefix={"vision_model.": None, "vit_large_projector.": None},
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.model = Step5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        moe = [
            layer.moe
            for layer in self.model.layers
            if isinstance(layer, Step5DecoderLayer) and layer.use_moe
        ]
        self.moe_layers = [m.experts for m in moe]
        self.num_moe_layers = len(moe)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        # A pipeline stage may hold only dense layers.
        example = moe[0] if moe else None
        self.num_logical_experts = example.n_logical_experts if example else 0
        self.num_physical_experts = example.n_physical_experts if example else 0
        self.num_local_physical_experts = (
            example.n_local_physical_experts if example else 0
        )
        self.num_routed_experts = example.n_routed_experts if example else 0
        self.num_redundant_experts = example.n_redundant_experts if example else 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer, Step5DecoderLayer) and layer.use_moe:
                layer.moe.n_physical_experts = num_physical_experts
                layer.moe.n_redundant_experts = self.num_redundant_experts
                layer.moe.experts.update_expert_map()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
