# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next model."""

import os
import time
from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    VllmConfig,
    get_current_vllm_config_or_none,
)
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
)
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3NextRMSNorm,
)
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    QKVParallelLinearOverlappingGQA,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.attention.head_partition import (
    make_attention_head_partition,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backend import AttentionType

from .interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def _experimental_replicate_uneven_full_attention() -> bool:
    return os.environ.get(
        "VLLM_EXPERIMENTAL_REPLICATE_UNEVEN_FULL_ATTENTION", ""
    ).lower() in ("1", "true", "yes", "on")


def _diagnostic_decoder_layer_full_boundary() -> bool:
    return os.environ.get(
        "VLLM_QWEN3_NEXT_DIAGNOSTIC_DECODER_LAYER_FULL_OP", ""
    ).lower() in ("1", "true", "yes", "on")


def _diagnostic_final_norm_full_boundary() -> bool:
    return os.environ.get(
        "VLLM_QWEN3_NEXT_DIAGNOSTIC_FINAL_NORM_FULL_OP", ""
    ).lower() in ("1", "true", "yes", "on")


def _decode_context_parallel_size() -> int:
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return 1
    return vllm_config.parallel_config.decode_context_parallel_size


def _should_replicate_full_attention_heads(
    total_num_heads: int,
    total_num_kv_heads: int,
    tp_size: int,
    dcp_size: int,
    enable_dcp_replicated_full_attention: bool,
) -> tuple[bool, bool]:
    use_overlapping_gqa = (
        total_num_heads % tp_size == 0
        and total_num_kv_heads % tp_size != 0
        and tp_size % total_num_kv_heads != 0
    )
    return (
        total_num_heads % tp_size != 0,
        use_overlapping_gqa,
    )


KVCache = tuple[torch.Tensor, torch.Tensor]


def _is_shared_expert_fse_compatible(quant_config) -> bool:
    """Check if shared expert can be fused with routed experts.

    FSE requires that shared and routed expert weights use the same
    quantization format. Returns False when the shared expert is
    excluded from quantization (e.g. float32 shared in an MXFP4 model)
    or has a different quant spec than routed experts.
    """
    if quant_config is None:
        return True
    # Quark stores its full config dict in quant_config.quant_config
    raw_config = getattr(quant_config, "quant_config", None)
    if not isinstance(raw_config, dict):
        return True
    exclude = raw_config.get("exclude", [])
    if not exclude:
        return True
    return not any("shared_expert." in str(e) for e in exclude)


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )

        _fse_requested = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        _fse_enabled = _fse_requested and _is_shared_expert_fse_compatible(quant_config)
        if _fse_requested and not _fse_enabled:
            logger.warning(
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled but "
                "shared expert has a different quantization spec than routed "
                "experts. Falling back to non-fused shared expert path."
            )
        if _fse_enabled or config.shared_expert_intermediate_size <= 0:
            self.shared_expert = None
        else:
            self.shared_expert = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                is_sequence_parallel=self.is_sequence_parallel,
                prefix=f"{prefix}.shared_expert",
            )

        self.experts = FusedMoE(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=1 if self.shared_expert is None else None,
            shared_expert_gate=self.shared_expert_gate
            if self.shared_expert is None
            else None,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the FusedMoE class
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(orig_shape)


class Qwen3NextAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3NextConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        dcp_size = _decode_context_parallel_size()
        (
            should_replicate_full_attention,
            use_overlapping_gqa,
        ) = _should_replicate_full_attention_heads(
            self.total_num_heads,
            self.total_num_kv_heads,
            tp_size,
            dcp_size,
            envs.VLLM_EXPERIMENTAL_DCP_REPLICATED_FULL_ATTENTION,
        )
        self.replicate_uneven_full_attention = (
            should_replicate_full_attention
            and _experimental_replicate_uneven_full_attention()
        )
        self.dcp_full_kv_attention_heads = (
            use_overlapping_gqa
            and dcp_size > 1
            and envs.VLLM_EXPERIMENTAL_DCP_REPLICATED_FULL_ATTENTION
            and not self.replicate_uneven_full_attention
        )
        if self.replicate_uneven_full_attention:
            reason = (
                "num_attention_heads is not divisible"
                if self.total_num_heads % tp_size != 0
                else "DCP with overlapping GQA KV-head partition"
            )
            logger.warning(
                "Replicating Qwen3Next full-attention layer %s on every TP rank "
                "because %s requires full local Q/KV head layout "
                "(num_attention_heads=%d, num_key_value_heads=%d, "
                "tensor_parallel_size=%d, decode_context_parallel_size=%d).",
                prefix,
                reason,
                self.total_num_heads,
                self.total_num_kv_heads,
                tp_size,
                dcp_size,
            )
            self.num_heads = self.total_num_heads
        else:
            assert self.total_num_heads % tp_size == 0
            self.num_heads = self.total_num_heads // tp_size
        self.attn_head_partition = None
        use_overlapping_gqa = (
            use_overlapping_gqa
            and not self.replicate_uneven_full_attention
            and not self.dcp_full_kv_attention_heads
        )
        if self.dcp_full_kv_attention_heads:
            # DCP sequence merging requires every DCP rank to expose the same
            # KV-head layout for its sequence shard. Keep Q/O tensor-parallel
            # sharded, but load all KV heads on each sequence shard.
            self.attn_head_partition = make_attention_head_partition(
                total_num_heads=self.total_num_heads,
                total_num_kv_heads=self.total_num_kv_heads,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
            self.num_kv_heads = self.total_num_kv_heads
        elif use_overlapping_gqa:
            self.attn_head_partition = make_attention_head_partition(
                total_num_heads=self.total_num_heads,
                total_num_kv_heads=self.total_num_kv_heads,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
            self.num_kv_heads = self.attn_head_partition.num_kv_heads
        elif self.replicate_uneven_full_attention:
            self.num_kv_heads = self.total_num_kv_heads
        elif self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
            self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
            self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        qkv_proj_cls = (
            QKVParallelLinearOverlappingGQA
            if use_overlapping_gqa or self.dcp_full_kv_attention_heads
            else QKVParallelLinear
        )
        qkv_kwargs = {}
        if self.dcp_full_kv_attention_heads:
            qkv_kwargs["kv_head_indices"] = tuple(range(self.total_num_kv_heads))
        elif self.attn_head_partition is not None:
            qkv_kwargs["kv_head_indices"] = self.attn_head_partition.kv_head_indices
        self.qkv_proj = qkv_proj_cls(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=self.replicate_uneven_full_attention,
            **qkv_kwargs,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            disable_tp=self.replicate_uneven_full_attention,
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            dual_chunk_attention_config=self.dual_chunk_attention_config,
        )

        # Late-interaction retrieval models (e.g. ColQwen3.5) run BIDIRECTIONAL
        # attention on the full_attention layers; they set config.is_causal=False
        # via a VerifyAndUpdateConfig handler. Generation models leave is_causal
        # unset (-> causal/DECODER), so this is a no-op for them. Mirrors qwen3.py.
        attn_type = (
            AttentionType.DECODER
            if getattr(config, "is_causal", True)
            else AttentionType.ENCODER_ONLY
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": self.dual_chunk_attention_config,
            }
            if self.dual_chunk_attention_config
            else {},
        )
        self.attn.dcp_replicated_full_attention_heads = (
            self.replicate_uneven_full_attention
        )
        self.attn.dcp_full_kv_attention_heads = self.dcp_full_kv_attention_heads
        if self.dcp_full_kv_attention_heads:
            assert self.attn_head_partition is not None
            self.attn.dcp_local_kv_head_indices = (
                self.attn_head_partition.kv_head_indices
            )

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None:
            compilation_config = vllm_config.compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            compilation_config.static_forward_context[prefix] = self

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Fuse the gated split + QK-RMSNorm + (partial) NeoX RoPE + gate copy.
        # TODO: support MRoPE
        mm_config = model_config.multimodal_config if model_config else None
        text_only = mm_config is None or mm_config.language_model_only
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and text_only
        )

    def _project_qkv_gate(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return post-norm, post-RoPE (q, k, v) and the pre-sigmoid gate.

        Dispatches between the fused Triton kernel and the eager
        split + QK-RMSNorm + RoPE path. ``gate`` is ``None`` when output
        gating is disabled.
        """
        if self.use_fused_qk_norm_rope_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            # mRoPE passes positions as (3, n_tokens) for T/H/W. Fusion is only
            # enabled text-only, where the three rows are identical, so taking
            # the T row is exact. (1D positions pass through.)
            pos = positions[0] if positions.ndim == 2 else positions
            q, k, gate = fused_qk_rmsnorm_rope_gate(
                q_gate,
                k,
                self.q_norm.weight.float() + 1.0,
                self.k_norm.weight.float() + 1.0,
                self.rotary_emb.cos_sin_cache,
                pos,
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_emb.rotary_dim,
            )
            return q, k, v, gate

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v, gate

    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        torch.ops.vllm.qwen3_next_attention_full_forward(
            positions,
            output,
            hidden_states,
            layer_name=_encode_layer_name(self.prefix),
        )

    def _forward_impl(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        attn_output = self.attn(q, k, v)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        output[:], _ = self.o_proj(attn_output)


def qwen3_next_attention_full_forward(
    positions: torch.Tensor,
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Full Qwen full-attention block wrapped as a custom op."""
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_impl(
        positions=positions,
        output=output,
        hidden_states=hidden_states,
    )


def qwen3_next_attention_full_forward_fake(
    positions: torch.Tensor,
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen3_next_attention_full_forward",
    op_func=qwen3_next_attention_full_forward,
    mutates_args=["output"],
    fake_impl=qwen3_next_attention_full_forward_fake,
)


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        self.prefix = prefix

        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=True,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (self.layer_idx not in mlp_only_layers) and (
            config.num_experts > 0
            and (self.layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None:
            compilation_config = vllm_config.compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            compilation_config.static_forward_context[prefix] = self

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        if _diagnostic_decoder_layer_full_boundary():
            hidden_out = torch.empty_like(hidden_states)
            residual_out = torch.empty_like(hidden_states)
            torch.ops.vllm.qwen3_next_decoder_layer_full_forward(
                hidden_states,
                residual,
                positions,
                hidden_out,
                residual_out,
                layer_name=_encode_layer_name(self.prefix),
            )
            return hidden_out, residual_out

        return self._forward_impl(
            hidden_states=hidden_states,
            residual=residual,
            positions=positions,
        )

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
    ):
        trace_enabled = (
            os.environ.get("AG2_VLLM_LAYER_PHASE_TRACE") == "1"
            and _diagnostic_decoder_layer_full_boundary()
        )
        trace_sync = os.environ.get("AG2_VLLM_LAYER_PHASE_TRACE_SYNC") == "1"
        trace_min_tokens = int(
            os.environ.get("AG2_VLLM_LAYER_PHASE_TRACE_MIN_TOKENS", "8192")
        )
        token_count = int(hidden_states.shape[0])

        def mark() -> float:
            if trace_enabled and trace_sync and token_count >= trace_min_tokens:
                torch.cuda.synchronize()
            return time.perf_counter()

        t0 = mark() if trace_enabled and token_count >= trace_min_tokens else 0.0
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        t1 = mark() if trace_enabled and token_count >= trace_min_tokens else 0.0

        self_attention_output = torch.empty_like(hidden_states)
        if self.layer_type == "linear_attention":
            self.linear_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
            )
        elif self.layer_type == "full_attention":
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output
        t2 = mark() if trace_enabled and token_count >= trace_min_tokens else 0.0

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        t3 = mark() if trace_enabled and token_count >= trace_min_tokens else 0.0
        hidden_states = self.mlp(hidden_states)
        t4 = mark() if trace_enabled and token_count >= trace_min_tokens else 0.0

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        if trace_enabled and token_count >= trace_min_tokens:
            logger.warning(
                "AG2_LAYER_PHASE_TRACE layer=%s type=%s tokens=%d "
                "input_norm=%.6f attention=%.6f post_norm=%.6f mlp=%.6f "
                "total=%.6f",
                self.prefix,
                self.layer_type,
                token_count,
                t1 - t0,
                t2 - t1,
                t3 - t2,
                t4 - t3,
                t4 - t0,
            )
        return hidden_states, residual


def qwen3_next_decoder_layer_full_forward(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    hidden_out: torch.Tensor,
    residual_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Diagnostic full decoder-layer custom op for compile isolation."""
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    trace_enabled = (
        os.environ.get("AG2_VLLM_LAYER_TRACE") == "1"
        and not torch.cuda.is_current_stream_capturing()
    )
    trace_sync = os.environ.get("AG2_VLLM_LONG_PREFILL_TRACE_SYNC") == "1"
    trace_min_tokens = int(os.environ.get("AG2_VLLM_LAYER_TRACE_MIN_TOKENS", "8192"))
    token_count = int(hidden_states.shape[0])
    if trace_enabled and token_count >= trace_min_tokens and trace_sync:
        torch.cuda.synchronize()
    trace_t0 = time.perf_counter() if trace_enabled else 0.0
    next_hidden, next_residual = self._forward_impl(
        hidden_states=hidden_states,
        residual=residual,
        positions=positions,
    )
    if trace_enabled and token_count >= trace_min_tokens and trace_sync:
        torch.cuda.synchronize()
    trace_t1 = time.perf_counter() if trace_enabled else 0.0
    hidden_out.copy_(next_hidden)
    residual_out.copy_(next_residual)
    if trace_enabled and token_count >= trace_min_tokens:
        if trace_sync:
            torch.cuda.synchronize()
        trace_t2 = time.perf_counter()
        logger.warning(
            "AG2_LAYER_TRACE layer=%s type=%s tokens=%d forward=%.6f copy=%.6f",
            layer_name,
            self.layer_type,
            token_count,
            trace_t1 - trace_t0,
            trace_t2 - trace_t1,
        )


def qwen3_next_decoder_layer_full_forward_fake(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    hidden_out: torch.Tensor,
    residual_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen3_next_decoder_layer_full_forward",
    op_func=qwen3_next_decoder_layer_full_forward,
    mutates_args=["hidden_out", "residual_out"],
    fake_impl=qwen3_next_decoder_layer_full_forward_fake,
)


@support_torch_compile
class Qwen3NextModel(nn.Module, EagleModelMixin):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            # weight_name: (param_name, shard_id)
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_expert.gate_proj": (".shared_expert.gate_up_proj", 0),
            ".shared_expert.up_proj": (".shared_expert.gate_up_proj", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Qwen3NextConfig = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        self.prefix = prefix

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3NextDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        vllm_config_for_context = get_current_vllm_config_or_none()
        if vllm_config_for_context is not None:
            compilation_config = vllm_config_for_context.compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            compilation_config.static_forward_context[prefix] = self

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if _diagnostic_final_norm_full_boundary():
            norm_out = torch.empty_like(hidden_states)
            torch.ops.vllm.qwen3_next_final_norm_forward(
                hidden_states,
                residual,
                norm_out,
                layer_name=_encode_layer_name(self.prefix),
            )
            hidden_states = norm_out
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = self.hf_to_vllm_mapper
        if rocm_aiter_ops.is_fusion_moe_shared_experts_enabled():
            # AITER fused-shared-experts: route the shared_expert checkpoint
            # weights into the extra fused expert slot. Merge (not mutate) so the
            # shared class mapper isn't permanently altered.
            num_routed = getattr(self.config, "num_experts", 0)
            mapper = mapper | WeightsMapper(
                orig_to_new_substr={"mlp.shared_expert.": f"mlp.experts.{num_routed}."}
            )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)


def qwen3_next_final_norm_forward(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Diagnostic final RMSNorm custom op for compile isolation."""
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    next_hidden, _ = self.norm(hidden_states, residual)
    output.copy_(next_hidden)


def qwen3_next_final_norm_forward_fake(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen3_next_final_norm_forward",
    op_func=qwen3_next_final_norm_forward,
    mutates_args=["output"],
    fake_impl=qwen3_next_final_norm_forward_fake,
)


class QwenNextMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, Qwen3NextDecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No Qwen3Next layer found in the model.layers.")

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


class Qwen3NextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    QwenNextMixtureOfExperts,
    IsHybrid,
    SupportsEagle3,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkvz"],
        "in_proj_ba": ["in_proj_ba"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3Next currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        if (
            hf_config.linear_num_key_heads % tp_size != 0
            or hf_config.linear_num_value_heads % tp_size != 0
        ):
            tp_size = 1
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        return loader.load_weights(weights)
