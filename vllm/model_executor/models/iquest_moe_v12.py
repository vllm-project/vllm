# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Inference-only IquestMoe model compatible with HuggingFace weights.
# Architecture: MoE with always-enabled shared expert, QK-norm attention,
# top-K then softmax routing. Same structure as Qwen3MoE with shared expert.

import typing
from collections.abc import Callable, Iterable
from dataclasses import replace
from itertools import islice
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors

from .interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def _iquest_moe_topk_then_softmax_routing(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_logits, selected_experts = torch.topk(gating_output, topk, dim=-1)
    routing_weights = torch.softmax(topk_logits, dim=-1, dtype=torch.float32)
    if renormalize:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    return routing_weights, selected_experts


class IquestMoeMLP(nn.Module):
    """MLP for dense layers and optional shared expert (with optional gate)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        expert_gate: nn.Linear | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()
        self.expert_gate = expert_gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)
        if self.expert_gate is not None:
            out = F.sigmoid(self.expert_gate(x)[0]) * out
        return out


class IquestMoeSparseMoeBlock(nn.Module):
    """Sparse MoE with always-enabled shared expert (optional shared_expert_gate)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config
        # NOTE(yxing): setting for expert parallelism
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = config.num_experts
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
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )

        # Same as HF: always-enabled shared expert (no condition on size).
        shared_expert_intermediate_size = getattr(
            config, "shared_expert_intermediate_size", 0
        )
        if shared_expert_intermediate_size > 0:
            use_shared_expert_gate = getattr(config, "use_shared_expert_gate", False)
            if use_shared_expert_gate:
                self.shared_expert_gate = ReplicatedLinear(
                    config.hidden_size,
                    1,
                    bias=False,
                    quant_config=None,
                    prefix=f"{prefix}.shared_expert_gate",
                )
            else:
                self.shared_expert_gate = None
            self.shared_expert = IquestMoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert = None

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            custom_routing_function=_iquest_moe_topk_then_softmax_routing,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() <= 2, (
            "IquestMoeSparseMoeBlock only supports 1D or 2D inputs"
        )
        is_input_1d = hidden_states.dim() == 1
        _, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, _ = self.gate(hidden_states)
        shared_out, fused_out = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        final_hidden_states = (
            shared_out + fused_out if shared_out is not None else fused_out
        )

        if self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )

        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


class LoopGateProjection(nn.Module):
    """Gate projection for mixed attention in Loop 2+.

    Computes: g = sigmoid(linear(Q)) for each head independently.
    This gate determines how much to use Loop1's KV (global) vs current
    loop's KV (local).

    Supports tensor parallelism: each GPU handles a subset of heads.
    The weight matrix has shape [num_heads, head_dim] and is split along
    the head dimension.
    """

    def __init__(
        self,
        total_num_heads: int,
        head_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_hidden_states=False,
        rms_norm_eps: float = 1e-6,
        hidden_size: int = 0,
    ):
        super().__init__()
        self.total_num_heads = total_num_heads
        self.head_dim = head_dim
        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.use_hidden_states = use_hidden_states

        if self.use_hidden_states:
            # Hidden-states mode: gate = sigmoid(linear(norm(hidden_states)))
            # Matches training: plt_attention.py LoopGateProjection with
            # plt_gate_use_hidden_states=True
            self.gate_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.gate_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=True,
                gather_output=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_proj",
            )
        else:
            self.gate_proj = ColumnParallelLinear(
                head_dim,
                self.total_num_heads,
                bias=True,
                gather_output=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_proj",
            )

    def forward(
        self,
        query: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute gate values.

        Args:
            query: [num_heads, num_tokens, head_dim] — used for computation
                in query-based mode, used only for shape in hidden_states mode
            hidden_states: [num_tokens, hidden_size] — pre-layernorm hidden
                states, required when use_hidden_states=True

        Returns:
            gate: [num_tokens, num_heads * head_dim]
        """
        num_heads, num_tokens, head_dim = query.shape

        if self.use_hidden_states:
            # Training equivalent (plt_attention.py:113-124):
            #   x = gate_norm(hidden_states)
            #   gate_logits = matmul(x, weight.t()) + bias
            #   gate = sigmoid(gate_logits)
            assert hidden_states is not None, (
                "hidden_states required when use_hidden_states=True"
            )
            x = self.gate_norm(hidden_states)  # [num_tokens, hidden_size]
            gate_logits, _ = self.gate_proj(x)  # [num_tokens, num_heads]
            gate = torch.sigmoid(gate_logits)  # [num_tokens, num_heads]
            # Expand to match attention output shape
            gate = gate.unsqueeze(-1)  # [num_tokens, num_heads, 1]
            gate = gate.expand(-1, -1, head_dim)  # [num_tokens, num_heads, head_dim]
            gate = gate.reshape(
                num_tokens, num_heads * head_dim
            )  # [num_tokens, num_heads * head_dim]
            return gate

        # Query-based mode: per-head dot product via ColumnParallelLinear
        # + diagonal extraction (mathematically equivalent to training's
        # einsum('...hd,hd->...h', query, weight))
        assert num_heads == self.num_heads, (
            f"Expected {self.num_heads} heads, got {num_heads}"
        )

        query_flat = query.reshape(-1, head_dim)

        gate_logits_flat, _ = self.gate_proj(query_flat)

        gate_logits = gate_logits_flat.reshape(
            num_heads, num_tokens, self.num_heads
        )  # [num_heads, num_tokens, num_heads]

        # Extract diagonal: each head h's query should use output column h
        # gate_logits[h, :, h] gives the output for head h at each token
        gate_logits = torch.diagonal(
            gate_logits, dim1=0, dim2=2
        )  # [num_tokens, num_heads]
        gate_logits = gate_logits.transpose(0, 1)  # [num_heads, num_tokens]
        gate_logits = gate_logits.unsqueeze(-1)  # [num_heads, num_tokens, 1]

        # Apply sigmoid
        gate = torch.sigmoid(gate_logits)  # [num_heads, num_tokens, 1]

        # Expand and reshape to match q shape: [num_tokens, num_heads * head_dim]
        gate = gate.transpose(0, 1)  # [num_tokens, num_heads, 1]
        gate = gate.expand(-1, -1, head_dim)  # [num_tokens, num_heads, head_dim]
        gate = gate.reshape(
            num_tokens, num_heads * head_dim
        )  # [num_tokens, num_heads * head_dim]

        return gate


class IquestMoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        dual_chunk_attention_config: dict[str, Any] | None = None,
        plt_loop_nums: int = 0,
        plt_num_hidden_layers: int = 0,
        plt_sliding_window_size: int = -1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        self.plt_loop_nums = plt_loop_nums
        assert self.plt_loop_nums > 1, (
            f"Expect plt_loop_nums > 1, while now is {self.plt_loop_nums}"
        )
        self.attn = nn.ModuleList()
        base_cache_config = cache_config

        for loop_num_idx in range(self.plt_loop_nums):
            base_layer_idx = extract_layer_index(prefix)
            unique_layer_idx = loop_num_idx * plt_num_hidden_layers + base_layer_idx

            unique_prefix = prefix.replace(
                f"layers.{base_layer_idx}", f"layers.{unique_layer_idx}"
            )

            if loop_num_idx == 0:
                loop_cache_config = base_cache_config
            else:
                if base_cache_config is not None:
                    loop_cache_config = replace(
                        base_cache_config,
                        sliding_window=plt_sliding_window_size,
                    )
                else:
                    loop_cache_config = CacheConfig(
                        sliding_window=plt_sliding_window_size,
                        cache_dtype="auto",
                    )

            self.attn.append(
                Attention(
                    self.num_heads,
                    self.head_dim,
                    self.scaling,
                    num_kv_heads=self.num_kv_heads,
                    cache_config=loop_cache_config,
                    quant_config=quant_config,
                    prefix=f"{unique_prefix}.attn",
                    **{
                        "layer_idx": extract_layer_index(prefix),
                        "dual_chunk_attention_config": dual_chunk_attention_config,
                    }
                    if dual_chunk_attention_config
                    else {},
                )
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        loop_idx: int,
        gate_proj: LoopGateProjection | None = None,
        gate_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if loop_idx == 0:
            attn = self.attn[0]

            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
            )
            q_by_head = self.q_norm(q_by_head)
            q = q_by_head.view(q.shape)
            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
            )
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)
            q, k = self.rotary_emb(positions, q, k)
            attn_output = attn(q, k, v)
            output, _ = self.o_proj(attn_output)
            return output
        else:
            global_attn = self.attn[0]
            local_attn = self.attn[loop_idx]
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
            )
            q_by_head = self.q_norm(q_by_head)
            q = q_by_head.view(q.shape)
            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
            )
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)
            q, k = self.rotary_emb(positions, q, k)

            num_tokens, _ = q.shape
            q_reshaped = q.view(num_tokens, self.num_heads, self.head_dim).transpose(
                0, 1
            )
            # NOTE(yxing): non-first loop
            global_attn_out = global_attn(q, None, None)
            local_attn_out = local_attn(q, k, v)
            assert gate_proj is not None, "gate_proj must be provided for loop_idx > 0"
            gate = gate_proj(q_reshaped, hidden_states=gate_hidden_states)
            output = global_attn_out * gate + local_attn_out * (1 - gate)
            output, _ = self.o_proj(output)

            return output


class IquestMoeDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is None:
            rope_scaling = getattr(config, "rope_scaling", None)
            if isinstance(rope_scaling, dict):
                rope_parameters = dict(rope_scaling)
                if "type" in rope_parameters and "rope_type" not in rope_parameters:
                    rope_parameters["rope_type"] = rope_parameters["type"]
            else:
                rope_parameters = {"rope_type": "default"}
            if getattr(config, "rope_theta", None) is not None:
                rope_parameters["rope_theta"] = config.rope_theta

        plt_loop_nums = vllm_config.model_config.get_plt_num_loops()
        plt_num_layers = config.num_hidden_layers
        plt_sliding_window_size = vllm_config.model_config.get_plt_window_size()
        print(f"yxing plt_window_size: {plt_sliding_window_size=}")
        self.self_attn = IquestMoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=rope_parameters,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=dual_chunk_attention_config,
            plt_loop_nums=plt_loop_nums,
            plt_num_hidden_layers=plt_num_layers,
            plt_sliding_window_size=plt_sliding_window_size,
        )

        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = getattr(config, "mlp_only_layers", []) or []
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = IquestMoeSparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = IquestMoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        loop_idx: int,
        gate_proj: LoopGateProjection | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # residual = pre-layernorm hidden_states, corresponds to training's
        # gate_hidden_states saved in PLTLayer._preprocess before input_layernorm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            loop_idx=loop_idx,
            gate_proj=gate_proj,
            gate_hidden_states=residual,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states


@support_torch_compile
class IquestMoeModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = IquestMoeDecoderLayer,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.config = config
        self.quant_config = quant_config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda *, prefix: decoder_layer_type(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        # NOTE(yxing): plt-related configuration
        model_config = vllm_config.model_config
        self.plt_loop_nums = model_config.get_plt_num_loops()
        self.plt_emb_scale = model_config.get_plt_emb_scale()
        self.plt_hidden_scale = model_config.get_plt_hidden_scale()
        self.plt_normalize_per_loop = model_config.get_plt_normalize_per_loop()
        self.plt_gate_use_hidden_states = model_config.get_plt_gate_use_hidden_states()

        # Gate projections for Loop 2+ (one per layer)
        head_dim = config.hidden_size // config.num_attention_heads
        _, _, self.gate_projections = make_layers(
            config.num_hidden_layers,
            lambda prefix: LoopGateProjection(
                total_num_heads=config.num_attention_heads,
                head_dim=head_dim,
                quant_config=quant_config,
                prefix=prefix,
                rms_norm_eps=config.rms_norm_eps,
                use_hidden_states=self.plt_gate_use_hidden_states,
                hidden_size=config.hidden_size,
            ),
            prefix=f"{prefix}.gate_projections",
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        loop_num_idx: int = 0,
        loop_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        # NOTE(yxing): plt-related preprocessing
        if loop_hidden_states is not None:
            hidden_states = (
                self.plt_emb_scale * hidden_states
                + self.plt_hidden_scale * loop_hidden_states
            )

        for i, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            layer_idx = self.start_layer + i
            # Get gate_proj for this layer (only for loop_idx > 0)
            gate_proj = self.gate_projections[layer_idx] if loop_num_idx > 0 else None
            hidden_states = layer(positions, hidden_states, loop_num_idx, gate_proj)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        # NOTE(yxing):
        if loop_num_idx < self.plt_loop_nums - 1 and self.plt_normalize_per_loop:
            hidden_states = self.norm(hidden_states)

        if loop_num_idx == self.plt_loop_nums - 1:
            hidden_states = self.norm(hidden_states)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                assert loaded_weight.numel() == 1, (
                    f"KV scale numel {loaded_weight.numel()} != 1"
                )
                loaded_weight = loaded_weight.squeeze()
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if (
                        name_mapped.endswith(ignore_suffixes)
                        and name_mapped not in params_dict
                    ):
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        continue

                    if "plt_gate" in name:
                        parts = name.split(".")
                        layer_idx = parts[1]
                        # Handle gate_norm sub-module:
                        # layers.{idx}.self_attn.loop_gate_proj.gate_norm.weight
                        if "gate_norm" in name:
                            subpath = ".".join(parts[4:])  # gate_norm.weight
                            vllm_name = f"gate_projections.{layer_idx}.{subpath}"
                        else:
                            subname = parts[4]  # weight or bias
                            vllm_name = (
                                f"gate_projections.{layer_idx}.gate_proj.{subname}"
                            )
                        if vllm_name not in params_dict:
                            continue
                        param = params_dict[vllm_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param=param, loaded_weight=loaded_weight)
                        loaded_params.add(vllm_name)
                        continue

                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale"
                        )
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv scale in the checkpoint (e.g. %s), "
                                "but not found the expected name in the model "
                                "(e.g. %s). kv-scale is not loaded.",
                                name,
                                remapped_kv_scale_name,
                            )
                            continue
                        name = remapped_kv_scale_name
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class IquestMoeV12ForCausalLM(nn.Module, SupportsPP, SupportsLoRA, MixtureOfExperts):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = IquestMoeModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.expert_weights = []
        self.moe_layers = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, IquestMoeDecoderLayer)
            if isinstance(layer.mlp, IquestMoeSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)
        if example_layer is None:
            raise RuntimeError("No IquestMoe MoE layer found in the model.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = (
            1 if getattr(config, "shared_expert_intermediate_size", 0) > 0 else 0
        )
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts

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
            if isinstance(layer.mlp, IquestMoeSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        loop_num_idx: int = 0,
        loop_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            loop_num_idx,
            loop_hidden_states,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
