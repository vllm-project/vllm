# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.linear.bailing_linear_attn import (
    BailingMoELinearAttention,
    _build_rope_parameters,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.bailing_moe import BailingMLP
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionMetadata

from .interfaces import HasInnerState, IsHybrid, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def is_linear_layer(layer_idx, layer_group_size):
    if layer_idx is None:
        return False
    if layer_group_size > 0:
        return (layer_idx + 1) % layer_group_size != 0
    else:
        return False


class BailingMoeV25MLAAttention(nn.Module):
    """
    MLA Attention for BailingMoeV2.5 full attention layers.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        layer_id: int = 0,
        prefix: str = "attention",
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.layer_id = layer_id
        self.prefix = prefix

        # MLA dimensions
        self.qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 128)
        self.qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = getattr(config, "v_head_dim", 128)

        # LoRA ranks
        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        self.kv_lora_rank = getattr(config, "kv_lora_rank", 512)

        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        self.num_local_heads = self.num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5

        # KV projections
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            eps=config.rms_norm_eps,
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )

        # Output projection
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if self.q_lora_rank is not None:
            # Use fused_qkv_a_proj when q_lora_rank is set
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
            self.q_a_layernorm = RMSNorm(
                self.q_lora_rank,
                eps=config.rms_norm_eps,
            )
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
            self.q_proj = None
            self.kv_a_proj_with_mqa = None
        else:
            # Direct projections when no q_lora_rank
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
            self.fused_qkv_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None

        rope_parameters = _build_rope_parameters(config) or {}
        # MLA rotates the full qk_rope_head_dim,
        # partial_rotary_factor is for the linear-attn head only.
        rope_parameters = {
            k: v for k, v in rope_parameters.items() if k != "partial_rotary_factor"
        }
        rope_parameters["rope_dim"] = self.qk_rope_head_dim
        max_position = getattr(config, "max_position_embeddings", 8192)
        self.rotary_emb = get_rope(
            head_size=self.qk_rope_head_dim,
            max_position=max_position,
            is_neox_style=False,
            rope_parameters=rope_parameters,
        )

        # Build MLAModules for MultiHeadLatentAttentionWrapper
        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=self.q_proj,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for MLA attention."""
        return self.mla_attn(positions, hidden_states)


class BailingMoEGate(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.weight = nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                dtype=self.params_dtype,
            ),
        )
        if getattr(config, "moe_router_enable_expert_bias", False):
            self.expert_bias = nn.Parameter(
                torch.empty((config.num_experts,), dtype=torch.float32),
            )
        else:
            self.expert_bias = None

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.to(self.weight.dtype), self.weight, None).to(
            hidden_states.dtype
        )
        return logits


class BailingMoeV25(nn.Module):
    """Bailing MoE v2.5 - standalone implementation for linear attention model."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        layer_id: int = 0,
        prefix: str = "",
    ):
        super().__init__()

        self.layer_id = layer_id
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        norm_topk_prob = getattr(config, "norm_topk_prob", None)
        # Ring-2.5 reference implementations normalize routing weights by default.
        self.norm_expert_prob = True if norm_topk_prob is None else bool(norm_topk_prob)
        self.hidden_size = config.hidden_size
        self.quant_config = quant_config
        self.num_shared_experts = config.num_shared_experts
        self.score_function = getattr(config, "score_function", None)
        self.n_group = getattr(config, "n_group", None)
        self.topk_group = getattr(config, "topk_group", None)
        self.use_grouped_topk = self.n_group is not None and self.topk_group is not None
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        router_dtype = getattr(config, "router_dtype", None)
        if router_dtype is None or router_dtype == "fp32":
            self.router_dtype = torch.float32
        else:
            self.router_dtype = torch.bfloat16

        # Gate for routing
        self.gate = BailingMoEGate(
            config=config,
            params_dtype=self.router_dtype,
            prefix=f"{prefix}.gate",
        )
        correction_bias = (
            self.gate.expert_bias if self.gate.expert_bias is not None else None
        )
        if self.score_function is not None:
            assert (self.score_function == "softmax" and correction_bias is None) or (
                self.score_function == "sigmoid" and correction_bias is not None
            ), (
                "score_function and correction_bias should be "
                "(softmax, None) or (sigmoid, not None)"
            )

        # Shared experts (using BailingMLP)
        if self.num_shared_experts > 0:
            if hasattr(config, "moe_shared_expert_intermediate_size"):
                intermediate_size = config.moe_shared_expert_intermediate_size
            else:
                intermediate_size = config.moe_intermediate_size
            intermediate_size *= config.num_shared_experts
            self.shared_experts = BailingMLP(
                intermediate_size=intermediate_size,
                config=config,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        # Routed experts using FusedMoE
        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=self.norm_expert_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.score_function,
            e_score_correction_bias=correction_bias,
            num_expert_group=self.n_group,
            topk_group=self.topk_group,
            use_grouped_topk=self.use_grouped_topk,
            router_logits_dtype=self.router_dtype,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        # Ensure contiguous token-major layout before router/projections.
        hidden_states = hidden_states.contiguous().view(-1, hidden_size)

        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states.to(self.router_dtype))
        router_logits = router_logits.to(hidden_states.dtype)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        return final_hidden_states.view(num_tokens, hidden_size)


class BailingMoeV25DecoderLayer(nn.Module):
    """Decoder layer supporting both linear and full attention."""

    def __init__(
        self,
        config: PretrainedConfig,
        vllm_config: VllmConfig,
        prefix: str = "layer",
        layer_id: int = 0,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size

        # Determine attention type (0 = linear, 1 = full)
        self.attention_type = getattr(config, "attention_type", 1)

        if self.attention_type == 0:  # Linear attention
            self.self_attn = BailingMoELinearAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:  # Full attention
            self.self_attn = BailingMoeV25MLAAttention(
                config,
                quant_config=vllm_config.quant_config,
                layer_id=layer_id,
                prefix=f"{prefix}.self_attn",
                cache_config=vllm_config.cache_config,
            )

        # MLP/MoE
        is_moe_layer = config.num_experts > 1 and layer_id >= getattr(
            config, "first_k_dense_replace", 0
        )

        if is_moe_layer:
            self.mlp = BailingMoeV25(
                config,
                quant_config=vllm_config.quant_config,
                layer_id=layer_id,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = BailingMLP(
                intermediate_size=config.intermediate_size,
                config=config,
                quant_config=vllm_config.quant_config,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
            )

        # Layer norms
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-5))
        self.input_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Input layernorm
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self attention
        if self.attention_type == 0:
            # Linear attention uses output tensor
            self_attention_output = torch.zeros_like(hidden_states)
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
        else:
            # Full attention
            self_attention_output = self.self_attn(hidden_states, positions)

        hidden_states, residual = self.post_attention_layernorm(
            self_attention_output, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class BailingMoeV25Model(nn.Module):
    """Bailing MoE v2.5 Model with hybrid attention support."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size

        # Determine layer types based on layer_group_size
        self.layer_group_size = getattr(config, "layer_group_size", 1)
        self.num_layers = config.num_hidden_layers

        # decoder_attention_types: 0 = linear, 1 = full
        self.decoder_attention_types = [
            0 if is_linear_layer(i, self.layer_group_size) else 1
            for i in range(self.num_layers)
        ]

        # Embeddings
        if get_pp_group().is_first_rank:
            self.word_embeddings = VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
                org_num_embeddings=self.vocab_size,
            )
        else:
            from vllm.model_executor.models.utils import PPMissingLayer

            self.word_embeddings = PPMissingLayer()

        # Layers
        def layer_fn(prefix):
            layer_idx = int(prefix.split(".")[-1])
            layer_config = copy.deepcopy(config)
            layer_config.attention_type = self.decoder_attention_types[layer_idx]

            return BailingMoeV25DecoderLayer(
                config=layer_config,
                vllm_config=vllm_config,
                prefix=prefix,
                layer_id=layer_idx,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            self.num_layers, layer_fn, prefix=f"{prefix}.layers"
        )

        # Final norm
        norm_kwargs = {}
        if hasattr(config, "rms_norm_eps"):
            norm_kwargs["eps"] = config.rms_norm_eps
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, **norm_kwargs)
        else:
            from vllm.model_executor.models.utils import PPMissingLayer

            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                hidden_states = self.word_embeddings(input_ids)
            else:
                hidden_states = inputs_embeds
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                positions=positions,
                attn_metadata=attn_metadata,
                residual=residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        else:
            if residual is not None:
                hidden_states, _ = self.norm(hidden_states, residual)
            else:
                hidden_states = self.norm(hidden_states)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Get expert parameter mapping for MoE layers."""
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
            num_redundant_experts=0,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load checkpoint weights with simplified mapping."""

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        # Stacked parameter mappings (fused projections)
        stacked_mappings = [
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        # Expert parameter mappings from FusedMoE
        expert_mappings = list(self.get_expert_mapping())

        def load_param(name: str, tensor: torch.Tensor, shard_id=None) -> bool:
            """Load a single parameter."""
            if name not in params_dict or is_pp_missing_parameter(name, self):
                return False
            if name.endswith(".bias") and name not in params_dict:
                return False

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)

            if shard_id is None:
                weight_loader(param, tensor)
            elif isinstance(shard_id, int):
                weight_loader(param, tensor, shard_id)
            else:
                # Expert param: (expert_id, shard_id)
                weight_loader(
                    param, tensor, name, expert_id=shard_id[0], shard_id=shard_id[1]
                )

            loaded_params.add(name)
            return True

        def normalize_name(name: str) -> str | None:
            """Normalize checkpoint name to model parameter name."""
            # Skip special weights
            if name.startswith("model.mtp"):
                return None
            # Remove 'model.' prefix if present
            # (e.g., 'model.layers.0...' -> 'layers.0...')
            name = name.removeprefix("model.")
            # Map attention.dense based on layer type
            if "attention.dense" in name:
                layer_idx = (
                    int(name.split("layers.")[1].split(".")[0])
                    if "layers." in name
                    else 0
                )
                attn_name = (
                    "self_attn.dense"
                    if is_linear_layer(layer_idx, self.config.layer_group_size)
                    else "self_attn.o_proj"
                )
                name = name.replace("attention.dense", attn_name)

            # Standard mappings
            name = name.replace("attention.", "self_attn.")
            name = name.replace(
                "mlp.gate.e_score_correction_bias", "mlp.gate.expert_bias"
            )

            return maybe_remap_kv_scale_name(name, params_dict)

        for orig_name, weight in weights:
            norm_name = normalize_name(orig_name)
            if norm_name is None:
                continue

            # Try stacked mappings
            loaded = False
            for param_suf, weight_suf, shard_id in stacked_mappings:
                if weight_suf not in norm_name:
                    continue
                mapped = norm_name.replace(weight_suf, param_suf).replace(
                    "attention.", "self_attn."
                )
                if load_param(mapped, weight, shard_id):
                    loaded = True
                    break
            if loaded:
                continue

            # Handle expert weights
            if "mlp.experts" in norm_name:
                # Expert bias
                if (
                    "mlp.experts.e_score_correction_bias" in norm_name
                    or "mlp.experts.expert_bias" in norm_name
                ):
                    alt = norm_name.replace(
                        "mlp.experts.e_score_correction_bias", "mlp.gate.expert_bias"
                    ).replace("mlp.experts.expert_bias", "mlp.gate.expert_bias")
                    if load_param(alt, weight) or load_param(norm_name, weight):
                        continue

                # Routed experts
                for param_name, weight_name, expert_id, shard_id in expert_mappings:
                    if weight_name not in norm_name:
                        continue
                    mapped = norm_name.replace(weight_name, param_name)
                    if load_param(mapped, weight, (expert_id, shard_id)):
                        break
                continue

            # General parameters
            load_param(norm_name, weight)

        return loaded_params


class BailingMoeV25ForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsPP):
    """Bailing MoE v2.5 For CausalLM."""

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.model = BailingMoeV25Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
            }
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], ...]:
        """Calculate shape for linear attention cache."""
        config = vllm_config.model_config.hf_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size

        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        # Return base state shape from linear attention (no padding)
        return MambaStateShapeCalculator.linear_attention_state_shape(
            num_heads=config.num_attention_heads,
            tp_size=tp_size,
            head_dim=head_dim,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.linear_attention_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple:
        return MambaStateCopyFuncCalculator.linear_attention_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
