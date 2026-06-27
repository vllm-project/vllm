# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Zaya model."""

import logging
from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.cca import CCA
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.zaya import ZayaConfig

from .interfaces import HasInnerState, IsHybrid
from .utils import make_empty_intermediate_tensors_factory, maybe_prefix

logger = logging.getLogger(__name__)


class _FP32EmbeddingMethod(UnquantizedEmbeddingMethod):
    """LM-head projection that returns fp32 logits via out_dtype."""

    def apply(self, layer, x, bias=None):
        if not torch.is_floating_point(x):
            return super().apply(layer, x, bias)
        out = torch.mm(x, layer.weight.t(), out_dtype=torch.float32)
        if bias is not None:
            out = out + bias.to(torch.float32)
        return out


class ZayaResidualScaling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_states_scale = nn.Parameter(torch.ones(hidden_size))
        self.hidden_states_bias = nn.Parameter(torch.zeros(hidden_size))
        self.residual_scale = nn.Parameter(torch.ones(hidden_size))
        self.residual_bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: torch.Tensor,
                residual: torch.Tensor) -> torch.Tensor:
        hidden_states = (
            hidden_states.float() + self.hidden_states_bias.to(torch.float32)
        ) * self.hidden_states_scale.to(torch.float32)
        residual = (
            residual.float() + self.residual_bias.to(torch.float32)
        ) * self.residual_scale.to(torch.float32)
        return hidden_states + residual


def _apply_norm_with_fp32_residual(
    norm: nn.Module,
    residual: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(norm, RMSNorm):
        if residual.dtype != norm.weight.dtype:
            hidden_states = norm.forward_native(residual)
        else:
            hidden_states = norm(residual)
        return hidden_states.to(target_dtype)
    return norm(residual.to(target_dtype))


def _rope_parameters_for_layer(config: ZayaConfig, layer_idx: int) -> dict:
    layer_types = getattr(config, "layer_types", None) or ["hybrid"]
    layer_type = layer_types[layer_idx]
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        params = rope_parameters.get(layer_type, rope_parameters)
        if isinstance(params, dict):
            params = dict(params)
        else:
            params = {}
    else:
        params = {}
    params.setdefault("rope_type", "default")
    params.setdefault("rope_theta", getattr(config, "rope_theta", 5000000))
    params.setdefault(
        "partial_rotary_factor", getattr(config, "partial_rotary_factor", 0.5))
    return params


class ZayaAttention(nn.Module):
    def __init__(
        self,
        config: ZayaConfig,
        layer_idx: int,
        prefix: str,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.qkv_proj = CCA(
            config=config,
            cca_num_k_heads=self.num_key_value_heads,
            cca_num_q_heads=self.num_attention_heads,
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            cca_time0=config.cca_time0,
            cca_time1=config.cca_time1,
            layer_number=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = ReplicatedLinear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.o_proj",
        )

        layer_type = config.layer_types[layer_idx]
        sliding_window = (
            config.sliding_window if layer_type == "hybrid_sliding" else None
        )
        self.attn = Attention(
            self.num_attention_heads,
            self.head_dim,
            self.scale,
            self.num_key_value_heads,
            per_layer_sliding_window=sliding_window,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            is_neox_style=True,
            rope_parameters=_rope_parameters_for_layer(config, layer_idx),
        )

        self.q_dim = self.num_attention_heads * self.head_dim
        self.k_dim = self.num_key_value_heads * self.head_dim
        self.v_dim = self.num_key_value_heads * self.head_dim
        self.qkv_dim = self.q_dim + self.k_dim + self.v_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        output_qkv = torch.zeros(
            (hidden_states.shape[0], self.qkv_dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        self.qkv_proj(hidden_states, output_qkv)
        q, k, v = output_qkv.split([self.q_dim, self.k_dim, self.v_dim], dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output)


class ZayaRouterMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.fc1 = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.fc2",
        )
        self.out_proj = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.out_proj",
        )
        self.act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = _apply_norm_with_fp32_residual(
            self.norm, hidden_states, self.norm.weight.dtype)
        hidden_states = self.act_fn(self.fc1(hidden_states))
        hidden_states = self.act_fn(self.fc2(hidden_states))
        return self.out_proj(hidden_states)


class ZayaRouter(nn.Module):
    def __init__(
        self,
        config: ZayaConfig,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts + 1
        self.topk = config.num_experts_per_tok
        self.router_hidden_size = config.router_hidden_size

        self.down_proj = ReplicatedLinear(
            self.hidden_size,
            self.router_hidden_size,
            bias=True,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.down_proj",
        )
        self.use_eda = layer_idx != 0
        if self.use_eda:
            self.router_states_scale = nn.Parameter(
                torch.ones(self.router_hidden_size))
        self.router_mlp = ZayaRouterMLP(
            self.router_hidden_size,
            self.num_experts,
            config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.router_mlp",
        )
        self.register_buffer(
            "balancing_biases", torch.zeros(self.num_experts, dtype=torch.float32))
        self.balancing_biases[-1] = -1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_router_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_length = hidden_states.shape[0]
        router_hidden_states = self.down_proj(hidden_states)
        if self.use_eda and prev_router_hidden_states is not None:
            router_hidden_states = (
                router_hidden_states
                + prev_router_hidden_states * self.router_states_scale)

        router_hidden_states_next = router_hidden_states[-seq_length:].clone()
        router_logits = self.router_mlp(router_hidden_states)
        router_probs = torch.softmax(router_logits, dim=-1)
        biased_router_probs = (
            router_probs.detach().to(torch.float32) + self.balancing_biases)
        _, router_indices = torch.topk(biased_router_probs, self.topk, dim=-1)
        router_probs = torch.gather(router_probs, dim=1, index=router_indices)

        skip_expert = router_indices == self.config.num_experts
        router_probs = router_probs.masked_fill(skip_expert, 0)
        router_indices = router_indices.masked_fill(skip_expert, 0)
        if router_probs.dtype != hidden_states.dtype:
            router_probs = router_probs.to(hidden_states.dtype)
        return router_probs, router_indices, router_hidden_states_next


class ZayaSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: ZayaConfig,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.topk = config.num_experts_per_tok
        self.gate = ZayaRouter(
            config,
            layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )

        def _custom_routing_fn(hidden_states, gating_output, topk, renormalize):
            topk_weights = gating_output[:, :topk]
            topk_ids = gating_output[:, topk : 2 * topk].to(torch.int64)
            return topk_weights, topk_ids

        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=self.topk,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=False,
            custom_routing_function=_custom_routing_fn,
            activation="silu",
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_router_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs, indices, router_hidden_states_out = self.gate(
            hidden_states,
            prev_router_hidden_states=prev_router_hidden_states,
        )
        packed_routing = torch.cat([probs, indices.to(probs.dtype)], dim=-1)
        hidden_states = self.experts(hidden_states, packed_routing)
        return hidden_states, router_hidden_states_out


class ZayaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ZayaConfig,
        layer_idx: int,
        prefix: str,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = ZayaAttention(
            config,
            layer_idx,
            f"{prefix}.self_attn",
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
        )
        self.post_attention_residual_scale = ZayaResidualScaling(
            config.hidden_size)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = ZayaSparseMoeBlock(
            config,
            layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.post_mlp_residual_scale = ZayaResidualScaling(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        prev_router_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        layer_input_dtype = self.input_layernorm.weight.dtype
        hidden_states = _apply_norm_with_fp32_residual(
            self.input_layernorm, residual, layer_input_dtype)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
        )

        residual = self.post_attention_residual_scale(hidden_states, residual)
        hidden_states = _apply_norm_with_fp32_residual(
            self.post_attention_layernorm,
            residual,
            self.post_attention_layernorm.weight.dtype,
        )
        hidden_states, prev_router_hidden_states = self.mlp(
            hidden_states, prev_router_hidden_states)
        hidden_states = self.post_mlp_residual_scale(hidden_states, residual)
        return hidden_states, prev_router_hidden_states


@support_torch_compile
class ZayaModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config: ZayaConfig = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        assert not lora_config

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.layers = nn.ModuleList([
            ZayaDecoderLayer(
                config,
                layer_idx,
                f"{prefix}.layers.{layer_idx}",
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
            ) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_hidden_states_scale = nn.Parameter(torch.ones(config.hidden_size))
        self.input_hidden_states_bias = nn.Parameter(torch.zeros(config.hidden_size))

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = (
            inputs_embeds.float() +
            self.input_hidden_states_bias.to(torch.float32)
        ) * self.input_hidden_states_scale.to(torch.float32)
        prev_router_hidden_states = None

        for decoder_layer in self.layers:
            hidden_states, prev_router_hidden_states = decoder_layer(
                hidden_states,
                positions,
                prev_router_hidden_states,
            )

        hidden_states = _apply_norm_with_fp32_residual(
            self.norm, hidden_states, self.norm.weight.dtype)
        return hidden_states


class ZayaForCausalLM(nn.Module, HasInnerState, IsHybrid):
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.cca_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[tuple[int, int], tuple[int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config

        return MambaStateShapeCalculator.cca_state_shape(
            tp_world_size=parallel_config.tensor_parallel_size,
            conv_kernel_size=(hf_config.cca_time0 - 1) + (hf_config.cca_time1 - 1),
            num_k_heads=hf_config.num_key_value_heads,
            num_q_heads=hf_config.num_attention_heads,
            head_dim=hf_config.head_dim,
            recurrent_state_size=(
                hf_config.num_key_value_heads * hf_config.head_dim // 2
            ),
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config
        assert config.num_experts_per_tok == 1, "Only topk=1 is supported in Zaya!"
        assert not cache_config.enable_prefix_caching, (
            "Zaya currently does not support prefix caching")

        tp_world_size = get_tensor_model_parallel_world_size()
        if tp_world_size > 1:
            logger.warning(
                "TP>1 detected; CCA currently replicates heads on every rank.")

        super().__init__()
        self.config = config
        self.lora_config = lora_config
        self.scheduler_config = scheduler_config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config
        self.model_config = vllm_config.model_config

        self.model = ZayaModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=(
                DEFAULT_VOCAB_PADDING_SIZE
                if not lora_config else lora_config.lora_vocab_padding_size),
            quant_config=None,
            bias=config.lm_head_bias,
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size, config.vocab_size)
        if bool(getattr(config, "zaya_high_prec", False)):
            self.lm_head.quant_method = _FP32EmbeddingMethod()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        for key, buffer in buffers_dict.items():
            if "cos_sin_cache" in key:
                continue
            params_dict[key] = buffer
        fused_moe_modules = {
            name: module
            for name, module in self.named_modules()
            if isinstance(module, FusedMoE)
        }

        loaded_params: set[str] = set()
        tp_rank = get_tensor_model_parallel_rank()
        disable_tqdm = tp_rank != 0

        import tqdm

        skipped_weights: list[str] = []
        for chkpt_weight_name, loaded_weight in tqdm.tqdm(
            weights,
            desc="Loading weights",
            unit_scale=True,
            unit="weights",
            disable=disable_tqdm,
        ):
            weight_name = chkpt_weight_name
            if weight_name.endswith(".self_attn.qk_norm.temp"):
                weight_name = weight_name.replace(
                    ".self_attn.qk_norm.temp",
                    ".self_attn.qkv_proj.temp",
                )

            if "lora" in weight_name:
                if "_A.weight" in weight_name:
                    weight_name = weight_name.replace("_A.weight", ".A.weight")
                elif "_B.weight" in weight_name:
                    weight_name = weight_name.replace("_B.weight", ".B.weight")

            if weight_name.endswith(".mlp.experts.gate_up_proj"):
                fused_moe_prefix = weight_name.removesuffix(".gate_up_proj")
                fused_moe_module = fused_moe_modules.get(fused_moe_prefix)
                if fused_moe_module is None:
                    skipped_weights.append(chkpt_weight_name)
                    continue

                param_name = f"{fused_moe_prefix}.w13_weight"
                param = params_dict[param_name]
                gate_weight, up_weight = loaded_weight.chunk(2, dim=1)
                for expert_id, (gate_expert, up_expert) in enumerate(
                        zip(gate_weight, up_weight)):
                    fused_moe_module.weight_loader(
                        param,
                        gate_expert,
                        param_name,
                        "w1",
                        expert_id,
                    )
                    fused_moe_module.weight_loader(
                        param,
                        up_expert,
                        param_name,
                        "w3",
                        expert_id,
                    )
                loaded_params.add(param_name)
                continue

            if weight_name.endswith(".mlp.experts.down_proj"):
                fused_moe_prefix = weight_name.removesuffix(".down_proj")
                fused_moe_module = fused_moe_modules.get(fused_moe_prefix)
                if fused_moe_module is None:
                    skipped_weights.append(chkpt_weight_name)
                    continue

                param_name = f"{fused_moe_prefix}.w2_weight"
                param = params_dict[param_name]
                for expert_id, down_expert in enumerate(loaded_weight):
                    fused_moe_module.weight_loader(
                        param,
                        down_expert,
                        param_name,
                        "w2",
                        expert_id,
                    )
                loaded_params.add(param_name)
                continue

            if weight_name not in params_dict:
                skipped_weights.append(weight_name)
                continue
            param = params_dict[weight_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(weight_name)
        if skipped_weights:
            raise RuntimeError(
                "Unexpected Zaya checkpoint weights were not loaded: "
                f"{sorted(skipped_weights)}")
        return loaded_params
