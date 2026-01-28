# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Jurassic model."""
from collections.abc import Iterable
from typing import Any, Optional, Union

import torch
from torch import nn

import vllm.envs as envs
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.distributed import (get_dp_group,
                              get_ep_group, get_pp_group,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              get_tp_group)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SwigluStepAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.fused_moe.shared_fused_moe import (
    SharedFusedMoE)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import MixtureOfExperts, SupportsPP
from .utils import (PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

logger = init_logger(__name__)

def sigmoid_routing_function(hidden_states: torch.Tensor,
                             gating_output: torch.Tensor, topk: int,
                             renormalize: bool):
    gating_output = gating_output.float()
    gate_prob = torch.sigmoid(gating_output)
    gate_prob = gate_prob / gate_prob.sum(dim=-1, keepdim=True)
    topk_prob, indices = torch.topk(gate_prob, k=topk, dim=1)
    expert_topk_weight = topk_prob
    if renormalize:
        expert_topk_weight = expert_topk_weight / torch.sum(
            expert_topk_weight, dim=-1, keepdim=True)
    return expert_topk_weight, indices.to(torch.int32) 

class Step3p5MLP(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj")

        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()
        self.prefix = prefix
        self.hidden_size = hidden_size
        self.limit = None
        layer_idx = int(prefix.split("layers.")[1].split(".")[0])
        if config.swiglu_limits_shared and config.swiglu_limits_shared[
                layer_idx] is not None and config.swiglu_limits_shared[
                    layer_idx] != 0:
            self.limit = config.swiglu_limits_shared[layer_idx]
            self.act_fn = SwigluStepAndMul(limit=self.limit)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        intermediate_act = self.act_fn.forward_cuda(gate_up)
        output, _ = self.down_proj(intermediate_act)
        return output

class Step3p5Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: Optional[Union[float, list[float]]] = 10000,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        rope_scaling: Optional[tuple] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        # Step3p5 specific args
        sliding_window: Optional[int] = None,
        use_head_wise_attn_gate: bool = False,
        layer_types: list = None,
        use_rope_layers: list = None,
        yarn_only_types: list = None,
        swa_num_attention_heads: Optional[int] = None,
        partial_rotary_factor: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        self.layer_idx = extract_layer_index(prefix)
        if layer_types:
            enable_sliding_window = layer_types[
                self.layer_idx] == "sliding_attention"
        else:
            enable_sliding_window = self.layer_idx % 2 == 0
        if yarn_only_types and layer_types[
                self.layer_idx] not in yarn_only_types:
            rope_scaling = None

        if sliding_window is not None and enable_sliding_window:
            sliding_window = (sliding_window)
            if swa_num_attention_heads is not None:
                num_heads = swa_num_attention_heads
                self.total_num_heads = swa_num_attention_heads
        else:
            sliding_window = None

        if isinstance(rope_theta, list):
            rope_theta = rope_theta[self.layer_idx]

        self.rank = get_tensor_model_parallel_rank()
        self.partial_rotary_factor = partial_rotary_factor
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
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

        rope_parameters: dict[str, Any] = {
            "rope_type": "default",
            "partial_rotary_factor": partial_rotary_factor,
        }
        if rope_scaling is not None:
            if not isinstance(rope_scaling, dict):
                raise ValueError(
                    "rope_scaling must be a dict for Step3p5Attention."
                )
            rope_parameters.update(rope_scaling)
        rope_parameters["rope_theta"] = self.rope_theta
        rope_parameters["partial_rotary_factor"] = partial_rotary_factor

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, rms_norm_eps)
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        if use_head_wise_attn_gate:
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=False,
                prefix=f"{prefix}.g_proj",
            )

        self.use_rope = True
        if use_rope_layers:
            self.use_rope = use_rope_layers[self.layer_idx]

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
        )

        self.max_position_embeddings = max_position
        assert self.partial_rotary_factor == 1 or self.partial_rotary_factor == 0.5
        self.rotary_dim = self.head_dim if self.partial_rotary_factor == 1 else self.head_dim // 2

    def qk_norm_rope(self, q, k, positions):
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        q_by_head = self.q_norm(q_by_head.contiguous())
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                           self.head_dim)
        k_by_head = self.k_norm(k_by_head.contiguous())
        k = k_by_head.view(k.shape)
        if self.use_rope:
            q, k = self.rotary_emb(positions, q, k)
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
                            dim=-1)
        q, k = self.qk_norm_rope(q, k, positions)
        attn_output = self.attn(q, k, v)
        if self.use_head_wise_attn_gate:
            extra_dims, _ = self.g_proj(hidden_states)
            output = attn_output.view(
                *attn_output.shape[:-1], self.num_heads,
                self.head_dim) * extra_dims.unsqueeze(-1).sigmoid()
            attn_output = output.view(*attn_output.shape)
        output, _ = self.o_proj(attn_output)
        return output

class FusedMoEBlock(nn.Module):

    def __init__(self,
                 config: ModelConfig,
                 parallel_config: ParallelConfig,
                 shared_experts: torch.nn.Module,
                 quant_config: Optional[QuantizationConfig] = None,
                 reduce_results: bool = True,
                 prefix: str = ""):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_idx = extract_layer_index(prefix)

        self.ep_size = get_ep_group().device_group.size()
        self.ep_rank = get_ep_group().device_group.rank()

        self.enable_eplb = parallel_config.enable_eplb
        self.n_routed_experts = config.moe_num_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = (self.ep_rank *
                                      self.n_local_physical_experts)
        self.physical_expert_end = (self.physical_expert_start +
                                    self.n_local_physical_experts)

        if self.tp_size > config.moe_num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.moe_num_experts}.")

        assert config.moe_dynamic_exp_p == 1, "Only support dynamic exp p=1"

        self.use_moe_router_bias = config.use_moe_router_bias
        if self.use_moe_router_bias:
            self.router_bias = nn.Parameter(torch.zeros(config.moe_num_experts,
                                                        dtype=torch.float32),
                                            requires_grad=False)
            custom_routing_function = self.router_bias_func
        elif config.moe_router_activation == "sigmoid":
            custom_routing_function = sigmoid_routing_function
        else:
            custom_routing_function = None
        self.need_fp32_gate = config.need_fp32_gate
        layer_idx = int(prefix.split("layers.")[1].split(".")[0])
        activation = "silu"
        swigluoai_step_limit = None
        if config.swiglu_limits and config.swiglu_limits[
                layer_idx] is not None and config.swiglu_limits[layer_idx] != 0:
            swigluoai_step_limit = config.swiglu_limits[layer_idx]
            activation = "swiglustep"
            logger.info(
                f"step3p5 layer_idx: {layer_idx}, activation limit: {config.swiglu_limits[layer_idx]}, will use swiglustep"
            )
        self.experts = SharedFusedMoE(
            shared_experts=shared_experts,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=reduce_results,
            renormalize=config.norm_expert_weight,
            quant_config=quant_config,
            activation=activation,
            activation_limit=swigluoai_step_limit if swigluoai_step_limit else None,
            prefix=f"{prefix}.experts",
            custom_routing_function=custom_routing_function,
            routed_scaling_factor=config.moe_router_scaling_factor,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
        )
        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.moe_num_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")

    def router_bias_func(self, hidden_states: torch.Tensor,
                         gating_output: torch.Tensor, topk: int,
                         renormalize: bool):
        gate_prob = torch.sigmoid(gating_output.float())
        gate_prob_with_bias = gate_prob + self.router_bias.unsqueeze(0)
        _, indices = torch.topk(gate_prob_with_bias, k=topk, dim=1)
        topk_prob = torch.gather(gate_prob, 1, indices)
        expert_topk_weight = topk_prob
        if renormalize:
            expert_topk_weight = expert_topk_weight / (
                torch.sum(expert_topk_weight, dim=-1, keepdim=True) + 1e-20)
        return expert_topk_weight, indices.to(torch.int32)

    def forward(
            self,
            hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.need_fp32_gate:
            router_logits = hidden_states.to(
                torch.float32) @ self.gate.weight.to(torch.float32).t()
        else:
            router_logits, _ = self.gate(hidden_states)
        shared_out, final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits)

        return shared_out, final_hidden_states.view(orig_shape)


class Step3p5DecoderLayer(nn.Module):

    def __init__(self,
                 config: ModelConfig,
                 parallel_config: ParallelConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        config = config.hf_config
        self.hidden_size = config.hidden_size
        rope_scaling = getattr(config, "rope_scaling", None)
        layer_idx = int(prefix.split("layers.")[1].split(".")[0])
        self.layer_idx = layer_idx
        if cache_config is not None:
            cache_config.sliding_window = None
        if config.att_impl_type == "GQA":
            num_attention_heads = None
            num_attention_groups = None
            head_dim = None
            if getattr(config, "attention_other_setting", None) and getattr(
                    config, "layer_types", []) and config.layer_types[
                        layer_idx] == config.attention_other_setting[
                            'attention_type']:
                num_attention_heads = config.attention_other_setting[
                    'num_attention_heads']
                num_attention_groups = config.attention_other_setting[
                    'num_attention_groups']
                head_dim = config.attention_other_setting['head_dim']
            partial_rotary_factors = getattr(config, "partial_rotary_factors",
                                             [])
            self.self_attn = Step3p5Attention(
                hidden_size=self.hidden_size,
                num_heads=num_attention_heads
                if num_attention_heads else config.num_attention_heads,
                max_position=config.max_position_embeddings,
                num_kv_heads=num_attention_groups
                if num_attention_groups else config.num_attention_groups,
                rope_theta=config.rope_theta,
                rms_norm_eps=config.rms_norm_eps,
                qkv_bias=getattr(config, 'attention_bias', False),
                head_dim=head_dim if head_dim else getattr(
                    config, 'head_dim', None),
                cache_config=cache_config,
                quant_config=quant_config,
                rope_scaling=rope_scaling,
                sliding_window=getattr(config, 'sliding_window', None),
                use_head_wise_attn_gate=getattr(config,
                                                "use_head_wise_attn_gate",
                                                False),
                layer_types=getattr(config, "layer_types", []),
                use_rope_layers=getattr(config, "use_rope_layers", []),
                yarn_only_types=getattr(config, "yarn_only_types", []),
                partial_rotary_factor=partial_rotary_factors[layer_idx]
                if partial_rotary_factors else 1.0,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(
                f"Unsupported attention implementation: {config.att_impl_type}"
            )
        self.use_moe = False
        self.tp_group = get_tp_group()
        self.use_fused_all_reduce = get_tensor_model_parallel_world_size(
        ) > 1 and get_dp_group().world_size == 1 and envs.VLLM_USE_FUSED_ALL_REDUCE
        if self.use_fused_all_reduce:
            logger.warning_once("Enable custom fused all reduce...")
        else:
            logger.warning_once("Disable custom fused all reduce...")

        moe_layers_enum = getattr(config, "moe_layers_enum", None)
        if moe_layers_enum is not None:
            moe_layers_idx = [
                int(i) for i in moe_layers_enum.strip().split(',')
            ]
        else:
            moe_layers_idx = [i for i in range(1, config.num_hidden_layers)]
        if layer_idx in moe_layers_idx:
            reduce_results = True
            if self.use_fused_all_reduce or self.tp_group.world_size == 1 and get_ep_group(
            ).world_size == 1:
                reduce_results = False
            self.share_expert = Step3p5MLP(
                config=config,
                hidden_size=self.hidden_size,
                intermediate_size=config.share_expert_dim,
                hidden_act="silu",
                reduce_results=reduce_results,
                quant_config=quant_config,
                prefix=f"{prefix}.share_expert")
            self.moe = FusedMoEBlock(shared_experts=self.share_expert,
                                     config=config,
                                     parallel_config=parallel_config,
                                     quant_config=quant_config,
                                     reduce_results=reduce_results,
                                     prefix=f"{prefix}.moe")
            self.use_moe = True
        else:
            self.mlp = Step3p5MLP(config=config,
                                  hidden_size=config.hidden_size,
                                  intermediate_size=config.intermediate_size,
                                  hidden_act="silu",
                                  quant_config=quant_config,
                                  reduce_results=True,
                                  prefix=f"{prefix}.mlp")
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.prefix = prefix

    def add_and_maybe_inplace_all_reduce(self, in1: torch.Tensor,
                                         in2: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_all_reduce:
            return in1 + in2
        return self.tp_group.all_reduce(in1 + in2)

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.use_moe:
            shared_output, moe_output = self.moe(hidden_states)
            # share expert & moe 可以合并all reduce
            ffn_output = self.add_and_maybe_inplace_all_reduce(
                moe_output, shared_output)
        else:
            ffn_output = self.mlp(hidden_states)
        hidden_states = ffn_output + residual
        return hidden_states

@support_torch_compile
class Step3p5Model(nn.Module):

    def __init__(self,
                 vllm_config: VllmConfig,
                 prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.vocab_size = config.vocab_size
        self.config = config

        self.moe_num_experts = config.moe_num_experts

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Step3p5DecoderLayer(
                config=vllm_config.model_config,
                parallel_config=vllm_config.parallel_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.hidden_size))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
            })

        return hidden_states


class Step3p5ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.vllm_config = vllm_config

        self.model = Step3p5Model(vllm_config=vllm_config,
                                  prefix=maybe_prefix(prefix, "model"))

        self.moe_layers: list[FusedMoEBlock] = []
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, Step3p5DecoderLayer)
            if hasattr(layer, "moe") and isinstance(layer.moe, FusedMoEBlock):
                self.moe_layers.append(layer.moe)

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE
                if not lora_config else lora_config.lora_vocab_padding_size,
            )
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        # Set MoE hyperparameters
        self.expert_weights = []
        assert len(self.moe_layers) > 0, "No MoE layers found in the model."
        example_layer = self.moe_layers[0]
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None):
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model.norm(hidden_states)
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        for layer_idx, layer in enumerate(self.moe_layers):
            experts = layer.experts
            assert isinstance(experts, FusedMoE)
            # Register the expert weights.
            self.expert_weights.append(experts.get_expert_weights())
            experts.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = (num_physical_experts -
                                      self.num_logical_experts)
        for layer in self.moe_layers:
            assert isinstance(layer, FusedMoEBlock)
            layer.n_local_physical_experts = num_local_physical_experts
            layer.n_physical_experts = num_physical_experts
            layer.n_redundant_experts = self.num_redundant_experts
            layer.experts.update_expert_map()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        vllm_config = self.vllm_config
        config = vllm_config.model_config.hf_config
        assert config.num_attention_groups > 1, "Only support GQA"
        qkv_params_mapping = []
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params = set()

        expert_params_mapping = [
            (".moe.experts.w13_weight", ".moe.gate_proj.weight", "w1"),
            (".moe.experts.w13_weight", ".moe.up_proj.weight", "w3"),
            (".moe.experts.w2_weight", ".moe.down_proj.weight", "w2")
        ]

        disable_moe_stacked_params = [data[1] for data in expert_params_mapping]

        for name, loaded_weight in weights:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if any(disable_moe_stacked_param in name
                       for disable_moe_stacked_param in
                       disable_moe_stacked_params):
                    continue
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    moe_expert_num = self.model.moe_num_experts
                    assert loaded_weight.shape[0] == moe_expert_num
                    for expert_id in range(moe_expert_num):
                        loaded_weight_expert = loaded_weight[expert_id]
                        weight_loader(param,
                                      loaded_weight_expert,
                                      name,
                                      shard_id=shard_id,
                                      expert_id=expert_id)
                    loaded_params.add(name)
                    break
                else:
                    for (param_name, weight_name, start_idx,
                         end_idx) in qkv_params_mapping:
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]
                        dim = param.shape[param.output_dim]
                        begin_idx = int(start_idx * dim)
                        end_idx = int(end_idx * dim)
                        param_slice = param.narrow(param.output_dim, begin_idx,
                                                   end_idx - begin_idx)
                        param_slice.copy_(loaded_weight)
                        loaded_params.add(name)
                        break
                    else:
                        if is_pp_missing_parameter(name, self):
                            continue
                        if "expert_bias" in name:
                            logger.warning_once("ignore expert_bias")
                            continue
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)
        return loaded_params


def get_spec_layer_idx_from_weight_name(config: ModelConfig,
                                        weight_name: str) -> Optional[int]:
    if hasattr(config,
               "num_nextn_predict_layers") and (config.num_nextn_predict_layers
                                                > 0):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx+i}."):
                return layer_idx + i
    return None
