# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Jurassic model."""

from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
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
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.step3_vl import Step3TextConfig

from .interfaces import SupportsPP
from .utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class FusedMoEBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.moe_num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.moe_num_experts}."
            )

        self.experts = FusedMoE(
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_expert_weight,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.moe_num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, _ = self.gate(hidden_states)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(orig_shape)


class Step3TextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
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
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        intermediate_act = self.act_fn(gate_up)
        output, _ = self.down_proj(intermediate_act)
        return output


class Step3TextAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        norm_eps: float,
        rope_parameters: dict[str, Any],
        share_q_dim: int | None = None,
        max_position_embedding: int = 8192,
        head_dim: int = 256,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        if num_kv_heads != 1:
            raise ValueError(
                f"Step3TextAttention num_kv_heads must be 1, but got {num_kv_heads}."
            )
        self.num_kv_heads = num_kv_heads

        self.head_dim = head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.q_size = share_q_dim if share_q_dim else self.head_dim

        self.qkv_proj = ReplicatedLinear(
            hidden_size,
            self.q_size + self.kv_size * 2,
            bias=False,
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
        self.inter_norm = RMSNorm(self.q_size, eps=norm_eps)
        self.wq = ColumnParallelLinear(
            self.q_size,
            self.head_dim * self.total_num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embedding,
            rope_parameters=rope_parameters,
        )
        scaling = self.head_dim**-0.5
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scaling,
            self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.inter_norm(q)
        q = self.wq(q)[0]
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        residual, _ = self.o_proj(attn_output)
        return residual


class Step3TextDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Step3TextConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Step3TextAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            norm_eps=config.rms_norm_eps,
            max_position_embedding=config.max_position_embedding,
            head_dim=config.head_dim,
            share_q_dim=config.share_q_dim,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
        )

        layer_idx = int(prefix.split("layers.")[1].split(".")[0])
        moe_layers_enum = getattr(config, "moe_layers_enum", None)
        if moe_layers_enum is not None:
            moe_layers_idx = [int(i) for i in moe_layers_enum.strip().split(",")]
        else:
            # Default to 1dense.
            moe_layers_idx = [i for i in range(1, config.num_hidden_layers)]

        if layer_idx in moe_layers_idx:
            self.moe = FusedMoEBlock(
                config=config, quant_config=quant_config, prefix=f"{prefix}.moe"
            )
            self.share_expert = Step3TextMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.share_expert_dim,
                hidden_act="silu",
                quant_config=quant_config,
                prefix=f"{prefix}.share_expert",
            )
            self.use_moe = True
        else:
            self.mlp = Step3TextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act="silu",
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.use_moe = False
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        if self.use_moe:
            share_output = self.share_expert(hidden_states)
            moe_output = self.moe(hidden_states)
            hidden_states = share_output + moe_output
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile
class Step3TextModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.vocab_size = config.vocab_size
        self.config = config

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Step3TextDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Step3TextForCausalLM(nn.Module, SupportsPP):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.config = config
        self.vllm_config = vllm_config

        self.model = Step3TextModel(vllm_config=vllm_config, prefix=prefix)

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        qkv_params_mapping = [
            # (param_name, shard_name, relative_start_idx, relative_end_idx)
            (
                ".qkv_proj",
                ".q_proj",
                0,
                self.config.share_q_dim
                / (self.config.share_q_dim + self.config.head_dim * 2),
            ),
            (
                ".qkv_proj",
                ".k_proj",
                self.config.share_q_dim
                / (self.config.share_q_dim + self.config.head_dim * 2),
                (self.config.share_q_dim + self.config.head_dim)
                / (self.config.share_q_dim + self.config.head_dim * 2),
            ),
            (
                ".qkv_proj",
                ".v_proj",
                (self.config.share_q_dim + self.config.head_dim)
                / (self.config.share_q_dim + self.config.head_dim * 2),
                (self.config.share_q_dim + self.config.head_dim * 2)
                / (self.config.share_q_dim + self.config.head_dim * 2),
            ),
        ]
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        expert_params_mapping = [
            (".moe.experts.w13_weight", ".moe.gate_proj.weight", "w1"),
            (".moe.experts.w13_weight", ".moe.up_proj.weight", "w3"),
            (".moe.experts.w2_weight", ".moe.down_proj.weight", "w2"),
        ]

        disable_moe_stacked_params = [data[1] for data in expert_params_mapping]

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if any(
                    disable_moe_stacked_param in name
                    for disable_moe_stacked_param in disable_moe_stacked_params
                ):
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
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    for expert_id in range(loaded_weight.shape[0]):
                        loaded_weight_expert = loaded_weight[expert_id]
                        weight_loader(
                            param,
                            loaded_weight_expert,
                            name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    loaded_params.add(name)
                    break
                else:
                    for (
                        param_name,
                        weight_name,
                        start_idx,
                        end_idx,
                    ) in qkv_params_mapping:
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]
                        dim = param.shape[param.output_dim]
                        begin_idx = int(start_idx * dim)
                        end_idx = int(end_idx * dim)
                        param_slice = param.narrow(
                            param.output_dim, begin_idx, end_idx - begin_idx
                        )
                        param_slice.copy_(loaded_weight)
                        loaded_params.add(name)
                        break
                    else:
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)
        return loaded_params
