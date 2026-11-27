# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only HY V4 model compatible with HuggingFace weights (NVIDIA)."""

import typing
from collections.abc import Callable, Iterable, MutableSequence, Sequence
from itertools import islice

import torch
from torch import nn
from transformers import PreTrainedConfig

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    compute_skip_topk_layers,
    copy_pp_buffers,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    pp_topk_buffers,
)
from vllm.models.common.ops.sequence_parallel import (
    is_sequence_parallel_active,
    sp_all_gather,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.models.hy_v4.common.mrope import set_token_rope
from vllm.models.hy_v4.common.sequence_parallel import (
    IHC_SP_MIN_TOKENS,
    ihc_sequence_parallel_enabled,
)
from vllm.models.hy_v4.weight_utils import (
    HYV4_MLP_MAPPER,
    HYV4_PACKED_MODULES_MAPPING,
    HYV4_WEIGHTS_MAPPER,
    remap_block_fp8_scale,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .attention import (
    HYV4MLAAttention,
    is_skip_topk_indexer_weight,
    sparse_layer_ids,
)
from .hc import HYV4HCHeadLayer, HYV4HCLayer, IHCBoundary
from .moe import HYV4FeedForward, HYV4MoEFused

logger = init_logger(__name__)


def _normalize_hyv4_config(config: PreTrainedConfig) -> PreTrainedConfig:
    """Populate the aliases consumed by the shared MoE implementation."""
    config.router_scaling_factor = config.routed_scaling_factor
    config.num_experts = config.n_routed_experts
    config.expert_hidden_dim = config.moe_intermediate_size
    config.num_shared_experts = config.n_shared_experts
    config.route_norm = config.norm_topk_prob
    return config


class IHCCarry(typing.NamedTuple):
    """A decoder layer's MLP output whose iHC post step is still pending."""

    mlp_out: torch.Tensor
    residual: torch.Tensor
    post_gates: torch.Tensor


class HYV4DecoderLayer(nn.Module):
    """One HY V4 decoder layer: MLA attention plus a dense or MoE MLP.

    When`config.enable_ihc` is set the layer runs on `hc_mult` residual
    channels and each sub-block is wrapped by an `HYV4HCLayer` boundary;
    otherwise it uses the standard single-stream residual.
    """

    feed_forward_cls = HYV4FeedForward

    def __init__(
        self,
        config: PreTrainedConfig,
        vllm_config: VllmConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        ihc_sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.enable_ihc = getattr(config, "enable_ihc", False)
        self.defer_reduction = ihc_sequence_parallel and self.enable_ihc
        layer_idx = int(prefix.split(".")[-1])
        self.layer_idx = layer_idx

        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = HYV4MLAAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            layer_idx=layer_idx,
            topk_indices_buffer=topk_indices_buffer,
            reduce_results=not self.defer_reduction,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.mlp_layer_types[layer_idx] == "dense":
            self.mlp = self.feed_forward_cls(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                reduce_results=not self.defer_reduction,
            )
            self.block_type = "feedforward"
        else:
            self.mlp = HYV4MoEFused(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                reduce_results=not self.defer_reduction,
            )
            self.block_type = "moe"
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hc_attn_layer = HYV4HCLayer(
            config, layer_idx, prefix=f"{prefix}.hc_attn_layer"
        )
        self.hc_mlp_layer = HYV4HCLayer(
            config, layer_idx, prefix=f"{prefix}.hc_mlp_layer"
        )
        self.init_ihc_boundaries()

    def init_ihc_boundaries(self) -> None:
        """Build the fused iHC boundaries used by `forward_ihc_fused`."""
        self.ihc_attn_boundary: IHCBoundary | None = None
        self.ihc_mlp_boundary: IHCBoundary | None = None
        if self.enable_ihc and IHCBoundary.supported(self.config):
            self.ihc_attn_boundary = IHCBoundary(
                self.hc_attn_layer.hc_pre, self.input_layernorm
            )
            self.ihc_mlp_boundary = IHCBoundary(
                self.hc_mlp_layer.hc_pre, self.post_attention_layernorm
            )

    def forward_ihc_fused(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        carry: IHCCarry | None,
    ) -> IHCCarry:
        """IHC forward that leaves the MLP post step to the next boundary.

        `carry` is the previous layer's unfinished MLP post (`None` for the
        first layer, whose `hidden_states` are the iHC channels). Finish the
        returned carry with `hc_mlp_layer.post(*carry)`.
        """
        assert self.ihc_attn_boundary is not None
        assert self.ihc_mlp_boundary is not None
        full_num_tokens = positions.shape[-1]
        sequence_parallel = hidden_states.shape[0] < full_num_tokens
        if carry is None:
            boundary = self.ihc_attn_boundary(
                self.hc_attn_layer.prepare_input(hidden_states)
            )
        else:
            boundary = self.ihc_attn_boundary(
                carry.residual, carry.mlp_out, carry.post_gates
            )
        residual, hidden_states, post_gates = boundary
        if sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self._deferred_reduce(hidden_states, sequence_parallel)
        residual, hidden_states, post_gates = self.ihc_mlp_boundary(
            residual, hidden_states, post_gates
        )
        if sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        hidden_states = self._deferred_reduce(
            self.mlp(hidden_states), sequence_parallel
        )
        return IHCCarry(hidden_states, residual, post_gates)

    def _deferred_reduce(
        self, hidden_states: torch.Tensor, sequence_parallel: bool
    ) -> torch.Tensor:
        """Finish the TP reduction a sub-block skipped under `defer_reduction`."""
        if not self.defer_reduction:
            return hidden_states
        if sequence_parallel:
            return sp_reduce_scatter(hidden_states)
        return tensor_model_parallel_all_reduce(hidden_states)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.enable_ihc:
            return self._forward_ihc(positions, hidden_states)
        return self._forward_normal(positions, hidden_states, residual)

    def _forward_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard single-residual-stream forward (iHC disabled)."""
        if residual is not None:
            hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def _forward_ihc(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """IHC forward: each sub-block reduces and re-scatters the channels."""
        full_num_tokens = positions.shape[-1]
        use_ihc_sequence_parallel = hidden_states.shape[0] < full_num_tokens
        hidden_states = self.hc_attn_layer.prepare_input(hidden_states)
        hidden_states, post_gates, residual = self.hc_attn_layer.pre(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        if use_ihc_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = self._deferred_reduce(hidden_states, use_ihc_sequence_parallel)
        hidden_states = self.hc_attn_layer.post(hidden_states, residual, post_gates)

        hidden_states = self.hc_mlp_layer.prepare_input(hidden_states)
        hidden_states, post_gates, residual = self.hc_mlp_layer.pre(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        if use_ihc_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        hidden_states = self.mlp(hidden_states)
        hidden_states = self._deferred_reduce(hidden_states, use_ihc_sequence_parallel)
        hidden_states = self.hc_mlp_layer.post(hidden_states, residual, post_gates)

        # Under iHC the residual is carried inside hidden_states.
        return hidden_states, None


class HYV4Model(nn.Module):
    """HY V4 backbone."""

    decoder_layer_cls: type[HYV4DecoderLayer] = HYV4DecoderLayer
    supports_ihc_sequence_parallel = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = _normalize_hyv4_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts
        self.device = current_platform.device_type
        self.vocab_size = config.vocab_size
        self.dfly_enabled = (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.method == "dfly"
        )
        self.is_sparse = hasattr(config, "index_topk")
        if self.is_sparse:
            self.topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.topk_indices_buffer = None
        self.config = config
        self.quant_config = quant_config
        self.enable_ihc = getattr(config, "enable_ihc", False)
        self.ihc_sequence_parallel = ihc_sequence_parallel_enabled(
            vllm_config, self.supports_ihc_sequence_parallel
        )

        if get_pp_group().is_first_rank or (
            self.config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: self.decoder_layer_cls(
                config=config,
                vllm_config=vllm_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                ihc_sequence_parallel=self.ihc_sequence_parallel,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.pp_buffers = pp_topk_buffers(
            config, self.topk_indices_buffer, sparse_layer_ids(config)
        )
        if self.enable_ihc:
            # iHC head layer: merge the residual channels into one stream.
            if get_pp_group().is_last_rank:
                self.hc_head = HYV4HCHeadLayer(
                    config,
                    hidden_size=config.hidden_size,
                    hc_mult=config.hc_mult,
                    hc_eps=config.hc_eps,
                    prefix=f"{prefix}.hc_head",
                )
            else:
                self.hc_head = PPMissingLayer()
            self.make_empty_intermediate_tensors = (
                make_empty_intermediate_tensors_factory(
                    ["hidden_states"],
                    config.hc_mult * config.hidden_size,
                    self.pp_buffers,
                )
            )
        else:
            self.make_empty_intermediate_tensors = (
                make_empty_intermediate_tensors_factory(
                    ["hidden_states", "residual"], config.hidden_size, self.pp_buffers
                )
            )

        # MoE hyperparameters (consumed by EPLB).
        self.expert_weights: MutableSequence[Sequence[torch.Tensor]] = []
        self.num_expert_groups = 1
        self.moe_layers: list[nn.Module] = []
        example_layer: HYV4MoEFused | None = None
        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, HYV4DecoderLayer)
            if layer.block_type == "moe":
                assert isinstance(layer.mlp, HYV4MoEFused)
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            self.num_moe_layers = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0
            return

        self.num_moe_layers = len(self.moe_layers)
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.dfly_enabled:
            # Preserve -1 in the original batch for forced draft rejection.
            input_ids = input_ids.clamp_min(0)
        return self.embed_tokens(input_ids)

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, HYV4MoEFused):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # (param_name, weight_name, expert_id, shard_id) for weights, fp8
        # weight scales and fp8 activation scales.
        if not hasattr(self, "_cached_expert_params_mapping"):
            self._cached_expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.num_experts,
                num_redundant_experts=self.num_redundant_experts,
            )
        return self._cached_expert_params_mapping

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        full_num_tokens = positions.shape[-1]
        use_ihc_sequence_parallel = is_sequence_parallel_active(
            self.ihc_sequence_parallel, full_num_tokens, IHC_SP_MIN_TOKENS
        )
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            # In iHC mode the flattened [num_tokens, hc*h] tensor from the
            # previous PP stage is reshaped back to 3D by the first layer's
            # prepare_input, and residual is unused.
            residual = None if self.enable_ihc else intermediate_tensors["residual"]
            copy_pp_buffers(intermediate_tensors, self.pp_buffers)

        if self.start_layer < self.end_layer:
            set_token_rope(
                self.layers[self.start_layer].self_attn.rotary_emb,
                positions,
                hidden_states,
            )

        if use_ihc_sequence_parallel:
            hidden_states = sp_shard(hidden_states)
        layers = islice(self.layers, self.start_layer, self.end_layer)
        if self.uses_fused_ihc():
            carry: IHCCarry | None = None
            for layer in layers:
                carry = layer.forward_ihc_fused(positions, hidden_states, carry)
            assert carry is not None
            output = (
                self.hc_head.fused_post(*carry, self.norm)
                if get_pp_group().is_last_rank
                else None
            )
            if output is not None:
                if use_ihc_sequence_parallel:
                    output = sp_all_gather(output)[:full_num_tokens]
                return output
            hidden_states = layer.hc_mlp_layer.post(*carry)
        else:
            for layer in layers:
                hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            buffers = {k: v[: positions.shape[0]] for k, v in self.pp_buffers.items()}
            if self.enable_ihc:
                # hidden_states is [num_tokens, hc, h]; flatten the channel dim
                # for PP transfer (matches the 2D receive buffer).
                return IntermediateTensors(
                    {"hidden_states": hidden_states.flatten(1), **buffers}
                )
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual, **buffers}
            )
        if self.enable_ihc:
            hidden_states = self.hc_head(hidden_states)
        else:
            hidden_states = hidden_states + residual

        hidden_states = self.norm(hidden_states)
        if use_ihc_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        return hidden_states

    def uses_fused_ihc(self) -> bool:
        return (
            self.start_layer < self.end_layer
            and self.layers[self.start_layer].ihc_attn_boundary is not None
        )

    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        param = params_dict[name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_local_expert = False
        for expert_id in range(num_experts):
            curr_expert_weight = loaded_weight[expert_id]
            success = weight_loader(
                param,
                curr_expert_weight,
                name,
                shard_id,
                expert_id,
                return_success=True,
            )
            if success:
                loaded_local_expert = True

        return loaded_local_expert

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_topk_layers = compute_skip_topk_layers(self.config)
        # Reload may replace parameters with kernel-specific tensors.
        params_dict = dict(self.named_parameters())
        expert_mapping = self.get_expert_mapping()
        base_layer = (
            "base_layer." if any(".base_layer." in name for name in params_dict) else ""
        )
        expert_prefix = f".experts.routed_experts.{base_layer}"
        fused_mapping = (
            (".experts.gate_up_proj", f"{expert_prefix}w13_weight", ("w1", "w3")),
            (".experts.down_proj", f"{expert_prefix}w2_weight", ("w2",)),
        )
        loaded_params: set[str] = set()

        for name, weight in HYV4_WEIGHTS_MAPPER.apply(weights):
            if is_skip_topk_indexer_weight(name, skip_topk_layers):
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            if name.endswith(".bias") and name not in params_dict:
                continue

            fused = next((entry for entry in fused_mapping if entry[0] in name), None)
            if fused is not None:
                source, target, shards = fused
                target = name.replace(source, target)
                if is_pp_missing_parameter(target, self):
                    continue
                pieces = weight.chunk(2, dim=-2) if len(shards) == 2 else (weight,)
                # Load every shard, even if an earlier one is not local.
                successes = [
                    self.load_fused_expert_weights(
                        target, params_dict, piece, shard, self.config.num_experts
                    )
                    for shard, piece in zip(shards, pieces)
                ]
                if all(successes):
                    loaded_params.add(target)
                continue

            is_expert = False
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in name:
                    continue
                is_expert = True
                target = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(target, self):
                    continue
                if target not in params_dict:
                    if ".indexer." not in target:
                        logger.warning_once(
                            "Skipping unknown checkpoint weight: %s", target
                        )
                    continue
                param = params_dict[target]
                if param.weight_loader(
                    param,
                    weight,
                    target,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                ):
                    loaded_params.add(target)
                # Continue through all physical replicas of this logical expert.
            if is_expert or is_pp_missing_parameter(name, self):
                continue

            name, weight = remap_block_fp8_scale(name, weight, self.quant_config)

            if name not in params_dict:
                if ".indexer." not in name:
                    logger.warning_once("Skipping unknown checkpoint weight: %s", name)
                continue

            param = params_dict[name]
            loader: Callable[..., typing.Any] = getattr(
                param, "weight_loader", default_weight_loader
            )
            stacked_shard = getattr(weight, "shard_id", None)
            if stacked_shard is None:
                loader(param, weight)
            else:
                loader(param, weight, stacked_shard)
            loaded_params.add(name)

        return loaded_params


class HYV4ForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    model_cls = HYV4Model

    hf_to_vllm_mapper = HYV4_MLP_MAPPER | WeightsMapper(
        orig_to_new_stacked={
            ".q_a_proj": (".fused_qkv_a_proj", 0),
            ".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),
        }
    )
    packed_modules_mapping = HYV4_PACKED_MODULES_MAPPING

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        if quant_config is not None:
            quant_config.packed_modules_mapping = self.packed_modules_mapping

        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            # The head stays in the model dtype; ``enable_lm_head_fp32`` is
            # surfaced as ``head_dtype`` on the config (see HYV4Config), which
            # makes LogitsProcessor accumulate the projection straight into
            # fp32 instead of materializing an fp32 copy of the weight.
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            # With tie_word_embeddings, embed_tokens is kept on the last rank
            # (see HYV4Model.__init__) so the weight can be shared here.
            if self.config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
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
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)

        if getattr(self.config, "soft_logits_capping", False):
            soft_cap = self.config.soft_logits_capping_logits
            logits = soft_cap * torch.nn.functional.tanh(logits / soft_cap)

        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # MTP layers are loaded by the MTP head, not by the target model. They
        # appear as model.mtp_layers.<i>.* and as the layer ids past the backbone.
        mtp_start = self.config.num_hidden_layers
        drop_prefixes: dict[str, str | None] = {
            "model.mtp_layers.": None,
            "model.dfly_layers.": None,
        }
        for i in range(getattr(self.config, "num_nextn_predict_layers", 0)):
            drop_prefixes[f"model.layers.{mtp_start + i}."] = None
        if self.config.tie_word_embeddings:
            drop_prefixes["lm_head."] = None
        drop_mapper = WeightsMapper(orig_to_new_prefix=drop_prefixes)
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper | drop_mapper)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
