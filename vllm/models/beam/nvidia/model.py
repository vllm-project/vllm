# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Beam model compatible with HuggingFace weights.

Beam is a decoder-only sparse MoE causal LM. Its config-gated features
are NoPE layers with SSMax query scaling, hybrid sliding-window attention,
per-head Q/K RMSNorm, sandwich norm, depth-scaled residuals, and sigmoid
attention output gating. The MoE router uses grouped sigmoid ``noaux_tc``
selection.
"""

import math
from collections.abc import Iterable
from functools import partial
from typing import cast

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.model_executor.layers.fused_moe import MoERunner
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.beam import BeamConfig

from .activation import beam_softcap_logits
from .allreduce import (
    BeamAllReduceWorkspace,
    beam_allreduce_norm_residual_norm,
    beam_finalize_allreduce_norm_residual_norm,
)
from .attention import BeamAttention, use_merged_qkv_gate
from .layernorm import RMSNorm, beam_norm_residual_norm, beam_rms_norm
from .moe import BeamMLP, BeamSparseMoeBlock


def _depth_scaled_coeffs(layer_id: int, is_attention: bool) -> tuple[float, float]:
    """Depth-scaled residual coefficients ``(x_scale, r_scale)`` for one sublayer.

    ``num_adds`` counts residual adds up to and including this one, with a
    1-based layer index ``L = layer_id + 1``: ``2L`` for the attention add,
    ``2L + 1`` for the MLP add.
    """
    one_based = layer_id + 1
    num_adds = 2 * one_based if is_attention else 2 * one_based + 1
    denom = math.sqrt(num_adds)
    return 1.0 / denom, math.sqrt(num_adds - 1) / denom


class BeamDecoderLayer(nn.Module):
    """Sandwich-norm block carrying residual and next normalized input separately."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        allreduce_workspace: BeamAllReduceWorkspace | None = None,
    ) -> None:
        super().__init__()
        config = cast(BeamConfig, vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.layer_id = extract_layer_index(prefix)
        self.rms_norm_eps = config.rms_norm_eps
        fuse_attention_reduce = (
            allreduce_workspace is not None and not config.attention_bias
        )

        self.self_attn = BeamAttention(
            config,
            cache_config,
            quant_config,
            prefix=f"{prefix}.self_attn",
            reduce_results=not fuse_attention_reduce,
        )

        if self.layer_id in config.mlp_only_layers:
            self.mlp: nn.Module = BeamMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                reduce_results=allreduce_workspace is None,
            )
        else:
            self.mlp = BeamSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
                reduce_results=allreduce_workspace is None,
            )

        fused_boundary = (
            partial(beam_allreduce_norm_residual_norm, workspace=allreduce_workspace)
            if allreduce_workspace is not None
            else beam_norm_residual_norm
        )
        self.attention_boundary = (
            fused_boundary if fuse_attention_reduce else beam_norm_residual_norm
        )
        fuse_mlp_reduce = allreduce_workspace is not None and (
            isinstance(self.mlp, BeamMLP)
            or self.mlp.experts.moe_config.skip_final_all_reduce
        )
        self.mlp_boundary = (
            fused_boundary if fuse_mlp_reduce else beam_norm_residual_norm
        )
        self.finalize_boundary = None
        if fuse_mlp_reduce and isinstance(self.mlp, BeamSparseMoeBlock):
            assert allreduce_workspace is not None
            moe_config = self.mlp.experts.moe_config
            allow_ep = moe_config.ep_size == allreduce_workspace.world_size
            max_tokens = allreduce_workspace.max_tokens
            if allow_ep:
                max_tokens = min(max_tokens, allreduce_workspace.ep_finalize_max_tokens)
            self.mlp.defer_finalize(max_tokens, allow_ep=allow_ep)
            self.finalize_boundary = partial(
                beam_finalize_allreduce_norm_residual_norm,
                workspace=allreduce_workspace,
                routed_scale=self.mlp.experts.routed_scaling_factor,
                oneshot_max_tokens=(
                    allreduce_workspace.ep_finalize_oneshot_max_tokens
                    if allow_ep
                    else 0
                ),
            )

        self.input_layernorm = RMSNorm(config.hidden_size)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = (
            RMSNorm(config.hidden_size) if config.enable_sandwich_norm else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.hidden_size) if config.enable_sandwich_norm else None
        )

        self.attn_scales = (
            _depth_scaled_coeffs(self.layer_id, is_attention=True)
            if config.self_attn_depth_scaled_residual
            else None
        )
        self.mlp_scales = (
            _depth_scaled_coeffs(self.layer_id, is_attention=False)
            if config.mlp_depth_scaled_residual
            else None
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        normalized_states: torch.Tensor | None = None,
        next_norm_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = (
            beam_rms_norm(hidden_states, self.input_layernorm.weight, self.rms_norm_eps)
            if normalized_states is None
            else normalized_states
        )
        x = self.self_attn(positions, x)
        hidden_states, x = self.attention_boundary(
            x,
            hidden_states,
            self.post_attention_layernorm.weight
            if self.post_attention_layernorm is not None
            else None,
            self.rms_norm_eps if self.post_attention_layernorm is not None else None,
            self.pre_feedforward_layernorm.weight,
            self.rms_norm_eps,
            self.attn_scales,
        )
        assert x is not None
        x = self.mlp(x)
        boundary = self.mlp_boundary
        if isinstance(x, UnfinalizedMoEOutput):
            assert self.finalize_boundary is not None
            boundary = self.finalize_boundary
        return boundary(
            x,
            hidden_states,
            self.post_feedforward_layernorm.weight
            if self.post_feedforward_layernorm is not None
            else None,
            self.rms_norm_eps if self.post_feedforward_layernorm is not None else None,
            next_norm_weight,
            self.rms_norm_eps if next_norm_weight is not None else None,
            self.mlp_scales,
        )


class BeamModel(nn.Module, EagleModelMixin):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            # Renamed so it cannot be matched as a `gate_up_proj` shard by
            # `packed_modules_mapping` consumers such as LoRA.
            ".self_attn.gate_proj": ".self_attn.output_gate_proj",
        },
        orig_to_new_stacked={
            # weight_name: (param_name, shard_id)
            ".self_attn.q_proj": (".self_attn.qkv_proj", "q"),
            ".self_attn.k_proj": (".self_attn.qkv_proj", "k"),
            ".self_attn.v_proj": (".self_attn.qkv_proj", "v"),
            # `.mlp.experts.N.gate_proj` does not contain `.mlp.gate_proj`, so
            # expert weights fall through to `MoERunner.load_weights` (which is
            # also what keeps them correct under EP and redundant experts).
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
        },
    )

    @classmethod
    def get_weights_mapper(cls, config: BeamConfig, tp_size: int) -> WeightsMapper:
        mapper = cls.hf_to_vllm_mapper
        if not use_merged_qkv_gate(config, tp_size):
            return mapper
        return WeightsMapper(
            orig_to_new_substr=mapper.orig_to_new_substr,
            orig_to_new_stacked={
                **mapper.orig_to_new_stacked,
                **{
                    f".self_attn.{name}": (".self_attn.qkv_gate_proj", shard)
                    for shard, name in enumerate(
                        ("q_proj", "k_proj", "v_proj", "output_gate_proj")
                    )
                },
            },
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hf_to_vllm_mapper = self.get_weights_mapper(
            config, get_tensor_model_parallel_world_size()
        )
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # Beam drives SWA per layer off `layer_types`, passing an explicit
        # `per_layer_sliding_window` (None on the full-attention layers).
        # Without this, those layers would pick up `cache_config.sliding_window`
        # as a fallback in `Attention.__init__` and silently get a window.
        if cache_config is not None:
            cache_config.sliding_window = None

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        self.embed_tokens: VocabParallelEmbedding | PPMissingLayer
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        tp_group = get_tp_group()
        self.allreduce_workspace = BeamAllReduceWorkspace.create(
            tp_group.cpu_group,
            config.hidden_size,
            max(
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.scheduler_config.max_num_seqs,
            ),
            vllm_config.model_config.dtype,
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: BeamDecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                allreduce_workspace=self.allreduce_workspace,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = (
            RMSNorm(config.hidden_size)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

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
                assert input_ids is not None
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, None
        )
        normalized_states = None
        for index in range(self.start_layer, self.end_layer):
            next_norm = (
                self.layers[index + 1].input_layernorm
                if index + 1 < self.end_layer
                else self.norm
            )
            hidden_states, normalized_states = self.layers[index](
                positions,
                hidden_states,
                normalized_states,
                next_norm.weight if isinstance(next_norm, RMSNorm) else None,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, index + 1, hidden_states, None
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = (
            beam_rms_norm(hidden_states, self.norm.weight, self.config.rms_norm_eps)
            if normalized_states is None
            else normalized_states
        )
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=[
                ".weight_scale",
                "_weight_scale",
                ".input_scale",
                "_input_scale",
            ],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class BeamForCausalLM(
    nn.Module,
    SupportsPP,
    MixtureOfExperts,
    SupportsQuant,
    SupportsEagle3,
):
    """Beam causal LM. LoRA support is deferred to a follow-up change."""

    hf_to_vllm_mapper = BeamModel.hf_to_vllm_mapper
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "qkv_gate_proj": ["q_proj", "k_proj", "v_proj", "output_gate_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    fall_back_to_pt_during_load = False

    def process_weights_after_loading(self) -> None:
        for module in self.model.modules():
            if isinstance(module, BeamAttention):
                module.warmup_projection()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        if parallel_config.enable_eplb:
            # Expert parallelism is validated; expert rebalancing is not.
            raise NotImplementedError(
                "Beam has not been validated with EPLB; run without --enable-eplb."
            )
        if parallel_config.use_sequence_parallel_moe:
            # Data-parallel attention with expert-parallel MoE routes tokens
            # through the sequence-parallel path, which is not yet validated.
            raise NotImplementedError(
                "Beam has not been validated with sequence-parallel MoE "
                "(data parallelism with expert parallelism)."
            )
        self.config = config
        self.quant_config = quant_config

        self.model = BeamModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if config.tie_word_embeddings:
            raise NotImplementedError("Beam requires tie_word_embeddings=False.")
        # Untied LM head, in the model dtype. An fp32 head is available the
        # standard way, via `head_dtype` on the config (see
        # `LogitsProcessor._apply_head`), rather than by pinning the parameter
        # dtype here.
        self.lm_head: ParallelLMHead | PPMissingLayer
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        # Apply the softcap in fp32 while keeping the head in the model dtype.
        self.final_logit_softcapping = config.final_logit_softcapping
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # MoE metadata for the MixtureOfExperts interface.
        self.moe_layers: list[MoERunner] = []
        example_layer: BeamSparseMoeBlock | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, BeamDecoderLayer)
            if isinstance(layer.mlp, BeamSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)
        if example_layer is None:
            # A pipeline stage may contain only the initial dense layers.
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_shared_experts = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0
            return

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = config.n_group or 1
        self.num_shared_experts = 0
        moe_config = example_layer.experts.moe_config
        self.num_logical_experts = moe_config.num_logical_experts
        self.num_physical_experts = moe_config.num_experts
        self.num_local_physical_experts = moe_config.num_local_experts
        self.num_routed_experts = config.num_experts
        self.num_redundant_experts = (
            self.num_physical_experts - self.num_logical_experts
        )

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.moe_layers:
            layer.update_expert_map()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None and self.final_logit_softcapping is not None:
            logits = beam_softcap_logits(logits, self.final_logit_softcapping)
        return logits

    def _strip_lm_head_padding(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        # Strip padding from the LM head's input dimension.
        hidden_size = self.config.hidden_size
        for name, weight in weights:
            if name.endswith("lm_head.weight") and weight.shape[-1] > hidden_size:
                weight = weight[..., :hidden_size].contiguous()
            yield name, weight

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(self._strip_lm_head_padding(weights))
