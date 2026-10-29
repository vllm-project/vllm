# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mistral Large 4: MLA + DSA sparse attention, MoE, NoPE layers, KV sharing.

The text model is hosted by ``PixtralForConditionalGeneration`` for the
multimodal checkpoint.
"""

from collections.abc import Callable, Iterable
from itertools import islice
from typing import cast

import regex as re
import torch
from torch import nn
from transformers import PreTrainedConfig

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
)
from vllm.model_executor.models.interfaces import EagleModelMixin, SupportsEagle3
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    spec_decode_needs_target_embed,
)
from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm
from vllm.models.mistral.common.layer_spec import MistralLayerSpec, get_layer_specs
from vllm.models.mistral.common.rope import MistralRotaryEmbedding
from vllm.models.mistral.nvidia.attention import MistralMLAAttention
from vllm.models.mistral.nvidia.mlp import MistralMLP
from vllm.models.mistral.nvidia.moe import MistralMoE
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.mla.index_group import (
    SparseMLAIndexGroupBuilder,
    get_sparse_mla_index_group_max_rows,
)


def defers_allreduce(vllm_config: VllmConfig) -> bool:
    """Whether sublayers leave their outputs TP-partial and fuse the all-reduce
    into the next residual RMSNorm."""
    parallel_config = vllm_config.parallel_config
    return bool(
        vllm_config.compilation_config.pass_config.fuse_allreduce_rms
        and vllm_config.model_config.dtype == torch.bfloat16
        and not parallel_config.enable_expert_parallel
        and not parallel_config.use_sequence_parallel_moe
    )


class MistralLarge4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: PreTrainedConfig,
        spec: MistralLayerSpec,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None,
        index_group_builder: SparseMLAIndexGroupBuilder | None,
        zero_kv_inputs: tuple[torch.Tensor, torch.Tensor],
        quant_config: QuantizationConfig | None,
        non_causal: bool = False,
        sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.defer_allreduce = defers_allreduce(vllm_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MistralMLAAttention(
            vllm_config,
            config,
            spec,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
            index_group_builder=index_group_builder,
            zero_kv_inputs=zero_kv_inputs,
            quant_config=quant_config,
            non_causal=non_causal,
            sliding_window=sliding_window,
            reduce_results=not self.defer_allreduce,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp: nn.Module
        if spec.is_moe:
            self.mlp = MistralMoE(
                vllm_config,
                config,
                prefix=f"{prefix}.mlp",
                reduce_results=not self.defer_allreduce,
            )
        else:
            self.mlp = MistralMLP(
                config.hidden_size,
                config.intermediate_size,
                quant_config,
                reduce_results=not self.defer_allreduce,
                prefix=f"{prefix}.mlp",
            )

    def residual_norm(
        self, norm: RMSNorm, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add a sublayer output to the residual stream and normalize; with
        ``defer_allreduce`` the output is a TP-partial sum reduced here."""
        if self.defer_allreduce:
            return fused_allreduce_rms_norm(hidden_states, residual, norm)
        return norm(hidden_states, residual)

    def input_norm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold the previous layer's output into the residual stream and
        normalize; the returned residual is the stream entering this layer."""
        if residual is None:
            return self.input_layernorm(hidden_states), hidden_states
        return self.residual_norm(self.input_layernorm, hidden_states, residual)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        log_positions: torch.Tensor,
        rotary_emb: MistralRotaryEmbedding,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the layer on the outputs of ``input_norm``."""
        hidden_states = self.self_attn(
            positions, hidden_states, log_positions, rotary_emb
        )
        hidden_states, residual = self.residual_norm(
            self.post_attention_layernorm, hidden_states, residual
        )
        return self.mlp(hidden_states), residual


class MistralLarge4Model(nn.Module, EagleModelMixin):
    supports_aux_hidden_states_over_pp = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.layer_specs = get_layer_specs(config)
        self.defer_allreduce = defers_allreduce(vllm_config)

        # Under PP the last stage also embeds the drafter's inputs.
        if get_pp_group().is_first_rank or spec_decode_needs_target_embed(vllm_config):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
            )
        else:
            self.embed_tokens = PPMissingLayer()
        # One RoPE table shared by all layers and indexers.
        self.rotary_emb = MistralRotaryEmbedding(config, config.qk_rope_head_dim)

        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        topk_indices_buffer = None
        index_group_builder = None
        if getattr(config, "index_topk", None) is not None:
            topk_indices_buffer = torch.empty(
                max_num_tokens, config.index_topk, dtype=torch.int32
            )
            index_group_builder = SparseMLAIndexGroupBuilder(
                topk_indices_buffer,
                get_sparse_mla_index_group_max_rows(vllm_config),
            )
        # Never-written latent inputs for KV-sharing layers and the zero rope
        # key of NoPE layers.
        zero_kv_inputs = (
            torch.zeros(max_num_tokens, config.kv_lora_rank),
            torch.zeros(max_num_tokens, 1, config.qk_rope_head_dim),
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MistralLarge4DecoderLayer(
                vllm_config,
                config,
                self.layer_specs[extract_layer_index(prefix)],
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
                index_group_builder=index_group_builder,
                zero_kv_inputs=zero_kv_inputs,
                quant_config=vllm_config.quant_config,
            ),
            prefix=f"{prefix}.layers",
        )
        _check_pp_stage(self.layer_specs, self.start_layer, self.end_layer)
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
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                assert input_ids is not None
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        residual: torch.Tensor | None = None
        log_positions = torch.log(positions.to(torch.float32) + 1)
        aux_hidden_states = self.collect_remote_aux_hidden_states(intermediate_tensors)
        num_remote_aux = len(aux_hidden_states)
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer.input_norm(hidden_states, residual)
            if idx in self.aux_hidden_state_layers:
                # Later residual norms may update the stream in place.
                aux_hidden_states.append(residual.clone())
            hidden_states, residual = layer(
                positions, hidden_states, residual, log_positions, self.rotary_emb
            )
        if not get_pp_group().is_last_rank:
            # Send the residual stream with the last output folded in: the next
            # stage then starts like the first one (residual=None), so its first
            # input_norm does not all-reduce again. PP sends only this rank's TP
            # slice and all-gathers it, so the stream must be TP-replicated.
            assert residual is not None
            if self.defer_allreduce:
                hidden_states = tensor_model_parallel_all_reduce(hidden_states)
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states + residual,
                    **self.pack_local_aux_hidden_states(
                        aux_hidden_states[num_remote_aux:]
                    ),
                }
            )
        hidden_states, residual = self.layers[self.end_layer - 1].residual_norm(
            self.norm, hidden_states, residual
        )
        if self.end_layer in self.aux_hidden_state_layers:
            aux_hidden_states.append(residual.clone())
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


def _check_pp_stage(specs: list[MistralLayerSpec], start: int, end: int) -> None:
    """A pipeline stage must own the KV cache and the top-k indices its layers
    read: KV-sharing sources and the indexer a shared layer reuses stay local."""
    for spec in specs[start:end]:
        assert start <= spec.kv_source_layer < end, (
            f"Layer {spec.layer_idx} shares the KV cache of layer "
            f"{spec.kv_source_layer}, outside its pipeline stage [{start}, {end}); "
            "set VLLM_PP_LAYER_PARTITION so they share a stage"
        )
    first = specs[start] if start < end else None
    assert first is None or not first.use_sparse or first.has_indexer, (
        f"Pipeline stage [{start}, {end}) starts at layer {start}, which reuses "
        "the top-k indices of a previous layer; set VLLM_PP_LAYER_PARTITION so "
        "every stage starts with an indexer layer"
    )


# Mistral consolidated checkpoint name -> vLLM name. Patterns match whole name
# segments, so the map renames both weights and the module names quantization
# configs list (e.g. `layers.1.attention.wq`).
_CHECKPOINT_NAME_MAP: list[tuple[str, str]] = [
    (r"^tok_embeddings(?=\.|$)", "model.embed_tokens"),
    (r"^norm(?=\.|$)", "model.norm"),
    (r"^output(?=\.|$)", "lm_head"),
    (r"^layers\.", "model.layers."),
    (r"\.attention_norm(?=\.|$)", ".input_layernorm"),
    (r"\.ffn_norm(?=\.|$)", ".post_attention_layernorm"),
    (r"\.attention\.indexer\.ssmax_scaling(?=\.|$)", ".self_attn.indexer.ssmax"),
    (r"\.attention\.ssmax_scaling(?=\.|$)", ".self_attn.ssmax"),
    (r"\.attention\.wq(?=\.|$)", ".self_attn.q_proj"),
    (r"\.attention\.wkv_a_with_mqa(?=\.|$)", ".self_attn.kv_a_proj_with_mqa"),
    (r"\.attention\.kv_a_norm(?=\.|$)", ".self_attn.kv_a_layernorm"),
    (r"\.attention\.wkv_b(?=\.|$)", ".self_attn.kv_b_proj"),
    (r"\.attention\.wo(?=\.|$)", ".self_attn.o_proj"),
    (r"\.attention\.indexer(?=\.|$)", ".self_attn.indexer"),
    (r"\.feed_forward(?=\.|$)", ".mlp"),
    (r"\.shared_experts(?=\.|$)", ".mlp.shared_experts"),
    (r"\.gate(?=\.|$)", ".mlp.gate"),
    (r"\.experts(?=\.|$)", ".mlp.experts"),
    (r"\.qscale_weight$", ".weight_scale_inv"),
]

# (fused param, checkpoint shard, shard id)
_STACKED_PARAMS: list[tuple[str, str, int]] = [
    (".gate_up_proj.", ".w1.", 0),
    (".gate_up_proj.", ".w3.", 1),
    (".down_proj.", ".w2.", -1),
    (".wk_weights_proj.", ".wk.", 0),
    (".wk_weights_proj.", ".weights_proj.", 1),
]


def _to_vllm_weight_name(name: str) -> str:
    for pattern, repl in _CHECKPOINT_NAME_MAP:
        name = re.sub(pattern, repl, name)
    return name


def _map_quant_config_layers(quant_config: QuantizationConfig, prefix: str) -> None:
    """Point the layer names of a Mistral-format quantization config at this
    model's modules.

    Such a config lists checkpoint modules (e.g. ModelOpt `quantized_layers`
    keys such as `layers.1.attention.wq` or `layers.1.experts`); a fused module
    resolves through the checkpoint modules stacked into it.
    """
    quant_config.apply_vllm_mapper(
        WeightsMapper(
            orig_to_new_regex={
                re.compile(pattern): repl for pattern, repl in _CHECKPOINT_NAME_MAP
            },
            orig_to_new_prefix={
                "model.": f"{maybe_prefix(prefix, 'model')}.",
                "lm_head": maybe_prefix(prefix, "lm_head"),
            },
        )
    )
    stacked: dict[str, list[str]] = {}
    for param_part, ckpt_part, _ in _STACKED_PARAMS:
        stacked.setdefault(param_part.strip("."), []).append(ckpt_part.strip("."))
    quant_config.packed_modules_mapping = {
        **quant_config.packed_modules_mapping,
        **stacked,
    }


def _load_weights(
    model: nn.Module,
    mistral_model: MistralLarge4Model,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load a Mistral consolidated checkpoint into ``model``.

    ``model`` holds ``mistral_model`` as ``model.model`` and the LM head as
    ``model.lm_head``.
    """
    config = mistral_model.config
    params = dict(model.named_parameters())
    expert_mapping = fused_moe_make_expert_params_mapping(
        model,
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
        num_experts=getattr(config, "n_routed_experts", None) or 0,
    )
    loaded: set[str] = set()
    lerope_alpha: torch.Tensor | None = None

    for ckpt_name, weight in weights:
        if ckpt_name == "lerope.alpha":
            lerope_alpha = weight
            continue
        if ckpt_name.startswith("mtp_"):
            # Speculative-decoding heads are loaded by their draft model.
            continue
        # Layers beyond the configured depth (e.g. a truncated debug model).
        layer = re.match(r"layers\.(\d+)\.", ckpt_name)
        if layer is not None and int(layer.group(1)) >= config.num_hidden_layers:
            continue
        name = _to_vllm_weight_name(ckpt_name)
        if is_pp_missing_parameter(name, model):
            continue

        if ".mlp.experts." in name:
            is_expert_weight = False
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in name:
                    continue
                is_expert_weight = True
                mapped = name.replace(weight_name, param_name)
                param = params[mapped]
                loader = cast(Callable[..., bool], param.weight_loader)
                if loader(
                    param,
                    weight,
                    mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                ):
                    loaded.add(mapped)
                    break
            else:
                # A known expert weight that is not local to this EP rank is skipped.
                if not is_expert_weight:
                    raise ValueError(f"Unexpected expert weight: {ckpt_name}")
            continue

        for param_part, ckpt_part, stacked_shard_id in _STACKED_PARAMS:
            if ckpt_part not in name:
                continue
            name = name.replace(ckpt_part, param_part)
            param = params[name]
            if stacked_shard_id < 0:
                param.weight_loader(param, weight)
            else:
                param.weight_loader(param, weight, stacked_shard_id)
            break
        else:
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, weight)
        loaded.add(name)

    # The default loader skips this check for quantized models.
    if missing := sorted(set(params) - loaded):
        raise ValueError(f"Parameters not loaded from the checkpoint: {missing}")

    if getattr(config, "lerope", False):
        assert lerope_alpha is not None, "Checkpoint is missing lerope.alpha"
        mistral_model.rotary_emb.set_lerope_alpha(lerope_alpha)
    return loaded


class MistralLarge4ForCausalLM(nn.Module, SupportsEagle3):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        if vllm_config.quant_config is not None:
            _map_quant_config_layers(vllm_config.quant_config, prefix)
        self.model = MistralLarge4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(
            config.vocab_size, soft_cap=getattr(config, "softmax_tanh_gating", None)
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return _load_weights(self, self.model, weights)
