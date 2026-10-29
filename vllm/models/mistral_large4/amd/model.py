# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mistral Large 4: MLA + DSA sparse attention, MoE, NoPE layers, KV sharing.

The text model is hosted by ``PixtralForConditionalGeneration`` for the
multimodal checkpoint.
"""

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, cast

import regex as re
import torch
from torch import nn
from transformers import PreTrainedConfig

from vllm.config import VllmConfig
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
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm
from vllm.models.mistral.amd.attention import MistralMLAAttention
from vllm.models.mistral.amd.mlp import MistralMLP
from vllm.models.mistral.amd.moe import MistralMoE
from vllm.models.mistral.common.layer_spec import MistralLayerSpec, get_layer_specs
from vllm.models.mistral.common.rope import MistralRotaryEmbedding
from vllm.models.mistral_large4.amd.specialized_decode import (
    DecodeConfig,
    specialized_decode_config,
)
from vllm.v1.attention.backends.mla.index_group import (
    SparseMLAIndexGroupBuilder,
    get_sparse_mla_index_group_max_rows,
)

if TYPE_CHECKING:
    from vllm.models.mistral_large4.amd.specialized_decode.runtime.decode import (
        SpecializedDecode,
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
        quant_config: QuantizationConfig | None = None,
        non_causal: bool = False,
        sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        if quant_config is None:
            quant_config = vllm_config.quant_config
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
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        # before any module allocates weights: an unsupported deployment that
        # asked for the specialized decode fails here, listing every reason
        self.specialized_decode_config: DecodeConfig | None = specialized_decode_config(
            vllm_config
        )
        self.specialized_decode: SpecializedDecode | None = None
        config = vllm_config.model_config.hf_config
        self.config = config
        self.layer_specs = get_layer_specs(config)
        assert vllm_config.parallel_config.pipeline_parallel_size == 1, (
            "Pipeline parallelism is not supported"
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
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

        self.layers = nn.ModuleList(
            MistralLarge4DecoderLayer(
                vllm_config,
                config,
                spec,
                prefix=f"{prefix}.layers.{spec.layer_idx}",
                topk_indices_buffer=topk_indices_buffer,
                index_group_builder=index_group_builder,
                zero_kv_inputs=zero_kv_inputs,
                quant_config=vllm_config.quant_config,
            )
            for spec in self.layer_specs
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def build_specialized_decode(self, vllm_config: VllmConfig) -> None:
        from vllm.models.mistral_large4.amd.specialized_decode.runtime.decode import (
            SpecializedDecode,
        )

        assert self.specialized_decode_config is not None
        self.specialized_decode = SpecializedDecode.build(
            self, self.specialized_decode_config, vllm_config
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: object = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        # getattr: tests build bare models without __init__
        decode = getattr(self, "specialized_decode", None)
        if decode is not None:
            step = decode.plan(positions, intermediate_tensors)
            hidden = (
                decode.run(step, input_ids, positions, inputs_embeds)
                if step is not None
                else self._stock_forward(input_ids, positions, inputs_embeds)
            )
            decode.after_step()
            return hidden
        return self._stock_forward(input_ids, positions, inputs_embeds)

    def _stock_forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            assert input_ids is not None
            hidden_states = self.embed_input_ids(input_ids)
        log_positions = torch.log(positions.to(torch.float32) + 1)
        residual = None
        aux_hidden_states: list[torch.Tensor] = []
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer.input_norm(hidden_states, residual)
            if idx in self.aux_hidden_state_layers:
                # Later residual norms may update the stream in place.
                aux_hidden_states.append(residual.clone())
            hidden_states, residual = layer(
                positions, hidden_states, residual, log_positions, self.rotary_emb
            )
        hidden_states, residual = self.layers[-1].residual_norm(
            self.norm, hidden_states, residual
        )
        if len(self.layers) in self.aux_hidden_state_layers:
            aux_hidden_states.append(residual.clone())
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


# Mistral consolidated checkpoint name -> vLLM name. Patterns match whole name
# segments so they map both checkpoint tensors and quantization-config modules.
_CHECKPOINT_NAME_MAP: list[tuple[str, str]] = [
    (r"^tok_embeddings(?=\.|$)", "model.embed_tokens"),
    (r"^norm(?=\.|$)", "model.norm"),
    (r"^output(?=\.|$)", "lm_head"),
    (r"^layers\.", "model.layers."),
    (r"^(model\.layers\.\d+)\.attention_norm(?=\.|$)", r"\1.input_layernorm"),
    (r"^(model\.layers\.\d+)\.ffn_norm(?=\.|$)", r"\1.post_attention_layernorm"),
    (
        r"^(model\.layers\.\d+)\.attention\.indexer\.ssmax_scaling(?=\.|$)",
        r"\1.self_attn.indexer.ssmax",
    ),
    (
        r"^(model\.layers\.\d+)\.attention\.ssmax_scaling(?=\.|$)",
        r"\1.self_attn.ssmax",
    ),
    (r"^(model\.layers\.\d+)\.attention\.wq(?=\.|$)", r"\1.self_attn.q_proj"),
    (
        r"^(model\.layers\.\d+)\.attention\.wkv_a_with_mqa(?=\.|$)",
        r"\1.self_attn.kv_a_proj_with_mqa",
    ),
    (
        r"^(model\.layers\.\d+)\.attention\.kv_a_norm(?=\.|$)",
        r"\1.self_attn.kv_a_layernorm",
    ),
    (
        r"^(model\.layers\.\d+)\.attention\.wkv_b(?=\.|$)",
        r"\1.self_attn.kv_b_proj",
    ),
    (r"^(model\.layers\.\d+)\.attention\.wo(?=\.|$)", r"\1.self_attn.o_proj"),
    (
        r"^(model\.layers\.\d+)\.attention\.indexer(?=\.|$)",
        r"\1.self_attn.indexer",
    ),
    (r"^(model\.layers\.\d+)\.feed_forward(?=\.|$)", r"\1.mlp"),
    (
        r"^(model\.layers\.\d+)\.shared_experts(?=\.|$)",
        r"\1.mlp.shared_experts",
    ),
    (r"^(model\.layers\.\d+)\.gate(?=\.|$)", r"\1.mlp.gate"),
    (r"^(model\.layers\.\d+)\.experts(?=\.|$)", r"\1.mlp.experts"),
    (r"\.qscale_weight$", ".weight_scale_inv"),
]

_QUANT_CONFIG_NAME_MAP: list[tuple[str, str]] = [
    *_CHECKPOINT_NAME_MAP,
    (
        r"^(model\.layers\.\d+\.mlp(?:\.shared_experts|\.experts\.\d+)?)"
        r"\.w1(?=\.|$)",
        r"\1.gate_proj",
    ),
    (
        r"^(model\.layers\.\d+\.mlp(?:\.shared_experts|\.experts\.\d+)?)"
        r"\.w2(?=\.|$)",
        r"\1.down_proj",
    ),
    (
        r"^(model\.layers\.\d+\.mlp(?:\.shared_experts|\.experts\.\d+)?)"
        r"\.w3(?=\.|$)",
        r"\1.up_proj",
    ),
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
    quant_config.apply_vllm_mapper(
        WeightsMapper(
            orig_to_new_regex={
                re.compile(pattern): repl for pattern, repl in _QUANT_CONFIG_NAME_MAP
            },
            orig_to_new_prefix={
                "model.": f"{maybe_prefix(prefix, 'model')}.",
                "lm_head": maybe_prefix(prefix, "lm_head"),
            },
        )
    )
    quant_config.packed_modules_mapping = {
        **quant_config.packed_modules_mapping,
        "gate_up_proj": ["gate_proj", "up_proj"],
        "wk_weights_proj": ["wk", "weights_proj"],
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
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            config.vocab_size, soft_cap=getattr(config, "softmax_tanh_gating", None)
        )
        # Required by multimodal wrappers; pipeline parallelism is unsupported.
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def get_model_state_cls(self):
        from vllm.models.mistral_large4.amd.model_state import (
            MistralLarge4ModelState,
        )

        return MistralLarge4ModelState

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: object = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return _load_weights(self, self.model, weights)
