# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Intern-S2-Mobius model."""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, replace, set_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config

from .interfaces import EagleModelMixin
from .interns2_preview import (
    InternS2PreviewForConditionalGeneration,
    InternS2PreviewProcessingInfo,
)
from .qwen2_moe import Qwen2MoeMLP
from .qwen3_5 import (
    Qwen3_5ForCausalLMBase,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
)
from .qwen3_5_mtp import Qwen3_5MoeMTP
from .qwen3_next import Qwen3NextAttention, _all_gather_hidden_and_residual
from .qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)


class InternS2MobiusProcessingInfo(InternS2PreviewProcessingInfo):
    pass


class InternS2MobiusMetaMoeBlock(nn.Module):
    """A routed MoE bank shared by multiple decoder layers."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config

        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > self.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.n_routed_experts}."
            )

        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.experts = FusedMoEFactory(
            gate=self.gate,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            activation=config.hidden_act,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        orig_shape = hidden_states.shape
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.experts.is_internal_router:
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

        if self.is_sequence_parallel and not already_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(orig_shape)


class InternS2MobiusSharedExpertBlock(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        is_sequence_parallel = vllm_config.parallel_config.use_sequence_parallel_moe
        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )
        self.shared_expert = Qwen2MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=vllm_config.quant_config,
            expert_gate=self.shared_expert_gate,
            is_sequence_parallel=is_sequence_parallel,
            prefix=f"{prefix}.shared_expert",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.shared_expert(hidden_states)


class InternS2MobiusDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        self.num_blocks = config.num_blocks
        self.use_attn_reduce_scatter_for_moe = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
        )

        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.self_attn",
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        self.mlp = InternS2MobiusSharedExpertBlock(
            vllm_config=vllm_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.ffn_layer_scale = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        meta_mlp: nn.ModuleList,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_num_tokens = positions.shape[-1]
        input_is_sequence_parallel = (
            self.use_attn_reduce_scatter_for_moe
            and residual is not None
            and hidden_states.shape[0] != full_num_tokens
        )

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if input_is_sequence_parallel:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[:full_num_tokens]

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states)
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )

        if self.layer_scale:
            scale = self.attn_layer_scale.to(hidden_states.dtype)
            hidden_states = hidden_states * (scale[0] + 1)

        if self.use_attn_reduce_scatter_for_moe:
            tp_world_size = get_tensor_model_parallel_world_size()
            sp_pad = (-hidden_states.shape[0]) % tp_world_size
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, sp_pad))
            hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
            if not input_is_sequence_parallel:
                residual = sequence_parallel_chunk(residual)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        block_idx = self.layer_idx % self.num_blocks
        routed_output = meta_mlp[block_idx](
            hidden_states,
            already_sequence_parallel=self.use_attn_reduce_scatter_for_moe,
        )
        hidden_states = routed_output + self.mlp(hidden_states)

        if self.layer_scale:
            scale = self.ffn_layer_scale.to(hidden_states.dtype)
            hidden_states = hidden_states * (scale[0] + 1)

        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class InternS2MobiusModel(nn.Module, EagleModelMixin):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )
        self.meta_mlp = nn.ModuleList(
            InternS2MobiusMetaMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.meta_mlp.{idx}",
            )
            for idx in range(config.num_blocks)
        )

        def get_layer(prefix: str) -> InternS2MobiusDecoderLayer:
            return InternS2MobiusDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int, ...] = ()

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
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        full_num_tokens = positions.shape[-1]
        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if (
                hidden_states.shape[0] != full_num_tokens
                and not layer.use_attn_reduce_scatter_for_moe
            ):
                hidden_states, residual = _all_gather_hidden_and_residual(
                    hidden_states,
                    residual,
                    full_num_tokens,
                    self.config.hidden_size,
                )
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                meta_mlp=self.meta_mlp,
            )
            if (layer_idx + 1) in self.aux_hidden_state_layers and hidden_states.shape[
                0
            ] != full_num_tokens:
                hidden_states, residual = _all_gather_hidden_and_residual(
                    hidden_states,
                    residual,
                    full_num_tokens,
                    self.config.hidden_size,
                )
            self._maybe_add_hidden_state(
                aux_hidden_states,
                layer_idx + 1,
                hidden_states,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if hidden_states.shape[0] != full_num_tokens:
            hidden_states, residual = _all_gather_hidden_and_residual(
                hidden_states,
                residual,
                full_num_tokens,
                self.config.hidden_size,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class InternS2MobiusForCausalLM(Qwen3_5ForCausalLMBase):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        self.scheduler_config = vllm_config.scheduler_config

        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Intern-S2-Mobius does not support 'all' prefix caching; "
                "use '--mamba-cache-mode=align' instead."
            )

        nn.Module.__init__(self)
        self.config = config
        self.model = InternS2MobiusModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )


class InternS2MobiusMTP(Qwen3_5MoeMTP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        speculative_config = vllm_config.speculative_config
        if speculative_config is None or speculative_config.draft_model_config is None:
            raise ValueError(
                "Intern-S2-Mobius MTP requires a draft model configuration."
            )

        draft_vllm_config = replace(
            vllm_config,
            model_config=speculative_config.draft_model_config,
        )
        with set_current_vllm_config(draft_vllm_config, prefix=prefix):
            super().__init__(vllm_config=draft_vllm_config, prefix=prefix)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=InternS2MobiusProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class InternS2MobiusForConditionalGeneration(InternS2PreviewForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        if getattr(config, "image_token_index", None) is None:
            config.image_token_index = config.image_token_id
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        self.use_deepstack = False

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = InternS2MobiusForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        meta_mlp = self.language_model.model.meta_mlp
        if not meta_mlp:
            raise RuntimeError("Intern-S2-Mobius requires at least one MoE bank.")

        self.moe_layers = [moe.experts for moe in meta_mlp]
        example_moe = meta_mlp[0]
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.language_model.model.meta_mlp:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()
