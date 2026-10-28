# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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
"""Inference-only Qwen3.5 Series compatible with HuggingFace weights."""

from collections.abc import Iterable
from itertools import islice
from typing import ClassVar, cast

import torch
from torch import nn
from transformers import (
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
)

from vllm.config import MultiModalConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
    flashinfer_trtllm_fused_allreduce_norm,
    fused_allreduce_gemma_rms_norm,
)
from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    _require_is_multimodal,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5MoeProcessingInfo,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    mrope_positions_factor,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config

from .ops.fused_gdn_gated_norm import fused_gdn_gated_rms_norm

logger = init_logger(__name__)


class Qwen3_5RMSNorm(nn.Module):
    """Gemma-style RMS normalization backed by flashinfer kernels.

    When ``residual`` is given, the fused add + norm runs in place and the
    updated ``(x, residual)`` pair is returned.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        set_weight_attrs(self.weight, {"dummy_weight_value": 0.0})
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm

        if residual is None:
            return gemma_rmsnorm(x, self.weight, self.variance_epsilon)

        # gemma_fused_add_rmsnorm mutates x and residual in place.
        gemma_fused_add_rmsnorm(x, residual, self.weight, self.variance_epsilon)
        return x, residual


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        is_moe_layer = config.model_type == "qwen3_5_moe_text"
        self.use_attn_reduce_scatter_for_moe = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
            and is_moe_layer
        )
        # Defer row-parallel all-reduces to the following norm
        # (deepseek_v32 pattern); exclude PP and SP.
        self.use_fused_allreduce_norm = (
            get_tensor_model_parallel_world_size() > 1
            and parallel_config.pipeline_parallel_size == 1
            and not self.use_attn_reduce_scatter_for_moe
            and flashinfer_trtllm_fused_allreduce_norm is not None
        )
        # Whether this layer's MLP/MoE leaves an un-reduced partial, i.e.
        # whether the *next* norm (or the final norm) must do the all-reduce.
        self.mlp_output_is_partial = self.use_fused_allreduce_norm
        reduce_results = not (
            self.use_attn_reduce_scatter_for_moe or self.use_fused_allreduce_norm
        )

        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                reduce_results=reduce_results,
            )
            # Recover the inductor gated-norm fusion with a Triton kernel that
            # also reads z strided from the packed qkvz projection.
            self.linear_attn._fused_gdn_gated_rms_norm = fused_gdn_gated_rms_norm
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                reduce_results=reduce_results,
                mrope_positions_factor=mrope_positions_factor(vllm_config),
            )
            self.self_attn.use_fused_output_gate = True
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
                reduce_results=not self.use_fused_allreduce_norm,
            )
            if self.use_fused_allreduce_norm:
                # reduce_results=False is only honored on the plain-TP late-AR
                # path (skip_final_all_reduce); EP/all2all, SP, zero experts,
                # and replicated shared experts all produce or add a complete
                # output, which a downstream fused norm would all-reduce again.
                moe_config = self.mlp.experts.moe_config
                moe_partial_ok = (
                    moe_config.skip_final_all_reduce
                    and moe_config.tp_size > 1
                    and moe_config.ep_size == 1
                    and moe_config.dp_size == 1
                    and not self.mlp.replicate_shared_expert
                )
                if moe_config.skip_final_all_reduce and not moe_partial_ok:
                    moe_config.skip_final_all_reduce = False
                self.mlp_output_is_partial = moe_partial_ok
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                reduce_results=not self.use_fused_allreduce_norm,
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        **kwargs: object,
    ):
        if not self.use_fused_allreduce_norm:
            return super().forward(hidden_states, residual, positions, **kwargs)

        # layer_scale is a per-channel multiply, which commutes with the
        # deferred all-reduce.
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif self.mlp_output_is_partial:
            hidden_states, residual = fused_allreduce_gemma_rms_norm(
                hidden_states, residual, self.input_layernorm
            )
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        hidden_states, residual = fused_allreduce_gemma_rms_norm(
            hidden_states, residual, self.post_attention_layernorm
        )
        hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual


class Qwen3_5Model(nn.Module, EagleModelMixin):
    # Qwen3.5 ships the GDN in_proj checkpoints separately (qwen3-next
    # pre-fuses them); fuse them on top of the qwen3-next QKV/gate_up mapping.
    hf_to_vllm_mapper = Qwen3NextModel.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_stacked={
            ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
            ".in_proj_z": (".in_proj_qkvz", 3),
            ".in_proj_b": (".in_proj_ba", 0),
            ".in_proj_a": (".in_proj_ba", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = (
            vllm_config.model_config.hf_text_config
        )
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        self.quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3_5DecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        # Mirror of the decoder layers' flag (uniform across layers): when
        # True, every layer returns its MLP/MoE output as an un-reduced TP
        # partial, so the final norm and eagle aux captures must all-reduce.
        self.mlp_output_is_partial = getattr(
            self.layers[self.start_layer], "mlp_output_is_partial", False
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

    @property
    def use_sequence_parallel(self) -> bool:
        return self.layers[self.start_layer].use_attn_reduce_scatter_for_moe

    def _maybe_add_hidden_state(
        self,
        aux_hidden_states: list[torch.Tensor],
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if residual is None and layer_idx in self.aux_hidden_state_layers:
            # The layer-0 capture aliases the embeddings, which become the
            # first residual and are mutated in place by the next layer's
            # fused-add norm; capture by value.
            hidden_states = hidden_states.clone()
        if (
            residual is not None
            and self.mlp_output_is_partial
            and layer_idx in self.aux_hidden_state_layers
        ):
            # hidden_states is an un-reduced TP partial; the aux capture needs
            # the complete hidden state.
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        return super()._maybe_add_hidden_state(
            aux_hidden_states, layer_idx, hidden_states, residual
        )

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
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        full_num_tokens = positions.shape[-1]
        if self.use_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)
            assert residual is None

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if self.mlp_output_is_partial:
            # The last layer's MLP/MoE output is an un-reduced TP partial;
            # fuse its all-reduce into the final norm.
            hidden_states, _ = fused_allreduce_gemma_rms_norm(
                hidden_states, residual, self.norm
            )
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        if self.use_sequence_parallel:
            if aux_hidden_states:
                hidden_size = hidden_states.shape[-1]
                hidden_states = torch.cat([hidden_states, *aux_hidden_states], dim=-1)
                hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
                hidden_states = hidden_states[:full_num_tokens]
                hidden_states, *aux_hidden_states = hidden_states.split(
                    hidden_size, dim=-1
                )
            else:
                hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
                hidden_states = hidden_states[:full_num_tokens]
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # GDN fused projections.
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
    # Maps PEFT embed/lm_head LoRA targets onto vLLM embedding wrappers.
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    # Some community text-only checkpoints keep the extraneous
    # `model.language_model.` prefix inherited from the VL training stack.
    # Strip it so both prefixed and clean checkpoints load correctly.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model.", "mtp.": None},
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config

        scheduler_config = vllm_config.scheduler_config
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_logits_local(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # set MoE hyperparameters
        self.set_moe_parameters()


########################################################
# Qwen3_5-Dense
########################################################


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    supports_multimodal_pruning = True

    hf_to_vllm_mapper = (
        Qwen3VLForConditionalGeneration.hf_to_vllm_mapper
        | WeightsMapper(orig_to_new_prefix={"mtp.": None})
    )

    packed_modules_mapping = Qwen3VLForConditionalGeneration.packed_modules_mapping | {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = cast(
            MultiModalConfig, vllm_config.model_config.multimodal_config
        )

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )
        self.video_pruning_rate = self.multimodal_config.video_pruning_rate
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        # attributes needed by EVS-related functions inherited from Qwen3-VL
        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                input_norm=build_mm_input_norm(self.model_config),
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5ForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Qwen3.5.

        With mrope enabled (default for Qwen3VL-class models), ``positions``
        has shape `(3, seq_len)`, otherwise `(seq_len,)`.
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits_local(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


########################################################
# Qwen3_5-MoE
########################################################


class Qwen3_5_MoeMixtureOfExperts(MixtureOfExperts):
    language_model: Qwen3_5MoeForCausalLM
    num_local_physical_experts: int

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.language_model.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.moe_layers = []
        example_moe = None
        for layer in self.language_model.model.layers:
            if isinstance(layer, Qwen3_5DecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError(
                "No Qwen3_5 layer found in the language_model.model.layers."
            )

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MoeForConditionalGeneration(
    Qwen3_5ForConditionalGeneration, Qwen3_5_MoeMixtureOfExperts
):
    # For MoE LoRA weights loading
    is_3d_moe_weight: ClassVar[bool] = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5MoeConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = cast(
            MultiModalConfig, vllm_config.model_config.multimodal_config
        )

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )
        self.video_pruning_rate = self.multimodal_config.video_pruning_rate
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        # attributes needed by EVS-related functions inherited from Qwen3-VL
        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                input_norm=build_mm_input_norm(self.model_config),
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5MoeForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )

        # set MoE hyperparameters
        self.set_moe_parameters()
