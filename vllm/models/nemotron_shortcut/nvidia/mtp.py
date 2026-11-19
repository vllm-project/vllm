# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Inference-only Nemotron Shortcut multi-token-prediction (MTP) draft head.

The head is a DeepSeek-style shell -- `enorm`/`hnorm`/`eh_proj`/`final_layernorm`
-- around one Nemotron Shortcut shortcut-MoE attention block, or, with
`moe_shortcut_disable_mtp`, attention then MoE. The block runs outside the
wide residual streams: the checkpoint carries no stream read/write logits, and
`eh_proj` is `[hidden, 2 * hidden]`, so the head works on a single hidden state.
`mtp_use_repeated_layer` means one physical block serves every predict step.
"""

from collections.abc import Iterable

import torch
from torch import nn
from transformers import PreTrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.config.parallel import ParallelConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    LocalArgmaxMixin,
    MultiModalEmbeddings,
    SupportsMultiModalEmbeddings,
    SupportsPP,
    SupportsQuant,
    _require_is_multimodal,
)
from vllm.model_executor.models.utils import (
    WeightsMapper,
    _merge_multimodal_embeddings,
    get_draft_quant_config,
    maybe_prefix,
)

from .model import (
    NemotronShortcutMoEAttentionLayer,
    ShortcutMoEAttentionLayer,
    load_nemotron_shortcut_weights,
)
from .sdsa import SdsaBuffers, allocate_sdsa_buffers


class NemotronShortcutMTPLayer(nn.Module):
    """Fuse the next-token embedding with the target hidden state, then refine."""

    def __init__(
        self,
        config: PreTrainedConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
        sdsa_buffers: SdsaBuffers | None = None,
    ) -> None:
        super().__init__()
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ColumnParallelLinear(
            input_size=config.hidden_size * 2,
            output_size=config.hidden_size,
            bias=False,
            gather_output=True,
            quant_config=quant_config,
            prefix=f"{prefix}.eh_proj",
        )
        # Outside the residual streams: no learned stream ports in the checkpoint.
        self.block: nn.Module
        if getattr(config, "moe_shortcut_disable_mtp", False):
            self.block = NemotronShortcutMoEAttentionLayer(
                config,
                layer_idx,
                model_config,
                cache_config,
                quant_config,
                parallel_config,
                prefix=f"{prefix}.block",
                sdsa_buffers=sdsa_buffers,
            )
        else:
            self.block = ShortcutMoEAttentionLayer(
                config,
                layer_idx,
                model_config,
                cache_config,
                quant_config,
                parallel_config,
                prefix=f"{prefix}.block",
                wide_residual=False,
                sdsa_buffers=sdsa_buffers,
            )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, inputs_embeds: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        fused = torch.cat(
            [self.enorm(inputs_embeds), self.hnorm(hidden_states)], dim=-1
        )
        hidden_states, _ = self.eh_proj(fused)
        hidden_states = self.block(hidden_states)
        return self.final_layernorm(hidden_states)


class NemotronShortcutMultiTokenPredictor(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        draft_model_config = speculative_config.draft_model_config
        assert draft_model_config is not None
        config = draft_model_config.hf_config.get_text_config()
        self.config = config

        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)
        # One physical block reused for every predict step.
        self.repeated_layer = bool(getattr(config, "mtp_use_repeated_layer", False))
        num_blocks = 1 if self.repeated_layer else self.num_mtp_layers

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        # The selection buffers of the draft's SDSA layer.
        self.sdsa_buffers = None
        if config.index_topk is not None:
            self.sdsa_buffers = allocate_sdsa_buffers(vllm_config, config.index_topk)
        self.layers = nn.ModuleDict(
            {
                str(i): NemotronShortcutMTPLayer(
                    config,
                    config.num_hidden_layers + i,
                    vllm_config.model_config,
                    vllm_config.cache_config,
                    quant_config,
                    vllm_config.parallel_config,
                    prefix=f"{prefix}.layers.{i}",
                    sdsa_buffers=self.sdsa_buffers,
                )
                for i in range(num_blocks)
            }
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.get_input_embeddings(input_ids)
        idx = 0 if self.repeated_layer else min(spec_step_idx, len(self.layers) - 1)
        return self.layers[str(idx)](inputs_embeds, hidden_states)


@support_torch_compile
class NemotronShortcutMTP(
    LocalArgmaxMixin,
    nn.Module,
    SupportsPP,
    SupportsQuant,
    SupportsMultiModalEmbeddings,
):
    """The draft head is trained on the target's input stream, so on image and
    video prompts the placeholder positions carry the target's vision
    embeddings, not the text embedding of the placeholder token."""

    # The draft head consumes only the `mtp.*` subtree plus the shared embedding
    # and head; everything else in the checkpoint belongs to the target.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.mtp.": "model.",
            "model.language_model.embed_tokens.": "model.embed_tokens.",
            "model.language_model.": None,
            "model.vision_tower.": None,
            "model.multi_modal_projector.": None,
            "model.video_patch_conv.": None,
        },
        orig_to_new_stacked={
            ".self_attn.q_proj": (".self_attn.qkv_proj", "q"),
            ".self_attn.k_proj": (".self_attn.qkv_proj", "k"),
            ".self_attn.v_proj": (".self_attn.qkv_proj", "v"),
        },
    )

    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        draft_model_config = speculative_config.draft_model_config
        assert draft_model_config is not None
        config = draft_model_config.hf_config.get_text_config()
        self.config = config
        self.vllm_config = vllm_config
        # The draft head attends in the target's KV cache group, with the
        # target's SDSA config. An hf_overrides of the target's config does not
        # reach the draft's config, so take the target's SDSA fields.
        target_config = vllm_config.model_config.hf_text_config
        config.index_topk = target_config.index_topk
        config.sdsa_indexer_dtype = getattr(target_config, "sdsa_indexer_dtype", "auto")
        config.sdsa_decode_core = getattr(target_config, "sdsa_decode_core", "auto")
        config.sdsa_decode_fused = getattr(target_config, "sdsa_decode_fused", "auto")
        config.sdsa_prefill_core = getattr(target_config, "sdsa_prefill_core", "auto")
        config.sdsa_prefill_fused = getattr(target_config, "sdsa_prefill_fused", "auto")

        self.model = NemotronShortcutMultiTokenPredictor(
            vllm_config=vllm_config,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    @staticmethod
    def _find_quant_config(*args, **kwargs) -> QuantizationConfig | None:
        # The draft head takes the draft model's quantization, as NemotronHMTP.
        vllm_config = kwargs.get("vllm_config")
        assert isinstance(vllm_config, VllmConfig)
        return get_draft_quant_config(vllm_config)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        return _merge_multimodal_embeddings(
            inputs_embeds,
            multimodal_embeddings,
            _require_is_multimodal(is_multimodal),
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        **kwargs: object,
    ) -> torch.Tensor:
        assert hidden_states is not None
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return load_nemotron_shortcut_weights(self, weights, self.hf_to_vllm_mapper)
