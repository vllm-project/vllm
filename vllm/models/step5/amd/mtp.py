# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 MTP on ROCm."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.mtp_validation import (
    is_mtp_completeness_check_enabled,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    _require_is_multimodal,
)
from vllm.model_executor.models.utils import (
    _merge_multimodal_embeddings,
    get_draft_quant_config,
    get_spec_layer_idx_from_weight_name,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .model import (
    Step5DecoderLayer,
    Step5ForCausalLM,
    load_step5_weights,
    use_fp32_norm_weights,
)

# Per-MTP-layer weights; everything else in the layer is its decoder block.
_MTP_LAYER_WEIGHTS = ("embed_tokens", "enorm", "hnorm", "eh_proj", "shared_head")


class Step5MTPSharedHead(nn.Module):
    def __init__(self, config, quant_config, prefix: str) -> None:
        super().__init__()
        self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.head",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class Step5MultiTokenPredictorLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.enorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = Step5MTPSharedHead(
            config, vllm_config.quant_config, prefix=f"{prefix}.shared_head"
        )
        self.mtp_block = Step5DecoderLayer(vllm_config, prefix=f"{prefix}.mtp_block")

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.eh_proj(
            torch.cat(
                [self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)],
                dim=-1,
            )
        )
        return self.mtp_block(positions, hidden_states.float()).to(hidden_states.dtype)


class Step5MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = nn.ModuleDict(
            {
                str(idx): Step5MultiTokenPredictorLayer(
                    vllm_config, f"{prefix}.layers.{idx}"
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        use_fp32_norm_weights(self, config)

    def _layer(self, spec_step_idx: int) -> Step5MultiTokenPredictorLayer:
        idx = self.mtp_start_layer_idx + spec_step_idx % self.num_mtp_layers
        return self.layers[str(idx)]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        return self._layer(spec_step_idx)(
            positions, previous_hidden_states, inputs_embeds
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        head = self._layer(spec_step_idx).shared_head
        return self.logits_processor(head.head, head(hidden_states))

    def get_top_tokens(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        head = self._layer(spec_step_idx).shared_head
        return self.logits_processor.get_top_tokens(head.head, head(hidden_states))


class Step5MTP(nn.Module, SupportsMultiModal):
    # Each MTP layer's shared_head carries its own LM head weights.
    has_own_lm_head = True
    supports_multimodal_embeddings = True
    packed_modules_mapping = Step5ForCausalLM.packed_modules_mapping

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        draft_quant_config = get_draft_quant_config(vllm_config)
        vllm_config = vllm_config.with_hf_config(
            vllm_config.speculative_config.draft_model_config.hf_config
        )
        vllm_config.quant_config = draft_quant_config
        self.config = vllm_config.model_config.hf_config
        self.model = Step5MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids, self.model.embed_tokens, is_multimodal=is_multimodal
        )
        if not multimodal_embeddings:
            return inputs_embeds
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=_require_is_multimodal(is_multimodal),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def get_top_tokens(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        return self.model.get_top_tokens(hidden_states, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def mtp_weights():
            for name, weight in weights:
                if name.startswith("model.embed_tokens."):
                    yield "model." + name.split(".", 1)[1], weight
                    continue
                layer = get_spec_layer_idx_from_weight_name(self.config, name)
                if layer is None:
                    continue
                name = name.replace(".transformer.", ".").replace(
                    "shared_head.output", "shared_head.head"
                )
                if not any(w in name for w in _MTP_LAYER_WEIGHTS):
                    name = name.replace(
                        f"model.layers.{layer}.", f"model.layers.{layer}.mtp_block."
                    )
                yield name, weight

        loaded = {
            f"model.{name}"
            for name in load_step5_weights(
                self.model,
                (
                    (name.removeprefix("model."), weight)
                    for name, weight in mtp_weights()
                ),
            )
        }
        if is_mtp_completeness_check_enabled():
            missing = {
                name
                for name, _ in self.named_parameters()
                if not name.endswith(
                    (".k_scale", ".v_scale", ".q_scale", ".prob_scale")
                )
            } - loaded
            if missing:
                raise RuntimeError(
                    f"Step5 MTP checkpoint is missing parameters: {sorted(missing)}"
                )
        return loaded
