# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3-ASR ForcedAligner model (token classification)."""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_classify
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRDummyInputsBuilder,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRMultiModalProcessor,
    Qwen3ASRProcessingInfo,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors


@default_pooling_type(tok_pooling_type="ALL")
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3ASRMultiModalProcessor,
    info=Qwen3ASRProcessingInfo,
    dummy_inputs=Qwen3ASRDummyInputsBuilder,
)
class Qwen3ASRForcedAlignerForTokenClassification(
    Qwen3ASRForConditionalGeneration,
):
    """Qwen3-ASR Forced Aligner model for per-token timestamp classification.

    This model shares the audio tower and language model backbone with
    Qwen3-ASR, but replaces the LM head with a classification head that
    predicts time bins at ``<timestamp>`` token positions.

    Usage::

        llm = LLM(
            model="Qwen/Qwen3-ForcedAligner-0.6B",
            runner="pooling",
            hf_overrides={
                "architectures": ["Qwen3ASRForcedAlignerForTokenClassification"]
            },
        )
        outputs = llm.encode(
            [{"prompt": prompt, "multi_modal_data": {"audio": audio}}],
            pooling_task="token_classify",
        )
    """

    is_pooling_model = True

    # Map thinker.lm_head -> classifier (not language_model.lm_head)
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "thinker.lm_head.": "classifier.",
            "thinker.model.": "language_model.model.",
            "thinker.": "",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config
        thinker_config = config.thinker_config

        # Remove the unused generation head created by the base class;
        # the forced aligner uses a classifier head instead.
        self.language_model.lm_head = None
        self.language_model.logits_processor = None

        self.classify_num = thinker_config.classify_num

        # Classification head replaces lm_head for time-bin prediction.
        # Use model dtype (not head_dtype which defaults to float32 for
        # pooling models) to match the hidden state dtype.
        self.classifier = nn.Linear(
            thinker_config.text_config.hidden_size,
            self.classify_num,
            bias=False,
            dtype=vllm_config.model_config.dtype,
        )

        # Token-level pooler to split per-token logits per request
        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = pooler_for_token_classify(pooler_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # Run through language model backbone (transformer layers only)
        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        # Apply classification head -> [num_tokens, classify_num]
        return self.classifier(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["talker.", "code2wav."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
