# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShieldGemma 2 (google/shieldgemma-2-*) served natively as a classifier.

The original checkpoint is a Gemma3 VLM whose answer is the logits of the
"Yes" and "No" tokens at the last position. A two-way softmax over those two
logits equals sigmoid((lm_head[Yes] - lm_head[No]) . h), so the model is a
Gemma3 converted with `as_seq_cls_model` and the output is P(Yes) of that
two-way softmax. `ShieldGemma2ForImageClassificationConfig` selects the
conversion.
"""

from typing import TYPE_CHECKING

from transformers import ShieldGemma2Config

from vllm.multimodal import MULTIMODAL_REGISTRY

from .adapters import as_seq_cls_model
from .gemma3_mm import (
    Gemma3DummyInputsBuilder,
    Gemma3ForConditionalGeneration,
    Gemma3MultiModalProcessor,
    Gemma3ProcessingInfo,
)
from .interfaces_base import default_pooling_type
from .utils import WeightsMapper


class ShieldGemma2ProcessingInfo(Gemma3ProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(ShieldGemma2Config)


if TYPE_CHECKING:
    _ShieldGemma2Base = Gemma3ForConditionalGeneration
else:
    _ShieldGemma2Base = as_seq_cls_model(Gemma3ForConditionalGeneration)


@default_pooling_type(seq_pooling_type="LAST")
@MULTIMODAL_REGISTRY.register_processor(
    Gemma3MultiModalProcessor,
    info=ShieldGemma2ProcessingInfo,
    dummy_inputs=Gemma3DummyInputsBuilder,
)
class ShieldGemma2ForImageClassification(_ShieldGemma2Base):
    # The checkpoint nests the text model one level deeper than Gemma3's.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.model.": "language_model.model.",
            "model.language_model.lm_head.": "language_model.lm_head.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
        }
    )
