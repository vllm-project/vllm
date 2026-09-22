# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch
from transformers import AutoProcessor

from vllm.multimodal import MULTIMODAL_REGISTRY

from .qwen3_5 import Qwen3_5MoeForConditionalGeneration
from .qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from .utils import AutoWeightsLoader


class InternS2PreviewProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object) -> AutoProcessor:
        return self.ctx.get_hf_processor(**kwargs)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=InternS2PreviewProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class InternS2PreviewForConditionalGeneration(Qwen3_5MoeForConditionalGeneration):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp.", "model.time_series.", "time_series."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
