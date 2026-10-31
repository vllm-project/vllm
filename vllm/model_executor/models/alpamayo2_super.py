# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alpamayo 2 Super: the Qwen3-VL reasoning model of NVIDIA's driving policy.

The checkpoint stores a Qwen3-VL model under ``vlm.`` and a flow-matching
trajectory expert under ``expert.``. vLLM serves the Qwen3-VL part, which
produces every text output of the model: visual question answering,
grounding, chain-of-causation reasoning with meta-actions, and auto-labeling.
The trajectory itself needs the expert and is not produced here.
"""

from collections.abc import Mapping

import torch

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
    Qwen3VLProcessor,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import BaseProcessingInfo
from vllm.transformers_utils.configs.alpamayo2_super import Alpamayo2SuperConfig


class Alpamayo2SuperProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self) -> Alpamayo2SuperConfig:
        return self.ctx.get_hf_config(Alpamayo2SuperConfig)

    # Image-only: the reference implementation passes camera frames as images
    # and never uses video input.
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}

    def get_supported_mm_processor_kwargs(self) -> dict[str, set[str]]:
        # Skip Qwen3-VL's override, which adds video-only kwargs.
        return BaseProcessingInfo.get_supported_mm_processor_kwargs(self)

    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        processor = super().get_hf_processor(**kwargs)
        # The per-image pixel budget from the model config, which the reference
        # implementation applies for every task; the checkpoint's preprocessor
        # config does not carry it.
        config = self.get_hf_config()
        if config.min_pixels is not None and config.max_pixels is not None:
            processor.image_processor.size = {
                "shortest_edge": config.min_pixels,
                "longest_edge": config.max_pixels,
            }
        return processor


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Alpamayo2SuperProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Alpamayo2SuperForConditionalGeneration(Qwen3VLForConditionalGeneration):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "vlm.model.visual.": "visual.",
            "vlm.lm_head.": "language_model.lm_head.",
            "vlm.model.language_model.": "language_model.model.",
            # The trajectory expert is not served by vLLM.
            "expert.": None,
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        # Discrete trajectory tokens (<i0>, <i1>, ...) only ever appear in
        # prompts; like the reference implementation, never sample them.
        traj_ids = config.traj_ids
        self.traj_vocab_start = (
            min(traj_ids["history_id0"], traj_ids["future_id0"])
            if "history_id0" in traj_ids and "future_id0" in traj_ids
            else None
        )
        self.traj_vocab_size = config.traj_vocab_size

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is not None and self.traj_vocab_start is not None:
            end = self.traj_vocab_start + self.traj_vocab_size
            logits[..., self.traj_vocab_start : end] = float("-inf")
        return logits
