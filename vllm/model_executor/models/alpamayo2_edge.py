# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alpamayo 2 Edge: the Cosmos3-Edge reasoning model of NVIDIA's driving policy.

The checkpoint stores a Cosmos3-Edge model under `vlm.` and a flow-matching
trajectory expert under `expert.`. vLLM serves the Cosmos3-Edge part, which
produces every text output of the model: visual question answering and
chain-of-causation reasoning with meta-actions. The trajectory itself needs
the expert and is not produced here.
"""

from collections.abc import Mapping

import regex
import torch

from vllm.config import VllmConfig
from vllm.model_executor.models.cosmos3_edge import (
    Cosmos3EdgeDummyInputsBuilder,
    Cosmos3EdgeForConditionalGeneration,
    Cosmos3EdgeMultiModalProcessor,
    Cosmos3EdgeProcessingInfo,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import BaseProcessingInfo
from vllm.transformers_utils.configs.alpamayo2_edge import Alpamayo2EdgeConfig


class Alpamayo2EdgeProcessingInfo(Cosmos3EdgeProcessingInfo):
    def get_hf_config(self) -> Alpamayo2EdgeConfig:
        return self.ctx.get_hf_config(Alpamayo2EdgeConfig)

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


@MULTIMODAL_REGISTRY.register_processor(
    Cosmos3EdgeMultiModalProcessor,
    info=Alpamayo2EdgeProcessingInfo,
    dummy_inputs=Cosmos3EdgeDummyInputsBuilder,
)
class Alpamayo2EdgeForConditionalGeneration(Cosmos3EdgeForConditionalGeneration):
    # The checkpoint stores the language model in the Transformers layout.
    # Rewrite it to the Diffusers layout, which the Cosmos3-Edge mapping splits
    # into Nemotron-H attention and MLP layers. Prefixes apply in order, so the
    # remaining `vlm.` names reach the Cosmos3-Edge mapping unprefixed.
    hf_to_vllm_mapper = (
        WeightsMapper(
            orig_to_new_regex={
                regex.compile(
                    r"^vlm\.model\.language_model\.(layers\.\d+\.mlp\.)fc1\."
                ): r"\1up_proj.",
                regex.compile(
                    r"^vlm\.model\.language_model\.(layers\.\d+\.mlp\.)fc2\."
                ): r"\1down_proj.",
                regex.compile(r"^vlm\.model\.language_model\."): "",
            },
            orig_to_new_prefix={
                "vlm.": "",
                # The trajectory expert is not served by vLLM.
                "expert.": None,
            },
        )
        | Cosmos3EdgeForConditionalGeneration.hf_to_vllm_mapper
    )

    # The weights are top-level safetensors, not Diffusers subfolders.
    allow_patterns_overrides = None  # type: ignore[assignment]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
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
