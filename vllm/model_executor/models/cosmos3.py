# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.model_executor.models.utils import WeightsMapper


class Cosmos3ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    # Cosmos3 unified checkpoints store a Qwen3-VL understanding tower
    # alongside a generation tower in a flat key layout. This mapper drops
    # the generation tower weights and rewrites the understanding tower keys
    # into the nested form expected by Qwen3VLForConditionalGeneration.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_regex={
            re.compile(
                r"^model\.(?!language_model\.)(.+)$"
            ): r"model.language_model.\1",
            re.compile(
                r"^(blocks\.|merger\.|patch_embed\.|pos_embed\.|deepstack_merger_list\.)"
            ): r"model.visual.\1",
            re.compile(r"^sound_modality_embed$"): None,
            re.compile(r"^action_modality_embed$"): None,
        },
        orig_to_new_substr={
            "_moe_gen": None,
        },
        orig_to_new_prefix={
            "llm2vae.": None,
            "vae2llm.": None,
            "time_embedder.": None,
            "llm2sound.": None,
            "sound2llm.": None,
            "llm2action.": None,
            "action2llm.": None,
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        overrides = getattr(
            vllm_config.model_config.hf_config, "allow_patterns_overrides", None
        )
        if overrides:
            self.allow_patterns_overrides = list(overrides)
