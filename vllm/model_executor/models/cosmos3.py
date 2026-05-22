# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import regex

from vllm.config import VllmConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.model_executor.models.utils import WeightsMapper


class Cosmos3ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    # Cosmos3 unified checkpoints store a Qwen3-VL understanding tower
    # alongside a generation tower in a flat key layout. This mapper drops
    # the generation tower weights and rewrites the understanding tower keys
    # into the nested form expected by Qwen3VLForConditionalGeneration.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_regex={
            regex.compile(
                r"^model\.(?!language_model\.)(.+)$"
            ): r"model.language_model.\1",
            regex.compile(
                r"^(blocks\.|merger\.|patch_embed\.|pos_embed\.|deepstack_merger_list\.)"
            ): r"model.visual.\1",
            regex.compile(r"^sound_modality_embed$"): None,
            regex.compile(r"^action_modality_embed$"): None,
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

    allow_patterns_overrides = ["transformer/*.safetensors"]

    """
    Cosmos3 checkpoint separates transformer weights and vision_encoder weights
    into separate directories, as it's in diffusers checkpoint format.
    Using secondary_weights here to load all necessary weights for
    the Reasoner-only part.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.secondary_weights = [
            DefaultModelLoader.Source(
                model_or_path=vllm_config.model_config.model,
                revision=vllm_config.model_config.revision,
                prefix="",
                allow_patterns_overrides=["vision_encoder/*.safetensors"],
            ),
        ]
