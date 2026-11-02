# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Step5 vision-language model."""

from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.step3p7 import Step3p7ForConditionalGeneration

from .model import STEP5_PACKED_MODULES_MAPPING, STEP5_SHARE_EXPERT_MAPPER


class Step5ForConditionalGeneration(Step3p7ForConditionalGeneration, SupportsLoRA):
    """Step5 VLM using the Step3p7 multimodal frontend and Step5 text model."""

    packed_modules_mapping = STEP5_PACKED_MODULES_MAPPING
    hf_to_vllm_mapper = (
        Step3p7ForConditionalGeneration.hf_to_vllm_mapper | STEP5_SHARE_EXPERT_MAPPER
    )

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="vit_large_projector",
            tower_model="vision_model",
        )
