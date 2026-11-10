# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Step5 vision-language model on ROCm."""

from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.step3p7 import Step3p7ForConditionalGeneration

from .model import Step5ForCausalLM


class Step5ForConditionalGeneration(Step3p7ForConditionalGeneration):
    """Step5 VLM using the Step3p7 multimodal frontend and Step5 text model.

    Step5-specific weight names are mapped by the text model's own mapper."""

    packed_modules_mapping = Step5ForCausalLM.packed_modules_mapping

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="vit_large_projector",
            tower_model="vision_model",
        )
