# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Step5 vision-language model."""

from vllm.model_executor.models.step3p7 import Step3p7ForConditionalGeneration


class Step5ForConditionalGeneration(Step3p7ForConditionalGeneration):
    """Step5 VLM using the Step3p7 multimodal frontend and Step5 text model."""

    _enable_weights_track_by_default = True
