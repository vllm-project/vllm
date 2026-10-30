# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x Omni for AMD ROCm GPUs: the shared Omni towers over the ROCm stack."""

from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.models.muse_spark.common.omni import (
    OMNI_WEIGHTS_MAPPER,
    MuseSpark1xOmniMixin,
    register_omni_processor,
)

from .model import MuseSpark1xForCausalLM


@register_omni_processor
class MuseSparkForConditionalGeneration(
    MuseSpark1xOmniMixin, MuseSpark1xForCausalLM, SupportsMultiModal
):
    """MuseSpark1x Omni: ROCm text stack plus NaViT vision and waveform audio."""

    hf_to_vllm_mapper = MuseSpark1xForCausalLM.hf_to_vllm_mapper | OMNI_WEIGHTS_MAPPER


__all__ = ["MuseSparkForConditionalGeneration"]
