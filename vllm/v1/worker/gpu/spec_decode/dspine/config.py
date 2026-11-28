# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.v1.worker.gpu.spec_decode.dflash2.config import DFlash2SpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.eagle.config import (
    check_draft_lm_head_quantization,
)

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


class DSpineSpeculatorConfig(DFlash2SpeculatorConfig):
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.dspine.speculator.DSpineSpeculator"
    )
    supported_options = DFlash2SpeculatorConfig.supported_options | frozenset(
        {"draft_lm_head_quantization"}
    )

    @classmethod
    def apply_options(cls, spec: "SpeculativeConfig") -> None:
        super().apply_options(spec)
        quantization = spec.draft_lm_head_quantization
        if quantization is None:
            return
        if quantization != "fp8":
            raise ValueError("DSpine candidate retrieval only supports fp8.")
        if spec.draft_parallel_config.tensor_parallel_size != 1:
            raise ValueError("DSpine FP8 candidate retrieval requires draft TP=1.")
        check_draft_lm_head_quantization(spec)
