# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.eagle.config import (
    check_draft_lm_head_quantization,
)

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


class LiLiCorrSpeculatorConfig(DFlashSpeculatorConfig):
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.lilicorr.speculator.LiLiCorrSpeculator"
    )
    supported_options = frozenset({"draft_lm_head_quantization"})
    # Draft-side watermarking doesn't cover candidate-head drafting yet.
    supports_watermarking = False

    @classmethod
    def apply_options(cls, spec: "SpeculativeConfig") -> None:
        check_draft_lm_head_quantization(spec)
