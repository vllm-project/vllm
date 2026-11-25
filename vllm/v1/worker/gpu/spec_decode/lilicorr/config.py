# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig


class LiLiCorrSpeculatorConfig(DFlashSpeculatorConfig):
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.lilicorr.speculator.LiLiCorrSpeculator"
    )
    # Draft-side watermarking doesn't cover candidate-head drafting yet.
    supports_watermarking = False
