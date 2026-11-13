# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig


class DFlash2SpeculatorConfig(DFlashSpeculatorConfig):
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.dflash2.speculator.DFlash2Speculator"
    )
    # Draft-side watermarking doesn't cover candidate-head drafting yet.
    supports_watermarking = False
