# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.worker.gpu.spec_decode.dflash2.config import DFlash2SpeculatorConfig


class DSpineSpeculatorConfig(DFlash2SpeculatorConfig):
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.dspine.speculator.DSpineSpeculator"
    )
