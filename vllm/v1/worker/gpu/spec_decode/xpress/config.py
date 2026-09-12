# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig


class XPressSpeculatorConfig(DFlashSpeculatorConfig):
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.xpress.speculator.XPressSpeculator"
    )
