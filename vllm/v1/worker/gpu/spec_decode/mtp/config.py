# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.worker.gpu.spec_decode.eagle.config import EagleSpeculatorConfig


class MTPSpeculatorConfig(EagleSpeculatorConfig):
    speculator_class = "vllm.v1.worker.gpu.spec_decode.mtp.speculator.MTPSpeculator"
    target_embed_sharing = "target_opt_in"
    supports_pcp = True
