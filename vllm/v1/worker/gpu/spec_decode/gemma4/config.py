# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.worker.gpu.spec_decode.mtp.config import MTPSpeculatorConfig


class Gemma4MTPSpeculatorConfig(MTPSpeculatorConfig):
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.gemma4.speculator.Gemma4Speculator"
    )
    # Gemma-4 MTP scores tied embedding rows directly; no lm_head to share.
    supported_options = MTPSpeculatorConfig.supported_options - {
        "draft_lm_head_quantization"
    }
