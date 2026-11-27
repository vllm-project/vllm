# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig


class DFlash2SpeculatorConfig(DFlashSpeculatorConfig):
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.dflash2.speculator.DFlash2Speculator"
    )
    # Draft-side watermarking doesn't cover candidate-head drafting yet.
    # Drafts are sampled by CandidateSampler, which draft_top_k does not cover.
    supported_options = DFlashSpeculatorConfig.supported_options - {"draft_top_k"}
    supports_watermarking = False
