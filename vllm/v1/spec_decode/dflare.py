# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import VllmConfig
from vllm.v1.spec_decode.dflash import DFlashProposer


class DFlareProposer(DFlashProposer):
    """DFlare proposer inherits DFlashProposer infrastructure.

    The draft model forward is identical between DFlash and DFlare
    (query-only pass with pre-filled context KV cache).  DFlare's
    differences (per-layer fusion + heterogeneous KV projections) are
    handled inside the model class itself.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dflare"
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            runner=runner,
        )
