# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.worker.gpu.spec_decode.mtp.config import MTPSpeculatorConfig


class MultiModuleMTPSpeculatorConfig(MTPSpeculatorConfig):
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator."
        "MultiModuleMTPSpeculator"
    )
    supports_pcp = False

    @property
    def num_prefill_lookahead_tokens(self) -> int:
        # Each MTP module reads one token further ahead than the one before it,
        # so the chain needs num_speculative_tokens of runway at a
        # chunked-prefill boundary.
        return self.num_speculative_tokens
