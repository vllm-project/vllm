# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig

if TYPE_CHECKING:
    from vllm.config import ParallelConfig


class DSparkSpeculatorConfig(DFlashSpeculatorConfig):
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.dspark.speculator.DSparkSpeculator"
    )
    uses_separate_workspace_lane = True
    supports_pcp = True

    @property
    def num_lookahead_tokens(self) -> int:
        hf_config = self.spec_config.draft_model_config.hf_config
        if getattr(hf_config, "sample_from_anchor", True):
            # Anchor-sampling DSpark drafts a block of num_speculative_tokens
            # query tokens in which the anchor itself is the first prediction
            # position (no separate bonus query).
            return self.num_speculative_tokens
        return super().num_lookahead_tokens

    def pcp_unsupported_reason(self, parallel_config: "ParallelConfig") -> str | None:
        dcp_size = parallel_config.decode_context_parallel_size
        pcp_size = parallel_config.prefill_context_parallel_size
        if (
            dcp_size not in (1, pcp_size)
            and self.spec_config.draft_model_config.use_mla
        ):
            return (
                "MRV2 PCP DSpark requires DCP=1 or DCP=PCP; got "
                f"DCP={dcp_size}, PCP={pcp_size}."
            )
        return super().pcp_unsupported_reason(parallel_config)
