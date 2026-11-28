# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config import SpeculativeConfig


def needs_mtp_hidden_buffer(spec_config: SpeculativeConfig | None) -> bool:
    """Whether the drafter reads the target's pre-hc_head residual buffer, which
    stands in for its final hidden states (aux-state drafters like DSpark don't).
    """
    return (
        spec_config is not None
        and spec_config.uses_target_hidden_states
        and spec_config.speculator_config.uses_target_hc_residual
        and not spec_config.uses_aux_hidden_states
    )
