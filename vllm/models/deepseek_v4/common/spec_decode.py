# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config import SpeculativeConfig


def needs_mtp_hidden_buffer(spec_config: SpeculativeConfig | None) -> bool:
    """Whether the drafter reads the target's pre-hc_head residual buffer.

    DSpark drafts from the target's aux hidden states, so it skips the
    (max_num_batched_tokens, hc_mult * hidden_size) buffer.
    """
    return (
        spec_config is not None
        and (spec_config.use_eagle() or spec_config.uses_draft_model())
        and not spec_config.use_dspark()
    )
