# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GDN (gated delta net) decode core of the Qwen4Exp specialized decode.

`gdn_decode_core` replaces the stock core of a GDN layer in a decode step
with MTP verification rows (the output's zero fill, `causal_conv1d_update`
and `fused_gdn_decode_post_conv_mtp`) with one CuTe DSL kernel that reads
and writes vLLM's convolution and recurrent caches in place, in vLLM's
speculative-decode state protocol or in the record/commit protocol of
`commit.py`, whose `gdn_state_commit` folds the accepted rows into the
recurrent state after sampling. `warm_up_gdn_decode_core` compiles and
loads the core before any CUDA graph capture.
"""

from .commit import (
    CommitSchedule,
    RecordLayout,
    commit_tables,
    compiled_commit,
    default_commit_schedule,
    gdn_state_commit,
)
from .core import (
    Schedule,
    default_schedule,
    gdn_decode_core,
    warm_up_gdn_decode_core,
)

__all__ = [
    "CommitSchedule",
    "RecordLayout",
    "Schedule",
    "commit_tables",
    "compiled_commit",
    "default_commit_schedule",
    "default_schedule",
    "gdn_decode_core",
    "gdn_state_commit",
    "warm_up_gdn_decode_core",
]
