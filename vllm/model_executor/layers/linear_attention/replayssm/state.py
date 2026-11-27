# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The per-layer ReplaySSM runtime state the model runner binds, shared by the
Mamba2 and Gated DeltaNet ReplaySSM decodes."""

from dataclasses import dataclass, field

import torch


@dataclass
class ReplaySSMLayerState:
    """Held by a ReplaySSM layer as `ssu_state`."""

    buffer_len: int
    # Per-slot ring position and committed token count, shared by a cache
    # group's layers; bound by `initialize_replayssm_ring_trackers`.
    ring_start: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int32)
    )
    prev_num_accepted: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int32)
    )

    def release(self) -> None:
        self.ring_start = torch.empty(0, dtype=torch.int32)
        self.prev_num_accepted = torch.empty(0, dtype=torch.int32)
