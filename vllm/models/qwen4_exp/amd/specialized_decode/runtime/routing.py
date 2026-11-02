# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which decode steps the kernels take, from the step's attention metadata
alone (no kernel imports: the host tests run it without a GPU)."""

from typing import Any

import torch


def step_reason(attn_metadata, gdn_prefix: str, width: int) -> str | None:
    """None if a step's GDN metadata (identical on every TP rank) is a pure
    decode step of one token a request whose state indices cover ``width``
    rows, else why not."""
    if not isinstance(attn_metadata, dict):
        return "no metadata"
    m: Any = attn_metadata.get(gdn_prefix)
    if m is None or m.num_prefills != 0:
        return "not pure decode"
    if m.spec_sequence_masks is not None or m.num_spec_decodes != 0:
        return "spec-verify"
    if m.num_decodes < 1:
        return "no decode"
    idx = m.non_spec_state_indices_tensor
    # a graph's pad rows read NULL_BLOCK_ID entries: the indices cover every row
    if idx is None or idx.dtype != torch.int32:
        return "no int32 state indices"
    if idx.numel() < width or not idx.is_contiguous():
        return "state indices do not cover the rows"
    return None
