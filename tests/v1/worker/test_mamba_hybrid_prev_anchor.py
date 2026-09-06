# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for the V2 model-runner all-mode write-anchor tracking kernel
(``_track_mamba_all_write_anchor_kernel``).

The kernel must mirror the V1 ``postprocess_mamba_all`` /
``preprocess_mamba_all_specdec`` pair in a single pass:
  * stage the previous step's per-request anchors into batch order
    (``staged[row] = tracking[idx_mapping[row]]``), then
  * record this step's anchor ``max(0, (seq_len - 1) // block)`` for rows
    scheduling exactly ``1 + num_spec_tokens`` tokens (a full decode window),
    and untrack (-1) every other scheduled row.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(
        reason="Triton anchor-tracking kernel requires CUDA", allow_module_level=True
    )

from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    _track_mamba_all_write_anchor_kernel,
)

BLOCK = 8
NUM_SPEC = 3
FULL = 1 + NUM_SPEC


def _step(idx_mapping, query_lens, seq_lens, tracking):
    num_reqs = len(query_lens)
    idx = torch.tensor(idx_mapping, dtype=torch.int32, device="cuda")
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32, device="cuda")
    qsl[1:] = torch.cumsum(
        torch.tensor(query_lens, dtype=torch.int32, device="cuda"), 0
    )
    sl = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    staged = torch.full((num_reqs,), -7, dtype=torch.int32, device="cuda")
    _track_mamba_all_write_anchor_kernel[(num_reqs,)](
        idx,
        qsl,
        sl,
        tracking,
        staged,
        FULL_DECODE_LEN=FULL,
        MAMBA_BLOCK_SIZE=BLOCK,
    )
    return staged


def _reference(idx_mapping, query_lens, seq_lens, tracking_before):
    """V1 semantics: stage prev, then track full-decode rows / untrack others."""
    staged = [tracking_before[slot] for slot in idx_mapping]
    tracking_after = list(tracking_before)
    for row, slot in enumerate(idx_mapping):
        if query_lens[row] == FULL:
            tracking_after[slot] = max(0, (seq_lens[row] - 1) // BLOCK)
        else:
            tracking_after[slot] = -1
    return staged, tracking_after


def test_stage_then_track():
    tracking = torch.full((6,), -1, dtype=torch.int32, device="cuda")
    tracking[3] = 7  # slot 3 tracked by an earlier step
    idx_mapping = [3, 0, 5]
    query_lens = [FULL, 17, FULL]  # decode window, prefill chunk, decode window
    seq_lens = [20, 33, 8]

    exp_staged, exp_after = _reference(
        idx_mapping, query_lens, seq_lens, tracking.tolist()
    )
    staged = _step(idx_mapping, query_lens, seq_lens, tracking)

    assert staged.tolist() == exp_staged  # [7, -1, -1]
    assert tracking.tolist() == exp_after  # slots 3 -> 2, 0 -> -1, 5 -> 0
    # rows not in the batch keep their value
    assert tracking[1].item() == -1 and tracking[2].item() == -1


def test_two_step_chain_and_untrack():
    tracking = torch.full((4,), -1, dtype=torch.int32, device="cuda")

    # step 1: both rows run full decode windows
    idx_mapping = [2, 1]
    seq_lens = [17, 9]
    staged1 = _step(idx_mapping, [FULL, FULL], seq_lens, tracking)
    assert staged1.tolist() == [-1, -1]  # nothing tracked yet
    assert tracking.tolist() == [-1, (9 - 1) // BLOCK, (17 - 1) // BLOCK, -1]

    # step 2: row order changes; slot 1 falls back to a partial (chunked) step
    staged2 = _step([1, 2], [3, FULL], [12, 21], tracking)
    # staged must serve step 1's anchors in the new batch order
    assert staged2.tolist() == [(9 - 1) // BLOCK, (17 - 1) // BLOCK]
    # slot 1 untracked (partial), slot 2 re-tracked at the new position
    assert tracking.tolist() == [-1, -1, (21 - 1) // BLOCK, -1]
