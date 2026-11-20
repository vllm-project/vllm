# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the draft-logits NaN statistics probe.

The probe is pure torch, so these tests run on any device. They verify
that the statistics input set equals the verifier's read set:

* only active (slot, step) pairs of the persistent draft cache are read
  -- stale cells in inactive slots or unscheduled steps must not count;
* bonus rows (each request's last row, ``step == num_steps``) are zeroed;
* padded vocab columns beyond the effective ``vocab_size`` are excluded;
* greedy rows (``temperature == 0``) are excluded, mirroring the
  verifier's early return on draft reads;
* per-request aggregation follows the packed ``cu_num_logits`` layout.
"""

import torch

from vllm.v1.worker.gpu.metrics.logits import (
    counts_rows_to_requests,
    get_num_draft_nans,
)

V = 32  # cache width
K = 4  # num_steps
NUM_REQS = 2
NUM_LOGITS = NUM_REQS * (K + 1)  # K draft rows + 1 bonus row per request


def _uniform_mappings(num_logits=NUM_LOGITS):
    rows_per_req = K + 1
    slot = torch.tensor(
        [i // rows_per_req for i in range(num_logits)], dtype=torch.int32
    )
    step = torch.tensor(
        [i % rows_per_req for i in range(num_logits)], dtype=torch.int32
    )
    return slot, step


def _cu(rows_per_req=K + 1, num_reqs=NUM_REQS):
    return (torch.arange(num_reqs, dtype=torch.int32) * rows_per_req).tolist() + [
        num_reqs * rows_per_req
    ]


def _cache(num_slots=NUM_REQS):
    return torch.zeros(num_slots, K, V)


def test_active_nan_counted():
    """A NaN in a read (slot, step) cell is reported at its own row."""
    draft = _cache()
    draft[0, 1, 5] = float("nan")
    slot, step = _uniform_mappings()
    temp = torch.ones(NUM_LOGITS)
    counts = get_num_draft_nans(draft, slot, step, V, temp)
    assert counts.shape[0] == NUM_LOGITS
    assert counts[1].item() == 1  # row 1 = req 0, step 1
    assert counts.sum().item() == 1


def test_inactive_slot_not_reported():
    """A NaN in a cache slot no request maps to is stale data: the census
    stays silent instead of reporting it."""
    draft = _cache(num_slots=4)
    draft[3, 2, :5] = float("nan")  # slot 3 unused by the 2-request batch
    slot, step = _uniform_mappings()
    temp = torch.ones(NUM_LOGITS)
    counts = get_num_draft_nans(draft, slot, step, V, temp)
    assert counts.sum().item() == 0


def test_bonus_row_not_reported():
    """Bonus rows read no draft in the verifier: the clamped gather is
    discarded, so a poisoned last draft step is reported once at its own
    row, never as a bonus row."""
    draft = _cache()
    draft[0, K - 1, :2] = float("nan")  # req 0, last real draft step
    slot, step = _uniform_mappings()  # rows K and 2*K+1 carry step == K
    temp = torch.ones(NUM_LOGITS)
    counts = get_num_draft_nans(draft, slot, step, V, temp)
    assert counts[K - 1].item() == 2  # row 3 = req 0, step 3
    assert counts[K].item() == 0  # row 4 = req 0 bonus (clamped, masked)
    assert counts.sum().item() == 2


def test_padded_vocab_not_reported():
    """Columns beyond the effective vocab are never loaded by the
    verifier; NaNs there must not be counted (either side of the min)."""
    draft = _cache()
    draft[0, 0, 20] = float("nan")  # padded draft column
    draft[0, 0, 7] = float("nan")  # effective column
    slot, step = _uniform_mappings()
    temp = torch.ones(NUM_LOGITS)
    counts = get_num_draft_nans(draft, slot, step, 16, temp)
    assert counts[0].item() == 1
    assert counts.sum().item() == 1


def test_greedy_rows_not_reported():
    """The verifier skips draft reads for temperature == 0 requests and
    accepts via target argmax, so their draft rows are not inputs."""
    draft = _cache()
    draft[0, 0, :7] = float("nan")  # req 0 (greedy) -- skipped
    draft[1, 0, :2] = float("nan")  # req 1 (stochastic) -- real
    slot, step = _uniform_mappings()
    temp = torch.tensor([0.0] * (K + 1) + [1.0] * (K + 1))
    counts = get_num_draft_nans(draft, slot, step, V, temp)
    assert counts.sum().item() == 2
    assert counts[K + 1].item() == 2  # row 5 = req 1, step 0


def test_rows_to_requests_packed():
    """Per-row counts aggregate along the packed cu_num_logits layout,
    including requests scheduled fewer rows and zero-row requests."""
    counts = torch.tensor([1, 2, 3, 0, 4], dtype=torch.int32)
    cu = torch.tensor([0, 1, 5], dtype=torch.int32)  # req0: 1 row, req1: 4
    assert counts_rows_to_requests(counts, cu).tolist() == [1, 9]
    # a request with zero scheduled rows contributes zero
    cu0 = torch.tensor([0, 0, 5], dtype=torch.int32)
    assert counts_rows_to_requests(counts, cu0).tolist() == [0, 10]


def test_degenerate_shapes():
    """Zero rows and zero effective vocab must not crash."""
    slot = torch.zeros(0, dtype=torch.int32)
    step = torch.zeros(0, dtype=torch.int32)
    temp = torch.zeros(0)
    counts = get_num_draft_nans(_cache(), slot, step, V, temp)
    assert counts.shape[0] == 0
    draft = _cache()
    slot, step = _uniform_mappings()
    temp = torch.ones(NUM_LOGITS)
    counts = get_num_draft_nans(draft, slot, step, 0, temp)
    assert counts.sum().item() == 0
    assert counts_rows_to_requests(
        torch.zeros(0, dtype=torch.int32), torch.tensor([0, 0])
    ).tolist() == [0]
