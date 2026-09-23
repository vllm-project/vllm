# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DraftTokensHandler step identity (#54437 piece 2 / Link A)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
from vllm.v1.worker.worker_base import WorkerWrapperBase

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

K = 4


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _batch(req_ids: list[str], structured: bool) -> SimpleNamespace:
    return SimpleNamespace(req_ids=list(req_ids), has_structured_output_reqs=structured)


def _drafts(start: int, num_reqs: int, device: torch.device) -> torch.Tensor:
    return torch.arange(
        start, start + num_reqs * K, dtype=torch.int32, device=device
    ).reshape(num_reqs, K)


def test_consumed_snapshot_survives_later_batch():
    """Link A: a later execute_model must not erase an earlier step's drafts."""
    device = _device()
    handler = DraftTokensHandler(device, max_snapshots=4)

    drafts_a = _drafts(1, 2, device)
    drafts_b = _drafts(101, 2, device)

    handler.snapshot_consumed_drafts(
        step_id=1,
        input_batch=_batch(["gA", "pA"], True),
        draft_tokens=drafts_a,
    )
    got = handler.get_draft_tokens(step_id=1)
    assert got is not None
    assert got.req_ids == ["gA", "pA"]
    assert got.draft_token_ids[0] == [1, 2, 3, 4]

    handler.snapshot_consumed_drafts(
        step_id=2,
        input_batch=_batch(["gB", "pB"], True),
        draft_tokens=drafts_b,
    )
    got_b = handler.get_draft_tokens(step_id=2)
    assert got_b is not None
    assert got_b.req_ids == ["gB", "pB"]

    got_a = handler.get_draft_tokens(step_id=1)
    assert got_a is not None
    assert got_a.req_ids == ["gA", "pA"]
    assert got_a.draft_token_ids[0] == [1, 2, 3, 4]


def test_plain_batch_does_not_evict_structured_snapshot():
    device = _device()
    handler = DraftTokensHandler(device, max_snapshots=4)
    drafts_a = _drafts(1, 2, device)
    drafts_plain = _drafts(201, 2, device)

    handler.snapshot_consumed_drafts(
        step_id=1,
        input_batch=_batch(["gA", "pA"], True),
        draft_tokens=drafts_a,
    )
    handler.snapshot_consumed_drafts(
        step_id=2,
        input_batch=_batch(["p1", "p2"], False),
        draft_tokens=drafts_plain,
    )

    got = handler.get_draft_tokens(step_id=1)
    assert got is not None
    assert got.req_ids == ["gA", "pA"]
    assert handler.get_draft_tokens(step_id=2) is None


def test_unknown_step_returns_none_for_fail_closed_fallback():
    device = _device()
    handler = DraftTokensHandler(device)
    handler.snapshot_consumed_drafts(
        step_id=3,
        input_batch=_batch(["gA"], True),
        draft_tokens=_drafts(1, 1, device),
    )

    assert handler.get_draft_tokens(step_id=99) is None


def test_proposal_slot_without_step_id_stays_latest():
    """Non-async post_step still reads the latest proposed drafts."""
    device = _device()
    handler = DraftTokensHandler(device)
    handler.set_draft_tokens(_batch(["gA", "pA"], True), _drafts(1, 2, device))
    handler.set_draft_tokens(_batch(["gB", "pB"], True), _drafts(101, 2, device))

    got = handler.get_draft_tokens()
    assert got is not None
    assert got.req_ids == ["gB", "pB"]
    assert got.draft_token_ids[0] == [101, 102, 103, 104]


def test_proposal_plain_batch_returns_placeholders():
    device = _device()
    handler = DraftTokensHandler(device)
    handler.set_draft_tokens(_batch(["p1", "p2"], False), _drafts(1, 2, device))

    got = handler.get_draft_tokens()
    assert got is not None
    assert got.req_ids == ["p1", "p2"]
    assert got.draft_token_ids == [[-1] * K, [-1] * K]


def test_consumed_lookup_ignores_proposal_slot():
    device = _device()
    handler = DraftTokensHandler(device)
    handler.snapshot_consumed_drafts(
        step_id=5,
        input_batch=_batch(["gA"], True),
        draft_tokens=_drafts(1, 1, device),
    )
    handler.set_draft_tokens(_batch(["gB"], True), _drafts(101, 1, device))

    got = handler.get_draft_tokens(step_id=5)
    assert got is not None
    assert got.req_ids == ["gA"]
    assert got.draft_token_ids[0] == [1, 2, 3, 4]


def test_ring_buffer_evicts_oldest_and_misses():
    device = _device()
    handler = DraftTokensHandler(device, max_snapshots=2)
    handler.snapshot_consumed_drafts(
        step_id=1,
        input_batch=_batch(["a"], True),
        draft_tokens=_drafts(1, 1, device),
    )
    handler.snapshot_consumed_drafts(
        step_id=2,
        input_batch=_batch(["b"], True),
        draft_tokens=_drafts(11, 1, device),
    )
    handler.snapshot_consumed_drafts(
        step_id=3,
        input_batch=_batch(["c"], True),
        draft_tokens=_drafts(21, 1, device),
    )

    assert handler.get_draft_tokens(step_id=1) is None
    got = handler.get_draft_tokens(step_id=2)
    assert got is not None
    assert got.req_ids == ["b"]
    got = handler.get_draft_tokens(step_id=3)
    assert got is not None
    assert got.req_ids == ["c"]


def test_worker_wrapper_drops_unsupported_step_id():
    wrapper = WorkerWrapperBase(rpc_rank=0)
    seen: list[tuple] = []

    class PluginWorker:
        def take_draft_token_ids(self):
            seen.append(())
            return "legacy"

    wrapper.worker = PluginWorker()
    assert wrapper.take_draft_token_ids(step_id=7) == "legacy"
    assert seen == [()]


def test_worker_wrapper_forwards_step_id_when_supported():
    wrapper = WorkerWrapperBase(rpc_rank=0)
    seen: list[int | None] = []

    class Worker:
        def take_draft_token_ids(self, step_id: int | None = None):
            seen.append(step_id)
            return "ok"

    wrapper.worker = Worker()
    assert wrapper.take_draft_token_ids(step_id=7) == "ok"
    assert wrapper.take_draft_token_ids() == "ok"
    assert seen == [7, None]


def test_worker_wrapper_drops_positional_only_step_id():
    wrapper = WorkerWrapperBase(rpc_rank=0)
    seen: list[tuple] = []

    class PluginWorker:
        def take_draft_token_ids(self, step_id: int | None = None, /):
            seen.append((step_id,))
            return "legacy"

    wrapper.worker = PluginWorker()
    assert wrapper.take_draft_token_ids(step_id=7) == "legacy"
    assert seen == [(None,)]
