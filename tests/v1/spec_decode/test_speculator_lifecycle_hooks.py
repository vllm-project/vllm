# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifecycle-hook contract for AutoRegressiveSpeculator subclasses.

``skip_topk`` (DSA's top-k reuse) is persistent Python state, so a draft step
that dies partway through could leave reuse mode on and every later request would
silently read a stale top-k. ``on_prefill_begin`` resetting unconditionally is
what prevents that: every request starts with a prefill.
"""

from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator


def test_base_hooks_are_noops():
    """The base class must not require subclasses to override the hooks."""
    for name in (
        "on_prefill_begin",
        "on_prefill_end",
        "on_multi_step_decode_begin",
        "on_multi_step_decode_end",
    ):
        hook = getattr(AutoRegressiveSpeculator, name)
        assert hook(None, 1) is None


class _FakeInnerModel:
    def __init__(self):
        self.skip_topk = False
        self.compacted: list[int] = []

    def set_skip_topk(self, skip: bool) -> None:
        self.skip_topk = skip

    def compact_topk_indices(self, slot_ids) -> None:
        self.compacted.append(len(slot_ids))


class _FakeDraftModel:
    def __init__(self):
        self.model = _FakeInnerModel()


class _StubMTPSpeculator(MTPSpeculator):
    """MTPSpeculator with the hooks live but no engine behind them."""

    def __init__(self, *, share: bool, num_speculative_steps: int = 5):
        self.share_mtp_topk_indices = share
        self.num_speculative_steps = num_speculative_steps
        self.model = _FakeDraftModel()
        self.last_token_indices = list(range(8))


def test_share_disabled_by_default():
    """Without load_draft_model the flag must stay off (no set_skip_topk call)."""
    assert MTPSpeculator.share_mtp_topk_indices is False


def test_skip_topk_is_reset_at_prefill_begin():
    """A leaked skip_topk=True from a failed step is cleared by the next prefill."""
    spec = _StubMTPSpeculator(share=True)
    spec.model.model.skip_topk = True  # simulate the leak

    spec.on_prefill_begin(4)

    assert spec.model.model.skip_topk is False


def test_skip_topk_toggles_around_multi_step_decode():
    spec = _StubMTPSpeculator(share=True)

    spec.on_multi_step_decode_begin(4)
    assert spec.model.model.skip_topk is True

    spec.on_multi_step_decode_end(4)
    assert spec.model.model.skip_topk is False


def test_hooks_are_inert_when_sharing_is_off():
    spec = _StubMTPSpeculator(share=False)

    spec.on_prefill_begin(4)
    spec.on_prefill_end(4)
    spec.on_multi_step_decode_begin(4)
    spec.on_multi_step_decode_end(4)

    assert spec.model.model.skip_topk is False
    assert spec.model.model.compacted == []


def test_compaction_only_for_multi_step_drafts():
    """A single draft step needs no compaction: there are no steps 1+ to feed."""
    single = _StubMTPSpeculator(share=True, num_speculative_steps=1)
    single.on_prefill_end(4)
    assert single.model.model.compacted == []

    multi = _StubMTPSpeculator(share=True, num_speculative_steps=5)
    multi.on_prefill_end(4)
    assert multi.model.model.compacted == [4]
