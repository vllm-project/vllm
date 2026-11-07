# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.multimodal.audio_alignment import AudioAlignmentStore

pytestmark = pytest.mark.cpu_test


def test_capture_owns_only_its_valid_frames():
    """Retaining a short item must not retain its padded encoder batch."""
    store = AudioAlignmentStore()
    store.register("request", ["audio"])
    batch = torch.arange(48, dtype=torch.float32).view(2, 3, 8)
    expected = batch[:1, :, :2].clone()
    assert store.put("audio", {"asr_encoded": batch[:1, :, :2]}, 0.16)
    batch.zero_()
    capture = store.get("request")
    assert capture is not None
    actual = capture["tensors"]["asr_encoded"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()
    assert actual.untyped_storage().nbytes() == store.stats()["bytes"] == 24
    assert capture["duration_s"] == 0.16


@pytest.mark.parametrize("evict", [False, True])
def test_shared_capture_outlives_request_and_encoder_references(evict):
    store = AudioAlignmentStore()
    for handle in ("first", "second"):
        store.register(handle, ["audio"])
    assert store.put("audio", {"states": torch.ones(1)}, 0.08)
    if evict:
        store.evict("audio")
    store.release(["first"])
    assert store.get("first") is None
    assert store.get("second") is not None
    store.release(["second"])
    assert store.stats()["captures"] == int(not evict)
    if not evict:
        store.register("later", ["audio"])
        assert not store.needs_capture("audio")
        assert store.get("later") is not None
        store.release(["later"])
        store.evict("audio")
    assert store.stats()["captures"] == store.stats()["handles"] == 0
    assert store.stats()["bytes"] == 0


@pytest.mark.parametrize("recache", [False, True])
def test_preemption_reregisters_capture_and_restores_residency(recache):
    store = AudioAlignmentStore()
    store.register("request", ["audio"])
    assert store.put("audio", {"states": torch.ones(1)}, 0.08)
    store.evict("audio")
    store.register("request", ["audio"])
    assert store.get("request") is not None
    assert store.stats()["handles"] == 1
    with pytest.raises(ValueError, match="already identifies other audio"):
        store.register("request", ["different"])
    if recache:
        store.register("reencoded", ["audio"])
        assert not store.needs_capture("audio")
        assert store.get("reencoded") is not None
        store.mark_cached("audio")
        store.release(["reencoded"])
    store.release(["request", "request"])
    assert store.stats()["captures"] == int(recache)
    if recache:
        store.register("next", ["audio"])
        assert not store.needs_capture("audio")
        assert store.get("next") is not None
        store.release(["next"])
        store.evict("audio")
    assert store.stats()["handles"] == store.stats()["bytes"] == 0


def test_capture_and_workspace_share_budget_and_recover_after_failure():
    store = AudioAlignmentStore(max_bytes=32)
    store.register("first", ["first-audio"])
    assert store.put("first-audio", {"states": torch.ones(4)}, 0.32)
    store.register("second", ["second-audio"])
    with store.reserve_workspace(12):
        assert not store.put("second-audio", {"states": torch.ones(2)}, 0.16)
        assert store.get("first") is not None
        assert store.get("second") is None
        assert store.error("second") == "capture_budget_exceeded"
        assert store.stats()["bytes"] == 16
        with (
            pytest.raises(ValueError, match="exceed the budget"),
            store.reserve_workspace(8),
        ):
            pass
        assert store.stats()["workspace_bytes"] == 12
    assert store.stats()["workspace_bytes"] == 0
    with (
        pytest.raises(RuntimeError, match="alignment failed"),
        store.reserve_workspace(16),
    ):
        raise RuntimeError("alignment failed")
    assert store.stats()["workspace_bytes"] == 0
    store.release(["first", "second"])
    for audio in ("first-audio", "second-audio"):
        store.evict(audio)
    store.register("retry", ["second-audio"])
    assert store.needs_capture("second-audio")
    assert store.put("second-audio", {"states": torch.ones(2)}, 0.16)
    assert store.get("retry") is not None


def test_reset_invalidates_captures_and_request_handles():
    store = AudioAlignmentStore()
    store.register("request", ["audio"])
    assert store.put("audio", {"states": torch.ones(1)}, 0.08)
    store.reset()
    assert store.get("request") is None
    assert store.stats()["captures"] == store.stats()["handles"] == 0
    assert store.stats()["bytes"] == 0
    store.register("new-request", ["audio"])
    assert store.needs_capture("audio")
