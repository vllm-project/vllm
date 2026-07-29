# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Whisper word-timestamp helpers.

These cover the CPU/GPU-agnostic pieces of the cross-attention + DTW pipeline:
the capture scatter, the attention-weight post-processing and the DTW itself.
The DTW and median filter are checked against the transformers reference
implementations they are derived from, so a divergence shows up as a test
failure rather than as drifting timestamps.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers.models.whisper.generation_whisper import (
    _dynamic_time_warping as hf_dtw,
)
from transformers.models.whisper.generation_whisper import _median_filter as hf_median

from vllm.model_executor.models.whisper import (
    _word_align_capture,
    _word_align_dtw,
    _word_align_median_filter,
    _word_align_neg_weights,
)
from vllm.v1.worker.gpu.word_align import WordAlignCapturer

TIME_PRECISION = 0.02


def _hf_onsets(neg_weights: np.ndarray) -> list[float]:
    """Reference onsets: transformers' DTW plus the same jump extraction."""
    text_idx, time_idx = hf_dtw(neg_weights)
    jumps = np.pad(np.diff(text_idx), (1, 0), constant_values=1).astype(bool)
    return (time_idx[jumps] * TIME_PRECISION).tolist()


@pytest.mark.parametrize("width", [3, 7])
def test_median_filter_matches_transformers(width: int):
    x = torch.randn(6, 12, 40)
    torch.testing.assert_close(
        _word_align_median_filter(x, width), hf_median(x, width), atol=0, rtol=0
    )


def test_median_filter_passthrough_on_short_input():
    """Near-silent clips can have fewer frames than the reflect-pad width."""
    x = torch.randn(2, 3, 3)
    torch.testing.assert_close(_word_align_median_filter(x, 7), x, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(8, 30), (1, 5), (40, 9)])
def test_dtw_matches_transformers(shape: tuple[int, int]):
    rng = np.random.default_rng(0)
    neg_weights = rng.standard_normal(shape).astype(np.float32)
    assert _word_align_dtw(neg_weights) == _hf_onsets(neg_weights)


def test_neg_weights_only_reads_alignment_heads():
    """Perturbing a non-alignment head must not change the result."""
    num_heads, head_dim, layers = 2, 4, 2
    d_model = num_heads * head_dim

    torch.manual_seed(0)
    qbuf = torch.randn(layers, 6, d_model)
    kbuf = torch.randn(layers, 16, d_model)
    args = ([(0, 0)], num_heads, head_dim, 16, 3)

    before = _word_align_neg_weights(qbuf, kbuf, 4, *args)
    # Head 1 of layer 0 occupies the second head_dim slice of d_model.
    qbuf[0, :, head_dim:] += 10.0
    kbuf[1] += 10.0
    after = _word_align_neg_weights(qbuf, kbuf, 4, *args)

    np.testing.assert_allclose(before, after, atol=0)


def _capture_buffers(slots, layers, max_tgt, max_src, d_model, max_q, max_k):
    """Layer-major capture buffers plus the row-index buffers the runner fills."""
    return (
        torch.zeros(layers, slots, max_tgt, d_model),
        torch.zeros(layers, slots, max_src, d_model),
        torch.zeros(max_q, dtype=torch.long),
        torch.zeros(max_k, dtype=torch.long),
    )


def test_capture_scatters_batched_requests_into_their_own_slots():
    """Two requests decoding in one batch must not land in the same slot."""
    layers, d_model, max_tgt, max_src = 2, 4, 6, 8
    qbuf, kbuf, qidx, kidx = _capture_buffers(
        2, layers, max_tgt, max_src, d_model, 4, 6
    )
    # Row 0 -> slot 0 position 3, row 1 -> slot 1 position 0.
    qidx[:2] = torch.tensor([0 * max_tgt + 3, 1 * max_tgt + 0])
    # Encoder frames: 3 for slot 0, then 3 for slot 1.
    kidx[:6] = torch.tensor([0, 1, 2, max_src, max_src + 1, max_src + 2])

    q = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    k = torch.arange(6 * d_model, dtype=torch.float32).view(6, d_model)
    layer = 1
    _word_align_capture(q, k, qbuf, kbuf, qidx, kidx, layer)

    torch.testing.assert_close(qbuf[layer, 0, 3], q[0])
    torch.testing.assert_close(qbuf[layer, 1, 0], q[1])
    torch.testing.assert_close(kbuf[layer, 0, :3], k[:3])
    torch.testing.assert_close(kbuf[layer, 1, :3], k[3:])

    # Nothing leaked into the other layer or into unwritten positions.
    assert qbuf[0].abs().sum() == 0
    assert kbuf[0].abs().sum() == 0
    assert qbuf[layer, 0, 0].abs().sum() == 0


def test_capture_skips_oversized_batches():
    """The profiling/warmup batch is larger than the index buffers: no-op."""
    qbuf, kbuf, qidx, kidx = _capture_buffers(1, 1, 4, 4, 4, 2, 2)
    q = torch.ones(8, 4)
    _word_align_capture(q, None, qbuf, kbuf, qidx, kidx, 0)
    assert qbuf.abs().sum() == 0
    assert kbuf.abs().sum() == 0


def test_capture_without_encoder_keys_leaves_kbuf_untouched():
    """Decode steps pass k=None; only the encoder prefill writes K."""
    qbuf, kbuf, qidx, kidx = _capture_buffers(1, 1, 4, 4, 4, 2, 2)
    kbuf.fill_(7.0)
    q = torch.ones(1, 4)
    _word_align_capture(q, None, qbuf, kbuf, qidx, kidx, 0)
    torch.testing.assert_close(qbuf[0, 0, 0], q[0])
    assert bool((kbuf == 7.0).all())


def _fake_batch(req_ids, num_scheduled, num_computed):
    """Minimal stand-in for the fields WordAlignCapturer reads off InputBatch."""
    num_scheduled = np.asarray(num_scheduled, dtype=np.int32)
    qsl = np.zeros(len(req_ids) + 1, dtype=np.int32)
    np.cumsum(num_scheduled, out=qsl[1:])
    return SimpleNamespace(
        req_ids=list(req_ids),
        num_reqs=len(req_ids),
        num_tokens=int(num_scheduled.sum()),
        num_tokens_after_padding=int(num_scheduled.sum()),
        query_start_loc=torch.from_numpy(qsl),
        idx_mapping=torch.arange(len(req_ids), dtype=torch.int32),
        num_scheduled_tokens=num_scheduled,
        num_computed_tokens_np=np.asarray(num_computed, dtype=np.int32),
    )


def _capturer(num_slots=2, max_frames=4, max_tokens=16, max_tgt=8):
    cap = WordAlignCapturer()
    cap.enabled = True
    cap.scratch = num_slots
    cap._free = list(range(num_slots))
    cap.max_frames = max_frames
    cap.max_tgt = max_tgt
    cap._qlimit = (num_slots + 1) * max_tgt - 1
    cap.device = torch.device("cpu")
    cap.qidx = torch.zeros(max_tokens, dtype=torch.int64)
    cap.kidx = torch.zeros(num_slots * max_frames, dtype=torch.int64)
    cap.positions = torch.zeros(max_tokens, dtype=torch.int64)
    cap._arange = torch.arange(max_tokens, dtype=torch.int32)
    cap._frames = np.arange(max_frames, dtype=np.int64)
    return cap


def test_capture_pool_routes_each_request_to_its_own_slot():
    cap = _capturer()
    cap.before_forward(_fake_batch(["a", "b"], [2, 3], [0, 0]))

    assert set(cap.slot_of) == {"a", "b"}
    assert cap.slot_of["a"] != cap.slot_of["b"]
    # Two tokens for "a" then three for "b", each on its own slot.
    expected = [cap.slot_of["a"]] * 2 + [cap.slot_of["b"]] * 3
    assert (cap.qidx[:5] // cap.max_tgt).tolist() == expected
    # Both are prefilling, so each contributes one full encoder window.
    frames = list(range(cap.max_frames))
    assert cap.kidx[:4].tolist() == [
        cap.slot_of["a"] * cap.max_frames + f for f in frames
    ]
    assert cap.kidx[4:8].tolist() == [
        cap.slot_of["b"] * cap.max_frames + f for f in frames
    ]


def test_capture_row_index_combines_slot_and_position():
    """The index the capture op scatters by must encode this step's position."""
    cap = _capturer()
    cap.positions[:2] = torch.tensor([5, 2])
    cap.before_forward(_fake_batch(["a", "b"], [1, 1], [5, 2]))

    assert cap.qidx[0].item() == cap.slot_of["a"] * cap.max_tgt + 5
    assert cap.qidx[1].item() == cap.slot_of["b"] * cap.max_tgt + 2


def test_capture_pool_keeps_slot_across_steps():
    cap = _capturer()
    cap.before_forward(_fake_batch(["a"], [3], [0]))
    slot = cap.slot_of["a"]
    cap.before_forward(_fake_batch(["a"], [1], [3]))  # decode step

    assert cap.slot_of["a"] == slot
    assert cap.npos["a"] == 4  # 3 computed + 1 scheduled


def test_capture_pool_exhaustion_leaves_requests_untracked():
    """Overflow requests must land on scratch, not share another's buffer."""
    cap = _capturer(num_slots=2)
    cap.before_forward(_fake_batch(["a", "b", "c"], [1, 1, 1], [0, 0, 0]))

    assert set(cap.slot_of) == {"a", "b"}
    assert "c" not in cap.slot_of
    assert cap.qidx[2].item() // cap.max_tgt == cap.scratch


def test_capture_pool_releases_slot_on_removal():
    cap = _capturer(num_slots=2)
    cap.before_forward(_fake_batch(["a", "b"], [1, 1], [0, 0]))
    freed = cap.slot_of["a"]
    cap.remove_request("a")

    assert "a" not in cap.slot_of and "a" not in cap.npos
    assert freed in cap._free
    # The freed slot is handed to the next request that needs one.
    cap.before_forward(_fake_batch(["b", "c"], [1, 1], [1, 0]))
    assert cap.slot_of["c"] == freed


def test_capture_pool_skips_oversized_encoder_batch():
    """The profiling run schedules more encoder rows than the pool holds."""
    cap = _capturer(num_slots=2, max_frames=4)
    cap.kidx.fill_(7)
    cap.before_forward(_fake_batch(["a", "b", "c"], [1, 1, 1], [0, 0, 0]))

    # 3 prefills x 4 frames = 12 rows, buffer holds 8: left untouched.
    assert bool((cap.kidx == 7).all())
