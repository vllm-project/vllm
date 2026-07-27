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


@pytest.mark.parametrize("width", [3, 5, 7])
def test_median_filter_matches_transformers(width: int):
    x = torch.randn(6, 12, 40)
    torch.testing.assert_close(
        _word_align_median_filter(x, width), hf_median(x, width), atol=0, rtol=0
    )


def test_median_filter_passthrough_on_short_input():
    """Near-silent clips can have fewer frames than the reflect-pad width."""
    x = torch.randn(2, 3, 3)
    torch.testing.assert_close(_word_align_median_filter(x, 7), x, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(8, 30), (1, 5), (25, 25), (40, 9)])
def test_dtw_matches_transformers(shape: tuple[int, int]):
    rng = np.random.default_rng(0)
    neg_weights = rng.standard_normal(shape).astype(np.float32)
    assert _word_align_dtw(neg_weights) == _hf_onsets(neg_weights)


def test_dtw_onsets_are_monotonic_and_bounded():
    rng = np.random.default_rng(1)
    n_positions, n_frames = 20, 60
    neg_weights = rng.standard_normal((n_positions, n_frames)).astype(np.float32)
    onsets = _word_align_dtw(neg_weights)

    # One onset per decoder position, non-decreasing, inside the audio window.
    assert len(onsets) == n_positions
    assert onsets == sorted(onsets)
    assert onsets[0] >= 0.0
    assert onsets[-1] <= (n_frames - 1) * TIME_PRECISION


def test_dtw_follows_a_block_diagonal_alignment():
    """Each token attending to its own frame block yields evenly spaced onsets."""
    n_positions, stride = 6, 4
    weights = np.zeros((n_positions, n_positions * stride), dtype=np.float32)
    for i in range(n_positions):
        weights[i, i * stride : (i + 1) * stride] = 1.0
    onsets = _word_align_dtw(-weights)
    expected = [i * stride * TIME_PRECISION for i in range(n_positions)]
    assert onsets == pytest.approx(expected, abs=stride * TIME_PRECISION)


def _reference_neg_weights(
    qbuf, kbuf, n_positions, heads, num_heads, head_dim, num_audio_frames, width
):
    scaling = head_dim**-0.5
    frames = num_audio_frames // 2
    per_head = []
    for layer, head in heads:
        q = qbuf[layer, :n_positions].float().view(n_positions, num_heads, head_dim)
        k = kbuf[layer].float().view(-1, num_heads, head_dim)
        per_head.append(torch.softmax(scaling * (q[:, head] @ k[:, head].T), dim=-1))
    w = torch.stack(per_head)[..., :frames]
    w = (w - w.mean(-2, keepdim=True)) / w.std(-2, keepdim=True, unbiased=False)
    return (-hf_median(w, width).mean(dim=0)).float().numpy()


def test_neg_weights_matches_reference_pipeline():
    num_heads, head_dim, layers, max_src = 2, 4, 3, 20
    d_model = num_heads * head_dim
    n_positions, num_audio_frames = 5, 24  # encoder emits num_audio_frames // 2

    torch.manual_seed(0)
    qbuf = torch.randn(layers, 8, d_model)
    kbuf = torch.randn(layers, max_src, d_model)
    heads = [(0, 1), (2, 0)]

    out = _word_align_neg_weights(
        qbuf, kbuf, n_positions, heads, num_heads, head_dim, num_audio_frames, 3
    )
    expected = _reference_neg_weights(
        qbuf, kbuf, n_positions, heads, num_heads, head_dim, num_audio_frames, 3
    )

    assert out.dtype == np.float32
    assert out.shape == (n_positions, num_audio_frames // 2)
    np.testing.assert_allclose(out, expected, atol=1e-5)


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
    return (
        torch.zeros(slots, layers, max_tgt, d_model),
        torch.zeros(slots, layers, max_src, d_model),
        torch.zeros(max_q, dtype=torch.long),
        torch.zeros(max_q, dtype=torch.long),
        torch.zeros(max_k, dtype=torch.long),
        torch.zeros(max_k, dtype=torch.long),
    )


def test_capture_scatters_batched_requests_into_their_own_slots():
    """Two requests decoding in one batch must not land in the same slot."""
    layers, d_model, max_tgt, max_src = 2, 4, 6, 8
    qbuf, kbuf, qslot, qpos, kslot, kpos = _capture_buffers(
        2, layers, max_tgt, max_src, d_model, 4, 6
    )
    # Row 0 -> slot 0 position 3, row 1 -> slot 1 position 0.
    qslot[:2] = torch.tensor([0, 1])
    qpos[:2] = torch.tensor([3, 0])
    # Encoder frames: 3 for slot 0, then 3 for slot 1.
    kslot[:6] = torch.tensor([0, 0, 0, 1, 1, 1])
    kpos[:6] = torch.tensor([0, 1, 2, 0, 1, 2])

    q = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    k = torch.arange(6 * d_model, dtype=torch.float32).view(6, d_model)
    layer = 1
    _word_align_capture(q, k, qbuf, kbuf, qslot, qpos, kslot, kpos, layer)

    torch.testing.assert_close(qbuf[0, layer, 3], q[0])
    torch.testing.assert_close(qbuf[1, layer, 0], q[1])
    torch.testing.assert_close(kbuf[0, layer, :3], k[:3])
    torch.testing.assert_close(kbuf[1, layer, :3], k[3:])

    # Nothing leaked into the other layer or into unwritten positions.
    assert qbuf[:, 0].abs().sum() == 0
    assert kbuf[:, 0].abs().sum() == 0
    assert qbuf[0, layer, 0].abs().sum() == 0


def test_capture_skips_oversized_batches():
    """The profiling/warmup batch is larger than the index buffers: no-op."""
    qbuf, kbuf, qslot, qpos, kslot, kpos = _capture_buffers(1, 1, 4, 4, 4, 2, 2)
    q = torch.ones(8, 4)
    _word_align_capture(q, None, qbuf, kbuf, qslot, qpos, kslot, kpos, 0)
    assert qbuf.abs().sum() == 0
    assert kbuf.abs().sum() == 0


def test_capture_without_encoder_keys_leaves_kbuf_untouched():
    """Decode steps pass k=None; only the encoder prefill writes K."""
    qbuf, kbuf, qslot, qpos, kslot, kpos = _capture_buffers(1, 1, 4, 4, 4, 2, 2)
    kbuf.fill_(7.0)
    q = torch.ones(1, 4)
    _word_align_capture(q, None, qbuf, kbuf, qslot, qpos, kslot, kpos, 0)
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


def _capturer(num_slots=2, max_frames=4, max_tokens=16):
    cap = WordAlignCapturer()
    cap.enabled = True
    cap.scratch = num_slots
    cap._free = list(range(num_slots))
    cap.max_frames = max_frames
    cap.device = torch.device("cpu")
    cap.qslot = torch.zeros(max_tokens, dtype=torch.int64)
    cap.kslot = torch.zeros(num_slots * max_frames, dtype=torch.int64)
    cap.kpos = torch.zeros(num_slots * max_frames, dtype=torch.int64)
    cap._arange = torch.arange(max_tokens, dtype=torch.int32)
    return cap


def test_capture_pool_routes_each_request_to_its_own_slot():
    cap = _capturer()
    cap.before_forward(_fake_batch(["a", "b"], [2, 3], [0, 0]))

    assert set(cap.slot_of) == {"a", "b"}
    assert cap.slot_of["a"] != cap.slot_of["b"]
    # Two tokens for "a" then three for "b", each on its own slot.
    expected = [cap.slot_of["a"]] * 2 + [cap.slot_of["b"]] * 3
    assert cap.qslot[:5].tolist() == expected
    # Both are prefilling, so each contributes one full encoder window.
    assert cap.kslot[:4].tolist() == [cap.slot_of["a"]] * 4
    assert cap.kslot[4:8].tolist() == [cap.slot_of["b"]] * 4


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
    assert cap.qslot[2].item() == cap.scratch


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
    cap.kslot.fill_(7)
    cap.before_forward(_fake_batch(["a", "b", "c"], [1, 1, 1], [0, 0, 0]))

    # 3 prefills x 4 frames = 12 rows, buffer holds 8: left untouched.
    assert bool((cap.kslot == 7).all())
