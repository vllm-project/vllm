# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Whisper word-timestamp helpers.

These cover the CPU/GPU-agnostic pieces of the cross-attention + DTW pipeline:
the capture scatter, the attention-weight post-processing and the DTW itself.
The DTW and median filter are checked against the transformers reference
implementations they are derived from, so a divergence shows up as a test
failure rather than as drifting timestamps.
"""

import logging
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
    _word_align_fetch,
    _word_align_finishing,
    _word_align_median_filter,
    _word_align_neg_weights,
)
from vllm.v1.worker.gpu.word_align import CAPTURE_MEMORY_FRACTION, WordAlignCapturer

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


# m = 750 is the real encoder-frame count for a 30s window (max_source_positions
# halved); n spans a ~5s clip up to max_target_positions.
@pytest.mark.parametrize("n", [1, 2, 40, 120, 300, 448])
def test_dtw_matches_transformers_at_production_shapes(n: int):
    """The compiled DTW must agree with transformers at the shapes the engine
    actually runs, not only at toy shapes."""
    rng = np.random.default_rng(n)
    neg_weights = rng.standard_normal((n, 750)).astype(np.float32)
    assert _word_align_dtw(neg_weights) == _hf_onsets(neg_weights)


def _tie_heavy_cases() -> list[np.ndarray]:
    """Cost matrices where the tie-breaking actually fires.

    ``t`` defaults to 2 and only a *strict* minimum moves it, so equal costs
    (e.g. ``c0 == c1 < c2``) resolve to the ``j - 1`` step even though it is not
    the cheapest. Few-valued matrices make that path fire on ~12% of cells;
    standardized, median-filtered attention weights produce exact cost ties on
    0.000% of cells (measured at n in {40,120,300,448}, m=750), so random floats
    never reach it.
    """
    cases = []
    for levels in (1, 2, 3):
        rng = np.random.default_rng(levels)
        for _ in range(20):
            n, m = int(rng.integers(1, 14)), int(rng.integers(1, 14))
            cases.append(rng.integers(0, levels + 1, (n, m)).astype(np.float32))
    # every cell tied: the degenerate end of the tie-breaking
    cases.append(np.zeros((30, 40), dtype=np.float32))
    cases.append(np.ones((30, 40), dtype=np.float32))
    # NaN: a one-position request standardizes by a zero std, so NaN does reach
    # the DTW, and np.minimum propagates it where a bare `<` chain would not.
    nan_case = np.full((4, 12), 0.5, dtype=np.float32)
    nan_case[1, 3:] = np.nan
    cases.append(nan_case)
    return cases


def test_dtw_kernel_matches_numpy_fallback():
    """The compiled DTW and the numba-less fallback must agree bit-for-bit.

    This is the real regression guard for the kernel: the fallback is the
    reference implementation this PR shipped, so any divergence here is a
    timestamp shift. It also covers the tie-breaking and NaN paths, which
    ``_hf_onsets`` cannot check -- see
    ``test_dtw_diverges_from_transformers_on_ties``.
    """
    from vllm.model_executor.models.whisper import _word_align_dtw_numpy

    rng = np.random.default_rng(99)
    cases = [rng.standard_normal((n, 750)).astype(np.float32) for n in (1, 2, 40, 200)]
    cases += _tie_heavy_cases()
    for neg_weights in cases:
        assert (
            _word_align_dtw(neg_weights)
            == _word_align_dtw_numpy(neg_weights, 0.02).tolist()
        )


def test_dtw_diverges_from_transformers_on_ties():
    """Documents a *pre-existing* gap between this DTW and transformers.

    transformers writes ``cost[i, j] = matrix[i-1, j-1] + c`` where ``c`` is the
    cost of the branch it *selected*, so on a tie it stores ``c2``; this
    implementation stores ``min(c0, c1, c2)``. The trace, and therefore the
    timestamps, then differ on tied cells. Untied inputs -- which is all
    realistic input, ties measured at 0.000% of cells -- still agree exactly, as
    ``test_dtw_matches_transformers*`` assert.

    Pinned as a test so the divergence is not mistaken for a regression in the
    compiled kernel, and so that fixing it is a deliberate, visible change.
    """
    diverged = [w for w in _tie_heavy_cases() if _word_align_dtw(w) != _hf_onsets(w)]
    assert diverged, (
        "expected the tie-breaking gap with transformers to still exist; if it "
        "was fixed, drop this test and assert parity on tied inputs instead"
    )


def test_neg_weights_only_reads_alignment_heads():
    """Perturbing a non-alignment head must not change the result."""
    num_heads, head_dim, layers = 2, 4, 2
    d_model = num_heads * head_dim

    torch.manual_seed(0)
    qbuf = torch.randn(layers, 6, d_model)
    kbuf = torch.randn(layers, 16, d_model)
    args = ([(0, 0)], num_heads, head_dim, 16, 3)

    before = _word_align_neg_weights(qbuf, kbuf, 4, *args).numpy()
    # Head 1 of layer 0 occupies the second head_dim slice of d_model.
    qbuf[0, :, head_dim:] += 10.0
    kbuf[1] += 10.0
    after = _word_align_neg_weights(qbuf, kbuf, 4, *args).numpy()

    np.testing.assert_allclose(before, after, atol=0)


def test_fetch_reassembles_each_request_bit_exactly():
    """The one-copy fetch must hand back exactly the per-request matrices.

    ``compute_word_align`` concatenates a step's weight matrices into a single
    device->host copy, so a slicing slip would silently give one request another
    request's alignment.
    """
    torch.manual_seed(0)
    weights = [torch.randn(n, m) for n, m in [(3, 5), (1, 7), (12, 2), (4, 4)]]

    fetched = _word_align_fetch(weights)

    assert [w.shape for w in fetched] == [tuple(w.shape) for w in weights]
    for want, got in zip(weights, fetched):
        assert np.array_equal(want.numpy(), got)
    # The single-matrix case takes a separate branch (no concatenation).
    assert np.array_equal(_word_align_fetch(weights[:1])[0], weights[0].numpy())


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


def _capturer(num_slots=2, max_frames=4, max_tokens=16, max_tgt=8, wants="abcde"):
    cap = WordAlignCapturer()
    cap.enabled = True
    # Capture is opt-in per request; these routing tests assume every request
    # asked for word timestamps. See test_capture_* gating tests for the opt-out.
    cap.wants = set(wants)
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


# --- per-request gating -----------------------------------------------------
# An armed server (--enable-word-timestamps) must cost nothing for requests that
# did not ask for words: without this, every request paid the readout (encoder
# cross-attention recompute + a blocking device->host copy + the DTW) and the
# response formatter discarded it.


def _params(word_timestamps: bool | None):
    extra = None if word_timestamps is None else {"word_timestamps": word_timestamps}
    return SimpleNamespace(extra_args=extra)


@pytest.mark.parametrize("flag", [None, False])
def test_add_request_ignores_requests_that_did_not_ask(flag):
    cap = _capturer(wants="")
    cap.add_request("a", _params(flag))
    assert cap.wants == set()


def test_add_request_opts_in_on_extra_args_flag():
    cap = _capturer(wants="")
    cap.add_request("a", _params(True))
    assert cap.wants == {"a"}


def test_add_request_tolerates_missing_sampling_params():
    cap = _capturer(wants="")
    cap.add_request("a", None)
    assert cap.wants == set()


def test_ungated_requests_get_no_slot_and_no_readout():
    """The whole readout must be skipped, not just its result discarded."""
    cap = _capturer(wants="")
    cap.add_request("a", _params(None))
    cap.before_forward(_fake_batch(["a"], [3], [0]))

    assert cap.slot_of == {}
    # Every capture row parks on the scratch slot, so the compiled capture op
    # still runs but writes nothing any request will read.
    assert (cap.qidx // cap.max_tgt).unique().tolist() == [cap.scratch]
    # No readout hook at all -> AsyncOutput.get_output() does no word-align work.
    assert cap.make_readout_fn() is None


def test_gating_is_per_request_within_a_batch():
    cap = _capturer(wants="")
    cap.add_request("a", _params(True))
    cap.add_request("b", _params(None))
    cap.before_forward(_fake_batch(["a", "b"], [1, 1], [0, 0]))

    assert set(cap.slot_of) == {"a"}
    assert cap.qidx[1].item() // cap.max_tgt == cap.scratch
    # The readout only sees the request that asked.
    assert cap.make_readout_fn() is not None


def test_removal_clears_the_opt_in():
    cap = _capturer(wants="")
    cap.add_request("a", _params(True))
    cap.before_forward(_fake_batch(["a"], [1], [0]))
    cap.remove_request("a")

    assert cap.wants == set()
    assert cap.make_readout_fn() is None


def test_parked_steps_touch_nothing():
    """While no request wants timestamps, a step must do no routing work at all.

    Parking once and then short-circuiting is what makes an armed server free.
    The cost of losing it is far below the resolution of any throughput
    benchmark (a few percent at best against ~4% run-to-run spread), so this
    test is the only thing that can catch a regression here.
    """
    cap = _capturer(wants="")
    batch = _fake_batch(["a"], [3], [0])
    cap.before_forward(batch)
    assert cap._parked
    # Poison the row buffers: a short-circuiting step leaves them untouched.
    cap.qidx.fill_(-1)
    cap.kidx.fill_(-1)
    cap.before_forward(batch)
    assert cap.qidx.unique().tolist() == [-1]
    assert cap.kidx.unique().tolist() == [-1]


def test_capture_reparks_once_the_last_opted_in_request_leaves():
    """Rows dirtied by an active step must go back to scratch, not stay aimed at
    a freed slot -- otherwise an ungated request writes into that slot and the
    next request to be handed it reads someone else's capture."""
    cap = _capturer(wants="")
    cap.add_request("a", _params(True))
    cap.before_forward(_fake_batch(["a"], [1], [0]))
    assert not cap._parked
    slot = cap.slot_of["a"]
    cap.remove_request("a")

    cap.before_forward(_fake_batch(["b"], [1], [0]))

    assert cap._parked
    assert (cap.qidx // cap.max_tgt).unique().tolist() == [cap.scratch]
    assert (cap.kidx // cap.max_frames).unique().tolist() == [cap.scratch]
    assert slot in cap._free


# --- capture pool sizing ----------------------------------------------------
# init() runs inside load_model(), i.e. before _initialize_kv_caches(), so every
# byte the capture pool takes comes straight out of the KV cache. The budget it
# is sized against therefore has to be relative to the memory allowance, not a
# fixed byte count: a fixed 2 GiB is 1.5% of the allowance at
# --gpu-memory-utilization=0.9 but 14% at 0.1, where it silently shrinks KV.


def _hf_config(d_model=1280, max_target_positions=448, max_source_positions=1500):
    return SimpleNamespace(
        d_model=d_model,
        max_target_positions=max_target_positions,
        max_source_positions=max_source_positions,
    )


def _gen_config(layers=(7, 10, 12)):
    return SimpleNamespace(alignment_heads=[[layer, 0] for layer in layers])


def test_slot_bytes_counts_only_the_alignment_layers():
    """turbo carries 6 heads on 2 layers; the other 30 must reserve nothing."""
    hf = _hf_config()
    one = WordAlignCapturer._slot_bytes(
        _gen_config((3,)), hf, itemsize=2, max_frames=1500
    )
    three = WordAlignCapturer._slot_bytes(
        _gen_config((3, 7, 9)), hf, itemsize=2, max_frames=1500
    )
    assert one == (1500 + 448) * 1 * 1280 * 2
    assert three == 3 * one


def test_slot_bytes_counts_a_layer_once_per_layer_not_per_head():
    hf = _hf_config()
    heads_on_two_layers = SimpleNamespace(
        alignment_heads=[[2, 4], [2, 11], [3, 3], [3, 6], [3, 11], [3, 14]]
    )
    assert WordAlignCapturer._slot_bytes(
        heads_on_two_layers, hf, itemsize=2, max_frames=1500
    ) == WordAlignCapturer._slot_bytes(
        _gen_config((2, 3)), hf, itemsize=2, max_frames=1500
    )


def test_pool_size_scales_with_the_budget():
    """Twice the budget, twice the coverage."""
    small = WordAlignCapturer._pool_size(
        max_num_reqs=1024, budget_bytes=11 * 100, slot_bytes=100
    )
    large = WordAlignCapturer._pool_size(
        max_num_reqs=1024, budget_bytes=21 * 100, slot_bytes=100
    )
    assert (small, large) == (10, 20)


def test_pool_size_reserves_the_scratch_slot():
    """init() allocates num_slots + 1 buffers, so the budget must cover them all."""
    assert (
        WordAlignCapturer._pool_size(
            max_num_reqs=1024, budget_bytes=10 * 100, slot_bytes=100
        )
        == 9
    )


def test_pool_size_is_capped_by_max_num_reqs():
    """More slots than concurrent requests is memory taken from KV for nothing."""
    assert (
        WordAlignCapturer._pool_size(
            max_num_reqs=8, budget_bytes=1000 * 100, slot_bytes=100
        )
        == 8
    )


@pytest.mark.parametrize("budget", [0, 100, 199])
def test_pool_size_never_drops_below_one_slot(budget: int):
    """The buffers are indexed assuming at least one real slot exists."""
    assert (
        WordAlignCapturer._pool_size(
            max_num_reqs=64, budget_bytes=budget, slot_bytes=100
        )
        == 1
    )


def test_capture_budget_is_a_fraction_of_the_unspent_allowance():
    """The allowance is total * util; the weights are already loaded by now."""
    total = 1000
    free = 900  # 100 already spent on weights + context
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        cache_config=SimpleNamespace(gpu_memory_utilization=0.5),
    )
    budget = WordAlignCapturer._capture_budget_bytes(
        runner, memory_info=lambda _dev: (free, total)
    )
    # allowance 500, already spent 100 -> 400 unspent, of which a fraction.
    assert budget == int(400 * CAPTURE_MEMORY_FRACTION)


def test_capture_budget_shrinks_with_a_small_utilization():
    """The bug this replaces: a fixed 2 GiB ignored --gpu-memory-utilization."""
    total = 1000
    runner_low = SimpleNamespace(
        device=torch.device("cpu"),
        cache_config=SimpleNamespace(gpu_memory_utilization=0.1),
    )
    runner_high = SimpleNamespace(
        device=torch.device("cpu"),
        cache_config=SimpleNamespace(gpu_memory_utilization=0.9),
    )
    info = lambda _dev: (950, total)  # noqa: E731
    low = WordAlignCapturer._capture_budget_bytes(runner_low, memory_info=info)
    high = WordAlignCapturer._capture_budget_bytes(runner_high, memory_info=info)
    assert low < high
    assert low == int(50 * CAPTURE_MEMORY_FRACTION)  # 100 allowance - 50 spent
    assert high == int(850 * CAPTURE_MEMORY_FRACTION)


def test_capture_budget_is_zero_when_the_allowance_is_already_spent():
    """Never hand back a negative budget; one slot is the floor, not a crash."""
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        cache_config=SimpleNamespace(gpu_memory_utilization=0.1),
    )
    budget = WordAlignCapturer._capture_budget_bytes(
        runner, memory_info=lambda _dev: (500, 1000)
    )
    assert budget == 0


# --- pool exhaustion must not be silent ------------------------------------
# A request that wants words and cannot get a slot returns words=None, which is
# indistinguishable from audio with no speech. That is how coverage fell to
# ~70% under load with nothing in the logs.


def test_pool_exhaustion_warns(caplog):
    cap = _capturer(num_slots=1, wants="abc")
    with caplog.at_level(logging.WARNING):
        cap.before_forward(_fake_batch(["a", "b", "c"], [1, 1, 1], [0, 0, 0]))
    assert "capture pool exhausted" in caplog.text
    assert cap.num_denied == 2


def test_pool_exhaustion_does_not_warn_again_for_the_same_request(caplog):
    """The denial is per request, so a decode loop must not log every step."""
    cap = _capturer(num_slots=1, wants="ab")
    cap.before_forward(_fake_batch(["a", "b"], [1, 1], [0, 0]))
    caplog.clear()  # the first denial is expected to log; the next 20 are not.
    with caplog.at_level(logging.WARNING):
        for _ in range(20):
            cap.before_forward(_fake_batch(["a", "b"], [1, 1], [1, 1]))
    assert caplog.text == ""
    assert cap.num_denied == 1


def test_pool_exhaustion_count_is_cumulative_and_logged_sparsely(caplog):
    """Bounded logging: one line at 1, 2, 4, 8 ... denials, never one per request."""
    cap = _capturer(num_slots=1, wants="")
    cap.wants = {str(i) for i in range(40)}
    with caplog.at_level(logging.WARNING):
        for i in range(40):
            cap.before_forward(_fake_batch([str(i)], [1], [0]))
    assert cap.num_denied == 39
    # 1, 2, 4, 8, 16, 32 -> 6 lines for 39 denials.
    assert caplog.text.count("capture pool exhausted") == 6


def test_removal_forgets_the_denial(caplog):
    """Otherwise the bookkeeping grows for the lifetime of the server."""
    cap = _capturer(num_slots=1, wants="ab")
    cap.before_forward(_fake_batch(["a", "b"], [1, 1], [0, 0]))
    assert cap._denied == {"b"}
    cap.remove_request("b")
    assert cap._denied == set()


# --- which requests get read out -------------------------------------------


def _finishing(sampled, npos, final_npos=None, eos=50257, slots=None):
    req_ids = list(npos)
    return _word_align_finishing(
        req_ids,
        [sampled[r] for r in req_ids],
        slots if slots is not None else {r: i for i, r in enumerate(req_ids)},
        npos,
        final_npos or {},
        eos,
    )


def test_readout_fires_for_the_request_that_emitted_eos():
    assert _finishing({"a": [7, 50257], "b": [7]}, {"a": 12, "b": 12}) == [("a", 0, 12)]


def test_readout_fires_when_the_token_budget_is_exhausted():
    """A request that stops on max_tokens finishes without ever emitting eos.

    Before this, such a request silently returned no word timestamps: the readout
    only recognised eos, and nothing else in the pipeline noticed.
    """
    assert _finishing({"a": [7]}, {"a": 40}, final_npos={"a": 40}) == [("a", 0, 40)]


def test_readout_does_not_fire_before_the_budget_is_reached():
    assert _finishing({"a": [7]}, {"a": 39}, final_npos={"a": 40}) == []


def test_readout_skips_requests_without_a_slot():
    """Pool exhaustion: no capture happened, so there is nothing to align."""
    assert _finishing({"a": [7, 50257]}, {"a": 12}, slots={}) == []


def test_readout_skips_requests_that_decoded_nothing():
    assert _finishing({"a": [7, 50257]}, {"a": 0}) == []


def test_readout_tolerates_a_short_sampled_token_list():
    assert _word_align_finishing(
        ["a", "b"], [[50257]], {"a": 0, "b": 1}, {"a": 3, "b": 3}, {}, 50257
    ) == [("a", 0, 3)]


def test_final_position_is_recorded_from_the_token_budget():
    cap = _capturer(wants="", max_tgt=448)
    cap.add_request(
        "a", SimpleNamespace(extra_args={"word_timestamps": True}, max_tokens=40), 4
    )
    assert cap._final_npos["a"] == 43


def test_final_position_is_capped_at_the_capture_buffer_limit():
    """max_tokens defaults to the whole context, past which nothing is captured."""
    cap = _capturer(wants="", max_tgt=448)
    cap.add_request(
        "a", SimpleNamespace(extra_args={"word_timestamps": True}, max_tokens=448), 4
    )
    assert cap._final_npos["a"] == 447


def test_requests_that_did_not_ask_record_no_final_position():
    cap = _capturer(wants="")
    cap.add_request("a", SimpleNamespace(extra_args=None, max_tokens=40), 4)
    assert cap._final_npos == {}


def test_removal_forgets_the_final_position():
    cap = _capturer(wants="", max_tgt=448)
    cap.add_request(
        "a", SimpleNamespace(extra_args={"word_timestamps": True}, max_tokens=40), 4
    )
    cap.remove_request("a")
    assert cap._final_npos == {}


def test_readout_fn_only_passes_the_budgets_of_wanting_requests():
    cap = _capturer(wants="", max_tgt=448)
    params = SimpleNamespace(extra_args={"word_timestamps": True}, max_tokens=40)
    cap.add_request("a", params, 4)
    cap.add_request("b", params, 4)
    cap.before_forward(_fake_batch(["a", "b"], [4, 4], [0, 0]))
    cap.model = SimpleNamespace(
        compute_word_align=lambda *args: args,
    )
    fn = cap.make_readout_fn()
    assert fn is not None
    _, _, slots, npos, final_npos = fn(["a", "b"], [[1], [1]])
    assert final_npos == {"a": 43, "b": 43}
    assert set(slots) == {"a", "b"}
    assert npos == {"a": 4, "b": 4}


# --- a partially covered pool must be known before traffic ------------------
# The runtime exhaustion warning only fires once requests are already being
# dropped. Whether the pool can cover max_num_seqs at all is knowable at startup,
# and "200 OK with silently missing data" is the exact failure mode this feature
# keeps producing, so it is reported before the server accepts anything.

_MiB = 1024**2


def _log_pool(num_slots, max_num_reqs, slot_bytes=10 * _MiB, budget_bytes=None):
    if budget_bytes is None:
        budget_bytes = (num_slots + 1) * slot_bytes
    WordAlignCapturer._log_pool_size(num_slots, max_num_reqs, slot_bytes, budget_bytes)


def test_startup_warns_when_the_pool_cannot_cover_max_num_seqs(caplog):
    with caplog.at_level(logging.INFO):
        _log_pool(num_slots=50, max_num_reqs=128)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "50 slot(s)" in message
    assert "max_num_seqs is 128" in message
    # The number an operator actually needs: how much coverage they will get.
    assert "39%" in message
    # ... and both levers for fixing it.
    assert "--gpu-memory-utilization" in message
    assert "--max-num-seqs" in message


def test_startup_warning_reports_what_full_coverage_would_cost(caplog):
    """Actionable means quantified: say how much memory the missing slots need."""
    with caplog.at_level(logging.INFO):
        _log_pool(num_slots=50, max_num_reqs=128, slot_bytes=10 * _MiB)
    message = "".join(r.getMessage() for r in caplog.records)
    # 129 slots x 10 MiB = 1.26 GiB wanted, 0.50 GiB available.
    assert "1.26 GiB" in message
    assert "0.50 GiB" in message


def test_startup_does_not_warn_at_full_coverage(caplog):
    with caplog.at_level(logging.INFO):
        _log_pool(num_slots=128, max_num_reqs=128)
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
    info = "".join(r.getMessage() for r in caplog.records)
    assert "128 slot(s)" in info
    assert "full coverage" in info


def test_pool_targets_max_num_seqs_when_the_budget_allows():
    """One slot per concurrent request is the target, not a ceiling to grow into.

    A 2-alignment-layer model needs 9.5 MiB per slot, so max_num_seqs=256 wants
    ~2.4 GiB of capture buffers; the pool must actually ask for all of it.
    """
    slot_bytes = (1500 + 448) * 2 * 1280 * 2
    budget = 4 * 1024**3
    assert (
        WordAlignCapturer._pool_size(
            max_num_reqs=256, budget_bytes=budget, slot_bytes=slot_bytes
        )
        == 256
    )
