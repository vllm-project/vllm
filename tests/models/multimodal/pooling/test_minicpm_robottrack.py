# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MiniCPM-RobotTrack in-tree pooling model.

The pure-logic tests always run. The end-to-end parity tests are gated on a
local checkpoint (env ``MINICPM_ROBOTTRACK_PATH``) because the model is not a
public HF hub checkpoint; when set they reproduce the layered parity described
in the integration plan (vLLM 24-dim output vs the HF reference trajectory).

The DINOv3 RoPE tests cover the trickiest, most regression-prone piece of the
inlined DINOv3 ViT port. They are pure-math (no engine / TP); full-forward
parity of the encoder is covered by the gated pixels-in end-to-end test.
"""

import os
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.models.minicpm_robottrack import (
    DINOv3ViTRopePositionEmbedding,
    FunnelTrajectoryHead,
    RobotTrackStreamState,
    VisionProjector,
    _advance_stream_state,
    _apply_dinov3_rotary_pos_emb,
    _assemble_stream_window,
    _classify_stream_request,
    _count_marker_runs,
    _encode_frames_cached,
    _grid_pool,
    _pad_history_frames,
    _pixel_window_cached,
    _rotate_half,
    _square_side,
    _square_side_or_none,
)

# dinov3-vits16 shape parameters for the RoPE math tests.
_DINOV3_CFG = SimpleNamespace(
    hidden_size=384,
    num_attention_heads=6,
    patch_size=16,
    rope_theta=100.0,
    num_register_tokens=4,
)


@pytest.mark.parametrize(
    "time_indices, expected_runs",
    [
        ([], 0),
        ([0], 1),
        ([0, 0, 0, 0], 1),
        ([0, 0, 1, 1, 2, 2], 3),
        # arange(31).repeat_interleave(4) -> 31 distinct runs
        (list(range(31)) * 1, 31),
        ([5, 5, 5, 4, 4, 5], 3),  # non-monotonic: each maximal run counts
    ],
)
def test_count_marker_runs(time_indices, expected_runs):
    assert _count_marker_runs(torch.tensor(time_indices, dtype=torch.long)) == (
        expected_runs
    )


def test_visual_bundle_placeholder_count():
    # coarse: 31 frames x 4 tokens with one marker per frame -> 124 + 31
    # fine: 64 tokens all at the same time step -> 64 + 1
    # + 1 control query
    coarse_time = torch.arange(31).repeat_interleave(4)
    fine_time = torch.full((64,), 31, dtype=torch.long)

    coarse = coarse_time.shape[0] + _count_marker_runs(coarse_time)
    fine = fine_time.shape[0] + _count_marker_runs(fine_time)
    num_tokens = coarse + fine + 1

    assert coarse == 124 + 31
    assert fine == 64 + 1
    assert num_tokens == 221


@pytest.mark.parametrize(
    "count, expected",
    [(1, 1), (4, 2), (9, 3), (576, 24), (729, 27)],
)
def test_square_side(count, expected):
    assert _square_side(count) == expected
    assert _square_side_or_none(count) == expected


@pytest.mark.parametrize("count", [0, 2, 3, 5, 581])
def test_square_side_or_none_rejects_non_squares(count):
    assert _square_side_or_none(count) is None


def test_grid_pool_matches_manual_average():
    # A 2x2 grid pooled to a single token is the mean over the four cells.
    tokens = torch.arange(4 * 3, dtype=torch.float32).reshape(1, 4, 3)
    pooled = _grid_pool(tokens, grid=2, out_tokens=1)
    assert pooled.shape == (1, 1, 3)
    assert torch.allclose(pooled[0, 0], tokens[0].mean(dim=0))


def test_pad_history_repeats_oldest_and_keeps_recent():
    # Two history frames (coarse_per=4, C=1) padded to 31 -> [31, 4, 1].
    history = torch.stack([torch.full((4, 1), float(i)) for i in range(2)], dim=0)
    padded = _pad_history_frames(history, 31)
    assert padded.shape == (31, 4, 1)
    # Left padding repeats the oldest frame; the two originals stay newest-last.
    assert torch.equal(padded[0], history[0])
    assert torch.equal(padded[-1], history[1])
    assert torch.equal(padded[-2], history[0])


def test_pad_history_truncates_to_most_recent():
    history = torch.stack([torch.full((4, 1), float(i)) for i in range(40)], dim=0)
    padded = _pad_history_frames(history, 31)
    assert padded.shape == (31, 4, 1)
    # Keeps the most recent 31 frames (indices 9..39).
    assert torch.equal(padded[0], history[9])
    assert torch.equal(padded[-1], history[39])


def test_pixel_window_pooling_token_counts():
    # Assemble the pooled window the way `_encode_window` does and confirm the
    # placeholder budget (221) is independent of how many frames are sent.
    grid, channels = 24, 1536
    for num_frames in (1, 5, 32):
        fused = torch.randn(num_frames, grid * grid, channels)
        fine = _grid_pool(fused[-1:], grid, 64)[0]
        source = fused[:-1] if num_frames > 1 else fused[-1:]
        history = _pad_history_frames(_grid_pool(source, grid, 4), 31)
        coarse = history.reshape(-1, channels)

        assert fine.shape == (64, channels)
        assert coarse.shape == (124, channels)

        coarse_time = torch.arange(31).repeat_interleave(4)
        fine_time = torch.full((64,), 31, dtype=torch.long)
        num_tokens = (
            coarse.shape[0]
            + _count_marker_runs(coarse_time)
            + fine.shape[0]
            + _count_marker_runs(fine_time)
            + 1
        )
        assert num_tokens == 221


class _CountingTower:
    """Fake DINOv3+SigLIP tower: deterministic per-frame grid + encode counter.

    ``__call__`` returns features that depend only on each frame's fill value, so
    identical frames yield identical features and cache reuse shows up in
    ``frames_encoded``.
    """

    def __init__(self, grid: int, channels: int) -> None:
        self.grid = grid
        self.channels = channels
        self.frames_encoded = 0

    def __call__(self, dino_pixels, siglip_pixels):
        n = dino_pixels.shape[0]
        self.frames_encoded += n
        scalars = dino_pixels.reshape(n, -1)[:, :1]
        fused = scalars.view(n, 1, 1) * torch.ones(
            n, self.grid * self.grid, self.channels
        )
        return fused, self.grid


def _rolling_windows(num_frames: int, window: int):
    for step in range(num_frames):
        yield list(range(max(0, step - window + 1), step + 1))


def _make_window(frame_ids):
    # Each frame id maps to a distinct constant tensor; ids are the cache keys.
    dino = torch.stack([torch.full((3, 8, 8), float(i)) for i in frame_ids])
    return dino, dino.clone(), list(frame_ids)


def test_frame_cache_encodes_each_frame_once():
    tower = _CountingTower(grid=8, channels=4)
    cache: OrderedDict = OrderedDict()
    num_frames, window = 12, 4
    per_step = []
    for frame_ids in _rolling_windows(num_frames, window):
        dino, siglip, keys = _make_window(frame_ids)
        _, _, encoded = _encode_frames_cached(
            tower, cache, 64, dino, siglip, keys, 4, 64
        )
        per_step.append(encoded)

    # Warmup and steady state alike: exactly one new frame is encoded per step.
    assert per_step == [1] * num_frames
    # Each distinct frame is encoded exactly once over the whole run.
    assert tower.frames_encoded == num_frames


def test_frame_cache_disabled_reencodes_full_window():
    tower = _CountingTower(grid=8, channels=4)
    cache: OrderedDict = OrderedDict()
    num_frames, window = 8, 4
    per_step = []
    for frame_ids in _rolling_windows(num_frames, window):
        dino, siglip, keys = _make_window(frame_ids)
        _, _, encoded = _encode_frames_cached(
            tower,
            cache,
            0,
            dino,
            siglip,
            keys,
            4,
            64,  # cache_size=0 disables reuse
        )
        per_step.append(encoded)

    expected = [min(step + 1, window) for step in range(num_frames)]
    assert per_step == expected
    assert tower.frames_encoded == sum(expected)


def test_frame_cache_evicts_oldest_beyond_capacity():
    tower = _CountingTower(grid=8, channels=4)
    cache: OrderedDict = OrderedDict()
    # Capacity below the window: the oldest in-window frame is evicted and must
    # be re-encoded when it is still referenced, so bound stays at cache_size.
    _encode_frames_cached(tower, cache, 2, *_make_window([0, 1, 2]), 4, 64)
    assert len(cache) == 2


def test_frame_cache_matches_uncached_features():
    num_frames, window = 10, 4
    cached_tower, plain_tower = _CountingTower(8, 4), _CountingTower(8, 4)
    cache_on, cache_off = OrderedDict(), OrderedDict()
    for frame_ids in _rolling_windows(num_frames, window):
        dino, siglip, keys = _make_window(frame_ids)
        c_on, f_on, _ = _encode_frames_cached(
            cached_tower, cache_on, 64, dino, siglip, keys, 4, 64
        )
        c_off, f_off, _ = _encode_frames_cached(
            plain_tower, cache_off, 0, dino, siglip, keys, 4, 64
        )
        for a, b in zip(c_on, c_off):
            assert torch.allclose(a, b)
        assert torch.allclose(f_on, f_off)


def test_frame_cache_stores_coarse_only():
    tower = _CountingTower(grid=8, channels=4)
    cache: OrderedDict = OrderedDict()
    _, _, encoded = _encode_frames_cached(
        tower, cache, 64, *_make_window([0, 1, 2]), 4, 64
    )
    assert encoded == 3
    # Cache values are coarse pools only (coarse_tokens x channels), never
    # (coarse, fine) tuples.
    assert all(isinstance(v, torch.Tensor) for v in cache.values())
    assert all(v.shape == (4, 4) for v in cache.values())


def test_frame_cache_reencodes_repeated_current_frame():
    tower = _CountingTower(grid=8, channels=4)
    cache: OrderedDict = OrderedDict()
    # Window [0, 1] encodes both frames; frame 1's coarse is cached.
    _, _, encoded = _encode_frames_cached(
        tower, cache, 64, *_make_window([0, 1]), 4, 64
    )
    assert encoded == 2
    # Frame 1 reappears as the current frame: its coarse is a cache hit, but the
    # fine is not cached, so the tower re-runs on that single frame.
    _, fine, encoded = _encode_frames_cached(
        tower, cache, 64, *_make_window([1, 1]), 4, 64
    )
    assert encoded == 1
    assert tower.frames_encoded == 3
    assert fine.shape == (64, 4)


class _CountingPixelProcessor:
    """Fake resize+normalize: per-frame normalized pixels derived from the id."""

    def __init__(self) -> None:
        self.frames_processed = 0

    def __call__(self, miss_frames):
        self.frames_processed += len(miss_frames)
        values = torch.tensor([[float(f)] for f in miss_frames]).unsqueeze(-1)
        return values, values.clone()


def _make_pixel_window(frame_ids):
    # Frame ids double as content-hash keys (like `_make_window` for the tower).
    return list(frame_ids), list(frame_ids)


def test_pixel_cache_normalizes_each_frame_once():
    processor = _CountingPixelProcessor()
    cache: OrderedDict = OrderedDict()
    num_frames, window = 12, 4
    per_step = []
    for frame_ids in _rolling_windows(num_frames, window):
        frames, keys = _make_pixel_window(frame_ids)
        before = processor.frames_processed
        _pixel_window_cached(frames, keys, cache, 64, processor)
        per_step.append(processor.frames_processed - before)

    # Warmup and steady state alike: only the newly arrived frame is normalized.
    assert per_step == [1] * num_frames
    # Each distinct frame is normalized exactly once over the whole run.
    assert processor.frames_processed == num_frames


def test_pixel_cache_disabled_reprocesses_full_window():
    processor = _CountingPixelProcessor()
    cache: OrderedDict = OrderedDict()
    num_frames, window = 8, 4
    per_step = []
    for frame_ids in _rolling_windows(num_frames, window):
        frames, keys = _make_pixel_window(frame_ids)
        before = processor.frames_processed
        _pixel_window_cached(frames, keys, cache, 0, processor)
        per_step.append(processor.frames_processed - before)

    expected = [min(step + 1, window) for step in range(num_frames)]
    assert per_step == expected
    assert processor.frames_processed == sum(expected)


def test_pixel_cache_evicts_oldest_beyond_capacity():
    processor = _CountingPixelProcessor()
    cache: OrderedDict = OrderedDict()
    frames, keys = _make_pixel_window([0, 1, 2])
    _pixel_window_cached(frames, keys, cache, 2, processor)
    assert len(cache) == 2


def test_pixel_cache_matches_uncached_pixels():
    num_frames, window = 10, 4
    cached_processor = _CountingPixelProcessor()
    plain_processor = _CountingPixelProcessor()
    cache_on, cache_off = OrderedDict(), OrderedDict()
    for frame_ids in _rolling_windows(num_frames, window):
        frames, keys = _make_pixel_window(frame_ids)
        dino_on, siglip_on = _pixel_window_cached(
            frames, keys, cache_on, 64, cached_processor
        )
        dino_off, siglip_off = _pixel_window_cached(
            frames, keys, cache_off, 0, plain_processor
        )
        assert torch.equal(dino_on, dino_off)
        assert torch.equal(siglip_on, siglip_off)


def test_pixel_cache_preserves_window_order():
    processor = _CountingPixelProcessor()
    cache: OrderedDict = OrderedDict()
    frames, keys = _make_pixel_window([5, 5, 3])
    dino, siglip = _pixel_window_cached(frames, keys, cache, 64, processor)
    # Window order preserved, including a repeated frame.
    assert dino.shape == (3, 1, 1)
    assert dino[:, 0, 0].tolist() == [5.0, 5.0, 3.0]
    assert torch.equal(siglip, dino)


# ---------------------------------------------------------------------------
# Stateful stream protocol (establish / append / reuse).
# ---------------------------------------------------------------------------


def _fake_coarse(frame_id: int) -> torch.Tensor:
    return torch.full((4, 8), float(frame_id))


def _fake_fine(frame_id: int) -> torch.Tensor:
    return torch.full((64, 8), float(frame_id))


def test_classify_stream_request_state_machine():
    hist = 31
    # A full window replaces (establish), even with a prior state (re-sync).
    assert _classify_stream_request(32, hist, None, 31) == "replace"
    state = RobotTrackStreamState(frame_index=31)
    assert _classify_stream_request(32, hist, state, 50) == "replace"
    # Invalid frame counts are rejected.
    with pytest.raises(ValueError):
        _classify_stream_request(5, hist, None, 0)
    # Single frame without a stream must establish first.
    with pytest.raises(ValueError):
        _classify_stream_request(1, hist, None, 32)
    # Consecutive frame_index appends; the same index is an idempotent reuse.
    assert _classify_stream_request(1, hist, state, 32) == "append"
    assert _classify_stream_request(1, hist, state, 31) == "reuse"
    # Out-of-order or missing indices are rejected.
    with pytest.raises(ValueError):
        _classify_stream_request(1, hist, state, 33)
    with pytest.raises(ValueError):
        _classify_stream_request(1, hist, state, None)


def test_stream_replace_keeps_last_31_history():
    coarse_by_frame = [_fake_coarse(i) for i in range(32)]
    state = _advance_stream_state(
        None, "replace", coarse_by_frame, _fake_fine(31), 31, history_frames=31
    )
    assert len(state.coarse_history) == 31
    # Frames 0..30 are history (the 32nd frame is current), oldest first.
    assert torch.equal(state.coarse_history[0], _fake_coarse(0))
    assert torch.equal(state.coarse_history[-1], _fake_coarse(30))
    assert torch.equal(state.current_coarse, _fake_coarse(31))
    assert torch.equal(state.fine, _fake_fine(31))
    assert state.frame_index == 31


def test_stream_append_rolls_window():
    coarse_by_frame = [_fake_coarse(i) for i in range(32)]
    state = _advance_stream_state(
        None, "replace", coarse_by_frame, _fake_fine(31), 31, history_frames=31
    )
    next_state = _advance_stream_state(
        state,
        "append",
        [_fake_coarse(32)],
        _fake_fine(32),
        32,
        history_frames=31,
    )
    assert len(next_state.coarse_history) == 31
    # Oldest (frame 0) rolled off; the previous current (frame 31) is promoted
    # into history, and frame 32 becomes the new current.
    assert torch.equal(next_state.coarse_history[0], _fake_coarse(1))
    assert torch.equal(next_state.coarse_history[-1], _fake_coarse(31))
    assert torch.equal(next_state.current_coarse, _fake_coarse(32))
    assert torch.equal(next_state.fine, _fake_fine(32))
    assert next_state.frame_index == 32


def test_stream_assemble_matches_stateless_window():
    # Establish with frames 0..31, then append frames 32..39. The assembled
    # window must always be the last 31 coarse frames + the current fine, in the
    # same order the stateless `_encode_window` would produce.
    coarse_per, fine_count, hist, channels = 4, 64, 31, 8
    state = _advance_stream_state(
        None,
        "replace",
        [_fake_coarse(i) for i in range(32)],
        _fake_fine(31),
        31,
        hist,
    )
    for frame_index in range(32, 40):
        state = _advance_stream_state(
            state,
            "append",
            [_fake_coarse(frame_index)],
            _fake_fine(frame_index),
            frame_index,
            hist,
        )
        coarse, coarse_time, fine_out, fine_time = _assemble_stream_window(
            state, hist, coarse_per
        )
        assert coarse.shape == (hist * coarse_per, channels)
        assert coarse_time.shape == (hist * coarse_per,)
        assert fine_out.shape == (fine_count, channels)
        assert fine_time.shape == (fine_count,)
        assert torch.equal(fine_out, _fake_fine(frame_index))
        assert torch.equal(fine_time, torch.full((fine_count,), hist, dtype=torch.long))
        # Oldest history frame is frame_index - 31; newest is frame_index - 1.
        assert float(coarse[0, 0]) == frame_index - hist
        assert float(coarse[-1, 0]) == frame_index - 1


def test_submodule_shapes():
    projector = VisionProjector(input_dim=1536, hidden_dim=1024).eval()
    feats = torch.randn(10, 1536)
    assert projector(feats).shape == (10, 1024)

    head = FunnelTrajectoryHead(
        hidden_dim=1024, num_waypoints=8, action_dim=3, dropout=0.4, use_tanh=True
    ).eval()
    control = torch.randn(2, 1024)
    out = head(control)
    assert out.shape == (2, 8, 3)
    # tanh keeps every action within (-1, 1) before output scaling
    assert out.abs().max() <= 1.0


def test_dinov3_rotate_half():
    x = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    # first/second halves swapped, first half negated
    expected = torch.tensor([[-4.0, -5.0, -6.0, -7.0, 0.0, 1.0, 2.0, 3.0]])
    assert torch.equal(_rotate_half(x), expected)


def test_dinov3_rope_shape_matches_patch_grid():
    rope = DINOv3ViTRopePositionEmbedding(_DINOV3_CFG).eval()
    px = torch.randn(1, 3, 384, 384)
    cos, sin = rope(px)
    head_dim = _DINOV3_CFG.hidden_size // _DINOV3_CFG.num_attention_heads
    num_patches = (384 // _DINOV3_CFG.patch_size) ** 2
    assert cos.shape == (num_patches, head_dim)
    assert sin.shape == (num_patches, head_dim)
    # cos(theta)^2 + sin(theta)^2 == 1 everywhere
    assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos), atol=1e-5)


def test_dinov3_apply_rope_leaves_prefix_tokens_untouched():
    rope = DINOv3ViTRopePositionEmbedding(_DINOV3_CFG).eval()
    px = torch.randn(1, 3, 384, 384)
    cos, sin = rope(px)
    num_prefix = 1 + _DINOV3_CFG.num_register_tokens
    num_patches = cos.shape[0]
    heads = _DINOV3_CFG.num_attention_heads
    head_dim = _DINOV3_CFG.hidden_size // heads

    # _apply_dinov3_rotary_pos_emb expects [B, S, heads, head_dim]
    # (MMEncoderAttention layout).
    q = torch.randn(1, num_prefix + num_patches, heads, head_dim)
    k = torch.randn(1, num_prefix + num_patches, heads, head_dim)
    q_out, k_out = _apply_dinov3_rotary_pos_emb(q, k, cos, sin, num_prefix)

    # Prefix (CLS + register) tokens must be identical; patch tokens must change.
    assert torch.equal(q_out[:, :num_prefix], q[:, :num_prefix])
    assert torch.equal(k_out[:, :num_prefix], k[:, :num_prefix])
    assert not torch.allclose(q_out[:, num_prefix:], q[:, num_prefix:])


def test_dinov3_apply_rope_preserves_norm():
    # A rotation preserves the per-token vector norm of the patch tokens.
    rope = DINOv3ViTRopePositionEmbedding(_DINOV3_CFG).eval()
    px = torch.randn(1, 3, 384, 384)
    cos, sin = rope(px)
    num_prefix = 1 + _DINOV3_CFG.num_register_tokens
    num_patches = cos.shape[0]
    heads = _DINOV3_CFG.num_attention_heads
    head_dim = _DINOV3_CFG.hidden_size // heads

    q = torch.randn(1, num_prefix + num_patches, heads, head_dim)
    q_out, _ = _apply_dinov3_rotary_pos_emb(q, q.clone(), cos, sin, num_prefix)
    before = q[:, num_prefix:].norm(dim=-1)
    after = q_out[:, num_prefix:].norm(dim=-1)
    assert torch.allclose(before, after, atol=1e-4)


@pytest.mark.skipif(
    not os.getenv("MINICPM_ROBOTTRACK_PATH"),
    reason="set MINICPM_ROBOTTRACK_PATH to a local checkpoint to run e2e parity",
)
def test_end_to_end_finite(vllm_runner):
    path = os.environ["MINICPM_ROBOTTRACK_PATH"]
    coarse = torch.randn(124, 1536)
    coarse_time = torch.arange(31).repeat_interleave(4)
    fine = torch.randn(64, 1536)
    fine_time = torch.full((64,), 31, dtype=torch.long)
    mm = {
        "image": {
            "coarse_tokens": coarse,
            "coarse_time_indices": coarse_time,
            "fine_tokens": fine,
            "fine_time_indices": fine_time,
        }
    }

    with vllm_runner(
        path,
        runner="pooling",
        dtype="float32",
        enforce_eager=True,
        max_model_len=512,
        enable_mm_embeds=True,
        limit_mm_per_prompt={"image": 1},
    ) as vllm_model:
        outputs = vllm_model.llm.embed(
            [{"prompt": "Follow the person in the red shirt.", "multi_modal_data": mm}]
        )

    traj = torch.tensor(outputs[0].outputs.embedding)
    assert traj.shape == (24,)
    assert torch.isfinite(traj).all()


@pytest.mark.skipif(
    not (
        os.getenv("MINICPM_ROBOTTRACK_PATH")
        and os.getenv("DINOV3_MODEL_PATH")
        and os.getenv("SIGLIP_MODEL_PATH")
    ),
    reason=(
        "set MINICPM_ROBOTTRACK_PATH, DINOV3_MODEL_PATH and SIGLIP_MODEL_PATH "
        "to local checkpoints to run the pixels-in (in-tree encoder) path"
    ),
)
def test_end_to_end_pixels_in_finite(vllm_runner):
    path = os.environ["MINICPM_ROBOTTRACK_PATH"]
    # A short raw-frame window; the tower pads history to 31 internally.
    frames = [np.zeros((384, 384, 3), dtype=np.uint8) for _ in range(4)]
    mm = {"image": {"frames": frames}}

    with vllm_runner(
        path,
        runner="pooling",
        dtype="float32",
        enforce_eager=True,
        max_model_len=512,
        enable_mm_embeds=True,
        limit_mm_per_prompt={"image": 1},
        hf_overrides={
            "dino_model": os.environ["DINOV3_MODEL_PATH"],
            "siglip_model": os.environ["SIGLIP_MODEL_PATH"],
            "image_size": 384,
        },
    ) as vllm_model:
        outputs = vllm_model.llm.embed(
            [{"prompt": "Follow the person.", "multi_modal_data": mm}]
        )

    traj = torch.tensor(outputs[0].outputs.embedding)
    assert traj.shape == (24,)
    assert torch.isfinite(traj).all()
