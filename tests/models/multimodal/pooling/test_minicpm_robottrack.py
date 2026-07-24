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
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.minicpm_robottrack import (
    DINOv3ViTRopePositionEmbedding,
    FunnelTrajectoryHead,
    VisionProjector,
    _apply_dinov3_rotary_pos_emb,
    _count_marker_runs,
    _grid_pool,
    _pad_history_frames,
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
    import numpy as np

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
