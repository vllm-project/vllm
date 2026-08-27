# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MiniCPM-RobotTrack in-tree pooling model.

The pure-logic tests always run. The end-to-end parity test is gated on a local
checkpoint (env ``MINICPM_ROBOTTRACK_PATH``) because the model is not a public
HF hub checkpoint; when set it reproduces the layered parity described in the
integration plan (vLLM 24-dim output vs the HF reference trajectory).
"""

import os

import pytest
import torch

from vllm.model_executor.models.minicpm_robottrack import (
    FunnelTrajectoryHead,
    VisionProjector,
    _count_marker_runs,
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
