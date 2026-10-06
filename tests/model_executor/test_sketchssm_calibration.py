# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loading and validation of SketchSSM calibrations."""

import pytest
import torch

from vllm.model_executor.layers.mamba.sketchssm.calibration import (
    SketchSSMCalibration,
    load_sketchssm_calibration,
)
from vllm.model_executor.layers.mamba.sketchssm.mamba2 import Mamba2SketchSSM

WINDOW = 16


@pytest.fixture(autouse=True)
def _clear_cache():
    load_sketchssm_calibration.cache_clear()
    yield
    load_sketchssm_calibration.cache_clear()


def _save(tmp_path, data, name="calibration.pt"):
    torch.save(data, tmp_path / name)
    return str(tmp_path / name)


def _frames(**overrides):
    g = torch.Generator().manual_seed(0)
    ranks = torch.tensor([[0, 1, 5, 3], [2, 0, 0, 0]])
    frames = torch.linalg.qr(torch.randn(2, 2, 8, 8, generator=g))[0]
    return dict(frames=frames, m_table=ranks) | overrides


def _portable(L=3, groups=2, heads=8, K=16, V=8, max_rank=4):
    g = torch.Generator().manual_seed(1)
    omega = torch.linalg.qr(torch.randn(L, groups, K, K, generator=g))[0]
    omega = omega[..., : max_rank + 2, :].contiguous().float()
    scale = torch.rand(L, heads, 1, generator=g, dtype=torch.float64) + 0.1
    error = scale * torch.exp(-0.5 * torch.arange(max_rank + 3)) * 64
    return dict(
        format="sketchssm-calibration",
        schema_version=1,
        geometry=dict(
            key_dim=K, value_dim=V, groups=groups, window=WINDOW, erase=False
        ),
        max_rank=max_rank,
        omega=omega,
        curves=dict(joint_dot_sq_sum=error, joint_nstep=64),
        layer_ids=[1, 4, 6],
    )


def test_exported_frames(tmp_path):
    calibration = load_sketchssm_calibration(
        _save(tmp_path, _frames(layer_ids=torch.tensor([3, 7]))), None, WINDOW
    )
    assert calibration.num_layers == 2
    assert calibration.layer_index(7) == 1
    with pytest.raises(ValueError, match="no entry for layer 5"):
        calibration.layer_index(5)


@pytest.mark.parametrize(
    ("data", "mean_rank", "match"),
    [
        (_frames(frames=torch.ones(2, 2, 8, 8)), None, "not orthogonal"),
        (_frames(m_table=torch.full((2, 4), 9)), None, r"lie in \[0, K\]"),
        (_frames(), 4.0, "already fixed"),
        (_frames(), None, "layer_ids"),
    ],
)
def test_rejects_bad_files(tmp_path, data, mean_rank, match):
    with pytest.raises(ValueError, match=match):
        load_sketchssm_calibration(_save(tmp_path, data), mean_rank, WINDOW)


def test_portable_mean_rank_allocation(tmp_path):
    c, mean_rank = _portable(), 2.5
    path = _save(tmp_path, c)
    calibration = load_sketchssm_calibration(path, mean_rank, WINDOW)
    ranks, frames = calibration.ranks, calibration.frames
    assert ranks.shape == (3, 8) and frames.shape == (3, 2, 16, 16)
    assert calibration.layer_ids == (1, 4, 6)
    rank_cost, dense_cost = 16 + 8, 16 * 8
    used = (ranks.sum() * rank_cost + (ranks == 0).sum() * dense_cost).item()
    assert used <= rank_cost * mean_rank * ranks.numel()
    # Each group frame starts with the basis prefix up to its largest rank.
    for layer in range(3):
        for group in range(2):
            m = int(ranks[layer, group * 4 : (group + 1) * 4].max())
            prefix = frames[layer, group, :m].double()
            basis = c["omega"][layer, group, :m].double()
            torch.testing.assert_close(
                basis @ prefix.T @ prefix, basis, atol=1e-5, rtol=0
            )


def test_portable_default_mean_rank(tmp_path):
    path = _save(tmp_path, _portable())
    default = load_sketchssm_calibration(path, None, WINDOW)
    explicit = load_sketchssm_calibration(path, 8.0, WINDOW)
    assert torch.equal(default.ranks, explicit.ranks)


@pytest.mark.parametrize(
    ("head_dim", "state_size", "match"),
    [
        (32, 16, None),
        (32, 24, "power of two"),
        (32, 8, "at least 16"),
        (8, 16, "head dim"),
    ],
)
def test_mamba2_sketchssm_dims(head_dim, state_size, match):
    """Mamba-2 shapes the SketchSSM kernels cannot run are rejected up front."""
    frames = torch.eye(state_size).expand(1, 2, -1, -1)
    calibration = SketchSSMCalibration(
        frames, torch.tensor([[0, 1, 4, 5]], dtype=torch.int32), (0,)
    )
    args = (calibration, 0, 4, head_dim, 2, state_size, 3)
    if match is None:
        layer = Mamba2SketchSSM(*args)
        assert layer.u.shape == (3, state_size + 1 + 4 + 5, head_dim)
        return
    with pytest.raises(ValueError, match=match):
        Mamba2SketchSSM(*args)
