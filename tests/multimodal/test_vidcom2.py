# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.multimodal.vidcom2 import (
    compute_retained_tokens_count,
    compute_retention_mask,
)


def _fake_video_embeds(
    num_frames: int,
    rows: int,
    cols: int,
    hidden: int = 64,
    seed: int = 0,
) -> torch.Tensor:
    """Deterministic fake ViT output with a distinct mean per frame."""
    g = torch.Generator().manual_seed(seed)
    frames = []
    for f in range(num_frames):
        base = torch.randn(hidden, generator=g) * (0.1 + 0.05 * f)
        frames.append(
            base[None, :].expand(rows * cols, hidden)
            + 0.01 * torch.randn(rows * cols, hidden, generator=g)
        )
    return torch.cat(frames, dim=0)


@pytest.mark.parametrize("q", [0.25, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("num_frames", [1, 4, 16])
def test_mask_shape_and_dtype(q: float, num_frames: int) -> None:
    merge = 2
    rows, cols = 6, 8
    embeds = _fake_video_embeds(num_frames, rows, cols)
    mask = compute_retention_mask(
        embeds,
        (num_frames, rows * merge, cols * merge),
        spatial_merge_size=merge,
        q=q,
    )
    assert mask.dtype == torch.bool
    assert mask.shape == (num_frames * rows * cols,)


def test_retained_count_floors_at_one_token_per_frame() -> None:
    """The global minimum is one token per frame (not a full first frame)."""
    assert (
        compute_retained_tokens_count(tokens_per_frame=48, num_frames=4, q=0.999) == 4
    )
    assert (
        compute_retained_tokens_count(tokens_per_frame=48, num_frames=4, q=0.0)
        == 48 * 4
    )


@pytest.mark.parametrize("q", [0.25, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("num_frames", [1, 4, 16])
def test_total_retained_matches_target(q: float, num_frames: int) -> None:
    """Mask total must equal the placeholder-sizing helper."""
    merge = 2
    rows, cols = 6, 8
    tpf = rows * cols
    embeds = _fake_video_embeds(num_frames, rows, cols)
    mask = compute_retention_mask(
        embeds,
        (num_frames, rows * merge, cols * merge),
        spatial_merge_size=merge,
        q=q,
    )
    expected = compute_retained_tokens_count(
        tokens_per_frame=tpf, num_frames=num_frames, q=q
    )
    assert int(mask.sum().item()) == expected


def test_per_frame_min_one_when_budget_allows() -> None:
    """No frame is fully dropped when the budget allows."""
    merge = 2
    rows, cols = 6, 8
    num_frames = 8
    embeds = _fake_video_embeds(num_frames, rows, cols)
    mask = compute_retention_mask(
        embeds,
        (num_frames, rows * merge, cols * merge),
        spatial_merge_size=merge,
        q=0.25,
    )
    per_frame = mask.view(num_frames, rows * cols).sum(dim=1)
    assert (per_frame >= 1).all(), f"zero-token frame detected: {per_frame.tolist()}"


def test_dynamic_per_frame_budget() -> None:
    """A distinctive frame gets more retained tokens than bland ones."""
    merge = 2
    rows, cols = 6, 8
    tpf = rows * cols
    hidden = 64
    torch.manual_seed(0)
    bland = 0.01 * torch.randn(tpf, hidden)
    frames = [torch.randn(tpf, hidden) * 1.0]
    for _ in range(7):
        frames.append(bland + 0.001 * torch.randn(tpf, hidden))
    embeds = torch.cat(frames, dim=0)
    mask = compute_retention_mask(
        embeds,
        (8, rows * merge, cols * merge),
        spatial_merge_size=merge,
        q=0.5,
    )
    per_frame = mask.view(8, tpf).sum(dim=1)
    assert per_frame[0].item() > per_frame[1:].float().mean().item()


def test_empty_input_safe() -> None:
    embeds = torch.zeros(0, 32)
    mask = compute_retention_mask(embeds, (0, 0, 0), spatial_merge_size=2, q=0.25)
    assert mask.numel() == 0


@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75])
def test_first_frame_not_privileged(q: float) -> None:
    """A bland first frame is not force-retained (unlike EVS)."""
    merge = 2
    rows, cols = 6, 8
    tpf = rows * cols
    torch.manual_seed(1)
    bland = 0.01 * torch.randn(tpf, 64)
    frames = [bland]
    for f in range(7):
        frames.append(torch.randn(tpf, 64) * (1.0 + 0.1 * f))
    embeds = torch.cat(frames, dim=0)
    mask = compute_retention_mask(
        embeds,
        (8, rows * merge, cols * merge),
        spatial_merge_size=merge,
        q=q,
    )
    per_frame = mask.view(8, tpf).sum(dim=1)
    assert per_frame[0].item() <= tpf
    if q > 0.0:
        assert per_frame[0].item() < int(mask.sum().item())
