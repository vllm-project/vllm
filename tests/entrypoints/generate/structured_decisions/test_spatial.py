# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-function tests for the box/ground decode (no GPU, no checkpoint).

The GUI-Actor pointer head was trained against a uniform distribution over the
in-box patches, so box edges are recovered from the patch distribution's
per-axis marginal moments: half-width = sqrt(3) * sigma (a length-w uniform
interval has sigma = w / sqrt(12)). The ground point is the best connected
region's activation-weighted centre.
"""

import pytest
import torch

from vllm.entrypoints.openai.decisions.spatial import (
    box_from_patch_dist,
    get_prediction_region_point,
    ground_answer,
    region_confidence,
    to_999,
)


def dist(n_w: int, n_h: int, rects):
    """A patch distribution with mass uniformly spread over the given inclusive
    rectangles.

    Returns ``(attn_scores [1, n_patches], patch_logits [n_patches], n_w, n_h)``;
    the logits are the log of that distribution, so a temperature-1 softmax
    reproduces it exactly.
    """
    p = torch.zeros(n_h, n_w)
    for c0, c1, r0, r1 in rects:
        p[r0 : r1 + 1, c0 : c1 + 1] = 1.0
    p = p / p.sum()
    return p.reshape(1, -1), torch.log(p.clamp_min(1e-30)).reshape(-1), n_w, n_h


def box_args(n_w: int, n_h: int, rects):
    """``dist`` without the attention scores, which only the ground decode uses."""
    _, patch_logits, width, height = dist(n_w, n_h, rects)
    return patch_logits, width, height


def test_uniform_rect_recovers_edges():
    # columns 2..7 of 16 -> [0.125, 0.5]; rows 4..9 of 12 -> [0.333.., 0.833..]
    a = box_from_patch_dist(*box_args(16, 12, [(2, 7, 4, 9)]))
    assert a["type"] == "box"
    assert a["box"]["left"] == pytest.approx(2 / 16, abs=0.02)
    assert a["box"]["right"] == pytest.approx(8 / 16, abs=0.02)
    assert a["box"]["top"] == pytest.approx(4 / 12, abs=0.02)
    assert a["box"]["bottom"] == pytest.approx(10 / 12, abs=0.02)
    assert a["confidence"] > 0.9  # (almost) all the mass lies inside the box


def test_single_patch_does_not_collapse():
    # sigma = 0 would give a zero-area box; the one-patch floor keeps
    # left < right.
    a = box_from_patch_dist(*box_args(16, 12, [(5, 5, 6, 6)]))
    assert a["box"]["left"] < a["box"]["right"]
    assert a["box"]["top"] < a["box"]["bottom"]
    assert a["box"]["right"] - a["box"]["left"] == pytest.approx(1 / 16, rel=0.5)
    assert a["box"]["bottom"] - a["box"]["top"] == pytest.approx(1 / 12, rel=0.5)


def test_bimodal_spreads_wide():
    # Two equal targets at opposite sides: the between-lump variance inflates
    # sigma and the box spans them (the behaviour the official Box shows on two
    # identical targets).
    one = box_from_patch_dist(*box_args(16, 12, [(6, 9, 4, 7)]))
    two = box_from_patch_dist(*box_args(16, 12, [(0, 2, 4, 7), (13, 15, 4, 7)]))
    assert two["box"]["right"] - two["box"]["left"] > 0.8
    assert two["box"]["right"] - two["box"]["left"] > 3 * (
        one["box"]["right"] - one["box"]["left"]
    )


def test_edges_are_clamped_to_unit_interval():
    a = box_from_patch_dist(*box_args(16, 12, [(0, 1, 0, 1)]))
    assert a["box"]["left"] == 0.0 and a["box"]["top"] == 0.0
    assert 0.0 <= a["box"]["right"] <= 1.0 and 0.0 <= a["box"]["bottom"] <= 1.0


def test_temperature_scales_confidence_not_geometry():
    attn, logits, n_w, n_h = dist(16, 12, [(4, 7, 3, 6)])
    cold = box_from_patch_dist(logits, n_w, n_h, temperature=0.5)
    warm = box_from_patch_dist(logits, n_w, n_h, temperature=2.0)
    assert cold["box"] == warm["box"]  # the point/edges never move
    assert cold["confidence"] >= warm["confidence"]
    assert 0.0 <= cold["confidence"] <= 1.0
    assert 0.0 <= warm["confidence"] <= 1.0


def test_full_image_uniform_is_a_valid_box():
    a = box_from_patch_dist(*box_args(8, 8, [(0, 7, 0, 7)]))
    assert a["box"]["left"] == 0.0 and a["box"]["right"] == 1.0
    assert a["confidence"] == pytest.approx(1.0)


def test_ground_point_is_the_region_centre_on_the_999_scale():
    # rows 0..5, columns 0..7 of a 16x8 grid: centre (x=0.25, y=0.375)
    a = ground_answer(*dist(16, 8, [(0, 7, 0, 5)]), topk=0)
    assert a["type"] == "ground"
    assert a["point"] == [round(0.25 * 999), round(0.375 * 999)]
    assert a["confidence"] > 0.9
    assert "topk_points" not in a


def test_ground_topk_lists_other_regions():
    two_regions = dist(16, 12, [(0, 2, 0, 3), (12, 15, 8, 11)])
    a = ground_answer(*two_regions, topk=5)
    assert len(a["topk_points"]) == 2
    for point in [a["point"], *a["topk_points"]]:
        assert all(0 <= coordinate <= 999 for coordinate in point)


def test_ground_point_does_not_move_with_temperature():
    attn, logits, n_w, n_h = dist(16, 12, [(4, 7, 3, 6)])
    cold = ground_answer(attn, logits, n_w, n_h, temperature=0.5)
    warm = ground_answer(attn, logits, n_w, n_h, temperature=2.0)
    assert cold["point"] == warm["point"]
    assert cold["confidence"] >= warm["confidence"]


def test_to_999_clamps_to_the_scale():
    assert to_999(-0.5, 1.5) == (0, 999)
    assert to_999(0.0, 1.0) == (0, 999)
    assert to_999(0.5, 0.5) == (500, 500)


def test_region_confidence_never_exceeds_one_on_a_whole_grid_region():
    # float32 accumulation of a whole-grid region can sum to slightly more than
    # 1; a probability mass must never exceed one.
    n_w, n_h = 48, 48
    logits = torch.zeros(n_h * n_w)
    every_patch = torch.arange(n_h * n_w)
    assert region_confidence(logits, every_patch, temperature=1.0) == pytest.approx(1.0)


def test_get_prediction_region_point_ranks_regions_by_mean_activation():
    # Two disconnected regions, unequal in size, so the mean activation (not the
    # patch count) decides the ranking.
    attn, _, n_w, n_h = dist(16, 12, [(0, 1, 0, 1), (10, 13, 6, 9)])
    best, centers, scores, _ = get_prediction_region_point(
        attn, n_w, n_h, return_all_regions=True, rect_center=False
    )
    assert len(centers) == 2
    assert best == centers[0]
    assert scores[0] == max(scores)
    # Both centres land inside their own region.
    assert centers[0][0] < 0.2 or centers[0][0] > 0.6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_helpers_are_device_agnostic():
    # the head's logits arrive on CUDA in a served pass; no helper may allocate
    # on CPU and mismatch.
    attn, logits, n_w, n_h = dist(16, 12, [(4, 7, 3, 6)])
    a = box_from_patch_dist(logits.cuda(), n_w, n_h)
    assert 0.0 <= a["box"]["left"] < a["box"]["right"] <= 1.0
    assert 0.0 <= a["confidence"] <= 1.0
