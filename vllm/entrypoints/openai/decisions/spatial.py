# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spatial decoding for the ``box`` and ``ground`` question types.

Both read the same per-patch distribution a GUI-Actor pointer head produces: a
softmax over the image's visual patches, scoring each patch against the
``<|pointer_pad|>`` query. The head neither regresses coordinates nor answers by
generating text, so the answer is decoded from the distribution:

- ``ground`` is the centre of the best connected region, activation-weighted.
- ``box`` is the per-axis marginal distribution's moment-matched edges.

The math is ported from Microsoft's GUI-Actor (MIT License, Copyright (c) 2025
Microsoft) via the reference serving layer, and must stay bit-compatible with
it, so a question answered here and there agrees. Two deliberate asymmetries,
inherited from the reference, must not be "fixed":

- The temperature is applied to the *confidence scalar only*, after the region
  or the box has been extracted from the raw distribution. Rescaling the
  distribution first would move the predicted point.
- The geometry helpers are device-agnostic: a served pass keeps the head's
  logits on CUDA, so no helper may allocate a tensor on the CPU and mismatch.
"""

import math
from collections import deque

import torch

#: Serialization precision for a probability mass. A 2,550-patch grid at 4
#: decimals can drift the sum by 0.13; 6 bounds it at 0.0013.
PATCH_PROB_DECIMALS = 6

#: Region centres carried by a ground answer besides the best one.
TOPK = 5

#: A patch passes the activation threshold at this fraction of the peak score.
DEFAULT_ACTIVATION_THRESHOLD = 0.3


def region_confidence(
    patch_logits: torch.Tensor,
    region_indices: torch.Tensor | list[int],
    temperature: float = 1.0,
) -> float:
    """Probability mass the temperature-scaled patch distribution puts on a region.

    ``patch_logits`` is one row of *raw* (pre-softmax) scores. Taking the raw
    scores rather than a distribution the bf16 forward pass has already rounded
    through a softmax is what lets the scalar be recomputed at any temperature
    without re-running the model. Subtracting the max is exactly softmax-shift
    invariant and keeps large logits from overflowing.
    """
    z = torch.as_tensor(patch_logits, dtype=torch.float64).detach()
    z = z - z.max()
    probs = torch.softmax(z / float(temperature), dim=-1)
    # float64 accumulation: in float32 a whole-grid region (hundreds of patches)
    # can sum to slightly more than 1, and a probability mass must never exceed
    # one.
    return min(
        1.0,
        float(probs[torch.as_tensor(region_indices, dtype=torch.long)].sum()),
    )


def get_prediction_region_point(
    attn_scores: torch.Tensor,
    n_width: int,
    n_height: int,
    activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    return_all_regions: bool = True,
    rect_center: bool = False,
    return_region_indices: bool = False,
):
    """Threshold the patch scores, split the survivors into 4-neighbour
    connected regions, and return the regions' weighted centres, best first.

    1. Select activated patches (> activation_threshold * max score)
    2. Divide connected patches into different regions
    3. Calculate the average activation value for each region
    4. Select the region with the highest average activation value
    5. Return the center point of that region as the final prediction point

    Patch ``idx`` sits at ``y = idx // n_width, x = idx % n_width``; a region's
    centre is the activation-weighted average of its patches' normalised
    centres, and regions are ranked by their mean activation. Coordinates are
    normalised to the model's (smart-resized) input.

    With ``return_region_indices`` the flat patch indices of every region are
    returned too (sorted alongside the centres and scores), which is what the
    confidence sums over.
    """
    if return_region_indices and not return_all_regions:
        raise ValueError("return_region_indices requires return_all_regions=True")
    # Get the highest activation value and threshold
    max_score = attn_scores[0].max().item()
    threshold = max_score * activation_threshold
    # Select all patches above the threshold
    mask = attn_scores[0] > threshold
    valid_indices = torch.nonzero(mask).squeeze(-1)
    topk_values = attn_scores[0][valid_indices]
    topk_indices = valid_indices

    # Convert indices to 2D coordinates
    topk_coords = []
    for idx in topk_indices.tolist():
        y = idx // n_width
        x = idx % n_width
        topk_coords.append((y, x, idx))

    # Divide into connected regions. ``index_of`` makes the neighbour lookup
    # O(1); scanning topk_coords inside the BFS made it O(k^2), which is seconds
    # per screenshot once a few thousand patches pass the threshold.
    regions = []
    visited = set()
    index_of = {idx: i for i, (_, _, idx) in enumerate(topk_coords)}
    for i, (y, x, idx) in enumerate(topk_coords):
        if idx in visited:
            continue

        # Start a new region
        region = [(y, x, idx, topk_values[i].item())]
        visited.add(idx)
        queue = deque([(y, x, idx, topk_values[i].item())])

        # BFS to find connected points
        while queue:
            cy, cx, c_idx, c_val = queue.popleft()

            # Check 4 adjacent directions
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < n_height and 0 <= nx < n_width):
                    continue
                n_idx = ny * n_width + nx

                # Check if this adjacent point is in the topk list
                j = index_of.get(n_idx)
                if j is not None and n_idx not in visited:
                    visited.add(n_idx)
                    region.append((ny, nx, n_idx, topk_values[j].item()))
                    queue.append((ny, nx, n_idx, topk_values[j].item()))

        regions.append(region)

    # Calculate the average activation value for each region
    region_scores = []
    region_centers = []
    region_points = []
    region_indices = []

    for region in regions:
        # Calculate average score for the region
        avg_score = sum(item[3] for item in region) / len(region)
        region_scores.append(avg_score)
        region_indices.append([item[2] for item in region])

        # Calculate normalized center coordinates for each patch, then take the
        # average
        normalized_centers = []
        weights = []
        y_coords = set()
        x_coords = set()

        for y, x, _, score in region:
            # Normalized coordinates of the center point for each patch
            center_y = (y + 0.5) / n_height
            center_x = (x + 0.5) / n_width
            normalized_centers.append((center_x, center_y))
            weights.append(score)

            y_coords.add(center_y)
            x_coords.add(center_x)

        region_points.append(normalized_centers)

        # Calculate the average of normalized coordinates as the region center
        if not rect_center:
            # Weighted average
            total_weight = sum(weights)
            weighted_x = (
                sum(nc[0] * w for nc, w in zip(normalized_centers, weights))
                / total_weight
            )
            weighted_y = (
                sum(nc[1] * w for nc, w in zip(normalized_centers, weights))
                / total_weight
            )
            avg_center_x, avg_center_y = weighted_x, weighted_y
        else:
            avg_center_x = sum(x_coords) / len(x_coords)
            avg_center_y = sum(y_coords) / len(y_coords)
        region_centers.append((avg_center_x, avg_center_y))

    # Select the region with the highest average activation value
    sorted_indices = sorted(
        range(len(region_scores)), key=lambda i: region_scores[i], reverse=True
    )
    sorted_scores = [region_scores[i] for i in sorted_indices]
    sorted_centers = [region_centers[i] for i in sorted_indices]
    sorted_points = [region_points[i] for i in sorted_indices]
    sorted_region_indices = [region_indices[i] for i in sorted_indices]
    best_point = sorted_centers[0]

    if return_all_regions:
        # 1. best_point: the center point of the region with the highest average
        #    activation value
        # 2. sorted_centers: every region's center point, by average activation,
        #    descending
        # 3. sorted_scores: every region's average activation, descending
        # 4. sorted_points: every patch's normalized center, by its region's
        #    average activation
        # 5. sorted_region_indices (opt-in): the flat patch indices of every
        #    region, same order
        if return_region_indices:
            return (
                best_point,
                sorted_centers,
                sorted_scores,
                sorted_points,
                sorted_region_indices,
            )
        return best_point, sorted_centers, sorted_scores, sorted_points
    return best_point


def to_999(x: float, y: float) -> tuple[int, int]:
    """Normalised [0,1] region-centre coordinates -> 0-999 integers (the Qwen-VL
    grounding convention: x across the width, y down the height, top-left
    origin)."""
    return min(999, max(0, round(x * 999))), min(999, max(0, round(y * 999)))


def ground_answer(
    attn_scores: torch.Tensor,
    patch_logits: torch.Tensor,
    n_width: int,
    n_height: int,
    topk: int = TOPK,
    temperature: float = 1.0,
) -> dict:
    """The Ground answer for one grounding pass: the connected region's
    activation-weighted centre as ``[x, y]`` in 0-999 integers.

    ``confidence`` is the probability mass the (temperature-scaled) patch
    distribution puts on the predicted region -- the fitted temperature only
    rescales it, never the point. ``topk_points`` are the other regions' centres,
    same 0-999 scale; ``topk=0`` drops them. Region extraction reads the *raw*
    distribution on CPU first (the decode calls ``.item()`` per activated patch),
    so the temperature cannot move the point.
    """
    point, points, _, _, region_indices = get_prediction_region_point(
        attn_scores.float().cpu(),
        n_width,
        n_height,
        return_all_regions=True,
        rect_center=False,
        return_region_indices=True,
    )
    top_region = [int(i) for i in region_indices[0]]
    answer = {
        "type": "ground",
        "point": list(to_999(point[0], point[1])),
        "confidence": round(
            region_confidence(patch_logits, top_region, temperature),
            PATCH_PROB_DECIMALS,
        ),
    }
    if topk:
        answer["topk_points"] = [list(to_999(x, y)) for x, y in points[:topk]]
    return answer


def _axis_edges(
    patch_logits: torch.Tensor, n_width: int, n_height: int, axis: str
) -> tuple[float, float]:
    """One axis's [lo, hi] edges from the patch distribution's marginal moments.

    The head was trained against a *uniform* distribution over the in-box
    patches, and a length-w uniform interval has sigma = w / sqrt(12) -- so
    w = 2 * sqrt(3) * sigma and the edges are ``mu -/+ sqrt(3) * sigma``. For one
    target this recovers its patch bounding box; for two separate targets the
    between-lump variance inflates sigma, so the box spans them (the behaviour
    the official Box shows on ambiguous targets).

    The ``(0.5/n)^2`` term keeps a near-delta distribution from collapsing to
    zero area, and the result is clamped to [0, 1] with a strict ``lo < hi``
    guarantee (``BoxAnswer`` requires it).
    """
    n = n_width if axis == "x" else n_height  # marginal length
    probs = torch.softmax(patch_logits.detach().float(), dim=-1)
    grid = probs.reshape(n_height, n_width)  # p(y, x): row = y, col = x
    marginal = grid.sum(dim=0 if axis == "x" else 1)  # collapse the other axis
    marginal = marginal / marginal.sum()
    # Patch i covers [i/n, (i+1)/n]; its centre is (i+0.5)/n.  Build on the
    # input's device (a served pass keeps the distribution on CUDA; a CPU tensor
    # here would mismatch).
    centres = (torch.arange(n, dtype=torch.float64, device=marginal.device) + 0.5) / n
    mu = float((marginal.double() * centres).sum())
    sigma = float(torch.sqrt(((centres - mu) ** 2 * marginal.double()).sum()))

    # Moment-match to the uniform in-box target.  A continuous interval of width
    # w has sigma = w / sqrt(12), so half-width = sqrt(3) * sigma; adding
    # (0.5/n)^2 under the root extends that to patch-cell boundaries and makes a
    # single patch span one cell rather than collapse (a uniform over k whole
    # cells of n has half-width sqrt(3*sigma^2 + (0.5/n)^2) = k / (2n), exact,
    # whereas the bare sqrt(3)*sigma only spans the cell centres).
    half = math.sqrt(3.0 * sigma**2 + (0.5 / n) ** 2)
    lo = min(max(mu - half, 0.0), 1.0)
    hi = min(max(mu + half, 0.0), 1.0)
    if lo >= hi:  # both clamped to the same bound
        lo, hi = (0.0, 1.0 / n) if hi == 0.0 else (1.0 - 1.0 / n, 1.0)
    return lo, hi


def box_from_patch_dist(
    patch_logits: torch.Tensor,
    n_width: int,
    n_height: int,
    temperature: float = 1.0,
) -> dict:
    """The Box answer for one grounding pass: four edges as fractions of the
    image, plus confidence.

    Edges: each axis's marginal distribution over the patch grid, moment-matched
    to the uniform in-box training target (``_axis_edges``) -- the same
    expectation the official Box documents ("each coordinate is the expected
    value of its edge's predicted position").

    Confidence: the probability mass the temperature-scaled patch distribution
    puts on the patches whose centres fall inside the box (``region_confidence``).
    Geometry ignores temperature -- the same convention as ``ground_answer`` --
    so the point never moves when calibration is retuned. Coordinates are
    normalised to the model's (smart-resized) input, like ``ground_answer``.
    """
    left, right = _axis_edges(patch_logits, n_width, n_height, "x")
    top, bottom = _axis_edges(patch_logits, n_width, n_height, "y")

    dev = patch_logits.device
    cx = (torch.arange(n_width, dtype=torch.float64, device=dev) + 0.5) / n_width
    cy = (torch.arange(n_height, dtype=torch.float64, device=dev) + 0.5) / n_height
    inside = torch.outer((cy >= top) & (cy <= bottom), (cx >= left) & (cx <= right))
    region = torch.nonzero(inside.reshape(-1)).squeeze(-1)

    return {
        "type": "box",
        "box": {
            "left": round(float(left), 6),
            "right": round(float(right), 6),
            "top": round(float(top), 6),
            "bottom": round(float(bottom), 6),
        },
        "confidence": round(
            region_confidence(patch_logits, region, temperature), PATCH_PROB_DECIMALS
        ),
    }


def spatial_answer(
    primitive: str,
    attn_scores: torch.Tensor,
    patch_logits: torch.Tensor,
    n_width: int,
    n_height: int,
    temperature: float = 1.0,
    topk: int = TOPK,
) -> dict:
    """Decode one question's primitive from a grounding pass's distribution."""
    if primitive == "ground":
        return ground_answer(
            attn_scores,
            patch_logits,
            n_width,
            n_height,
            topk=topk,
            temperature=temperature,
        )
    if primitive == "box":
        return box_from_patch_dist(
            patch_logits, n_width, n_height, temperature=temperature
        )
    raise ValueError(f"unknown spatial primitive {primitive!r}")
