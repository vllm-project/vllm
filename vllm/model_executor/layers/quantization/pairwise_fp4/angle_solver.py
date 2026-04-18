# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import torch


def _heuristic(pairs: torch.Tensor, risk_scores: torch.Tensor) -> torch.Tensor:
    if risk_scores is None:
        raise ValueError("risk_scores is required for heuristic solver")
    eps = 1e-10
    ri = risk_scores[pairs[:, 0]]
    rj = risk_scores[pairs[:, 1]]
    theta = (math.pi / 4) * (ri - rj).abs() / (ri + rj + eps)
    return theta


def _small_search(pairs: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    n_angles = 16
    angles = torch.linspace(-math.pi / 4, math.pi / 4, n_angles)
    n = pairs.shape[0]
    best_angles = torch.zeros(n, dtype=torch.float32)

    for k in range(n):
        i = pairs[k, 0].item()
        j = pairs[k, 1].item()
        col_i = ref[..., i].to(torch.float32).reshape(-1)
        col_j = ref[..., j].to(torch.float32).reshape(-1)

        best_err = float("inf")
        best_a = 0.0
        for a_idx in range(n_angles):
            theta = angles[a_idx].item()
            c = math.cos(theta)
            s = math.sin(theta)
            ri = c * col_i + s * col_j
            rj = -s * col_i + c * col_j
            rotated = torch.cat([ri, rj])

            scale_val = rotated.abs().max().item() / 7.5
            if scale_val == 0.0:
                err = 0.0
            else:
                quantized = torch.round(rotated / scale_val) * scale_val
                err = (quantized - rotated).abs().sum().item()

            if err < best_err:
                best_err = err
                best_a = theta

        best_angles[k] = best_a

    return best_angles


def solve_angles(
    pairs: torch.Tensor,
    solver: str = "heuristic",
    weight: torch.Tensor | None = None,
    activation: torch.Tensor | None = None,
    risk_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    if solver not in ("heuristic", "small_search"):
        raise ValueError(
            f"Unknown solver '{solver}', must be 'heuristic' or 'small_search'"
        )

    if pairs.shape[0] == 0:
        return torch.empty(0, dtype=torch.float32)

    if solver == "heuristic":
        if risk_scores is None:
            raise ValueError("risk_scores is required for heuristic solver")
        return _heuristic(pairs, risk_scores.float()).float().cpu()

    # small_search
    if weight is None and activation is None:
        raise ValueError(
            "At least one of weight or activation is required for small_search solver"
        )
    ref = weight if weight is not None else activation
    return _small_search(pairs, ref).float().cpu()
