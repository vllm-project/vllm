# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-head rank allocation and frames of a portable SketchSSM calibration."""

from typing import Any

import numpy as np
import torch

from vllm.utils.torch_utils import set_default_torch_num_threads

FORMAT = "sketchssm-calibration"
SCHEMA_VERSION = 1
# LAPACK QR results depend on the CPU thread count.
EXPORT_THREADS = 2
# Bisection steps on the Lagrange multiplier of the rank budget.
LAGRANGIAN_STEPS = 160


def is_portable_calibration(data: dict[str, Any]) -> bool:
    return data.get("format") == FORMAT


def validate(c: dict[str, Any]) -> None:
    if c.get("format") != FORMAT:
        raise ValueError(f"Not a {FORMAT} file")
    if c.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported calibration schema: {c.get('schema_version')}")
    g, omega = c["geometry"], c["omega"]
    if (
        omega.ndim != 4
        or omega.shape[1] != g["groups"]
        or omega.shape[-1] != g["key_dim"]
    ):
        raise ValueError("Basis shape differs from the declared geometry")
    error = c["curves"]["joint_dot_sq_sum"]
    if (
        error.ndim != 3
        or error.shape[0] != omega.shape[0]
        or error.shape[1] % g["groups"]
    ):
        raise ValueError("Score curve shape differs from the basis")


def _solve(objective: np.ndarray, cost: np.ndarray, budget: float) -> np.ndarray:
    """Pick one option per head (the last is dense) under a cost budget.

    A multiple-choice knapsack: Lagrangian relaxation with bisection on the
    multiplier, then a greedy fill of the remaining budget.
    """
    heads, nchoice = objective.shape
    rows = np.arange(heads)
    scaled = objective / max(float(np.mean(np.abs(objective))), 1e-30)

    def lagrangian(lam):
        penalized = scaled + lam * cost[None, :]
        chosen = penalized.argmin(axis=1)
        dual = float(penalized[rows, chosen].sum() - lam * budget)
        return (
            dual,
            float(cost[chosen].sum()),
            float(scaled[rows, chosen].sum()),
            chosen,
        )

    uniform = int(min(nchoice - 1, max(1, np.floor(budget / (heads * cost[0])))))
    best_choice = np.full(heads, uniform - 1, dtype=np.int64)
    best = float(scaled[rows, best_choice].sum())

    lo, hi = 0.0, 1.0
    while lagrangian(hi)[1] > budget:
        hi *= 2.0
    best_dual, best_lam = -np.inf, hi

    def visit(lam):
        nonlocal best_dual, best_lam, best_choice, best
        dual, used, primal, chosen = lagrangian(lam)
        if dual > best_dual:
            best_dual, best_lam = dual, lam
        if used <= budget and primal < best:
            best_choice, best = chosen.copy(), primal
        return used

    for _ in range(LAGRANGIAN_STEPS):
        lam = (lo + hi) / 2.0
        if visit(lam) > budget:
            lo = lam
        else:
            hi = lam
    for lam in (lo, hi, best_lam):
        visit(lam)

    # Greedily spend the remaining budget on the best single-head upgrade.
    choice = best_choice.copy()
    used = float(cost[choice].sum())
    while True:
        delta = cost[None, :] - cost[choice][:, None]
        gain = scaled[rows, choice][:, None] - scaled
        ok = (delta > 0) & (delta <= budget - used + 1e-9) & (gain > 0)
        if not np.any(ok):
            return choice
        head, option = np.unravel_index(
            np.argmax(np.where(ok, gain, -np.inf)), gain.shape
        )
        used += float(delta[head, option])
        choice[head] = option


def crossover(geometry: dict[str, Any], window: int) -> int:
    """Dense crossover rank (the default rank cap) of a geometry at a window."""
    K, V = geometry["key_dim"], geometry["value_dim"]
    return min(K, K * V // (K + V + (window if geometry["erase"] else 0)))


def max_rank_at(c: dict[str, Any], window: int) -> int:
    """Rank cap at a serving window."""
    g = c["geometry"]
    if window == g["window"]:
        return c["max_rank"]
    cap = crossover(g, window)
    if c["max_rank"] < crossover(g, g["window"]):
        cap = min(cap, c["max_rank"])
    available = c["curves"]["joint_dot_sq_sum"].shape[-1] - 1
    return min(cap, c["omega"].shape[-2], available)


def allocate(c: dict[str, Any], mean_rank: float, window: int) -> torch.Tensor:
    """Per-head sketch ranks (0 = dense) for a mean rank at a serving window."""
    g = c["geometry"]
    K, V, erase = g["key_dim"], g["value_dim"], g["erase"]
    W = window
    max_rank = max_rank_at(c, W)
    curves = c["curves"]
    error = torch.as_tensor(curves["joint_dot_sq_sum"]).cpu().double()
    error = error / int(curves["joint_nstep"])
    L, H, available = error.shape
    rank_cost = K + V + (W if erase else 0)
    dense_cost = K * V
    if not 1 <= max_rank <= crossover(g, W):
        raise ValueError("SketchSSM max_rank must lie within the dense crossover")
    if max_rank >= available or max_rank > c["omega"].shape[-2]:
        raise ValueError("Requested rank cap exceeds the calibrated curves or basis")
    total_budget = rank_cost * mean_rank * L * H
    ranks = np.arange(1, max_rank + 1, dtype=np.int64)
    costs = np.r_[ranks * rank_cost, dense_cost].astype(np.float64)
    objective = torch.cat(
        [
            error[:, :, 1 : max_rank + 1].reshape(-1, max_rank),
            torch.zeros(L * H, 1, dtype=torch.float64),
        ],
        1,
    ).numpy()
    choice = _solve(objective, costs, total_budget)
    dense = choice == max_rank
    m = np.zeros(L * H, dtype=np.int64)
    m[~dense] = ranks[choice[~dense]]
    return torch.from_numpy(m).reshape(L, H).to(torch.int32)


def complete_frame(rows: torch.Tensor) -> torch.Tensor:
    """Complete ordered prefix rows to an orthogonal state rotation."""
    state_dim = rows.shape[1]
    augmented = torch.cat(
        [rows.T.double(), torch.eye(state_dim, dtype=torch.float64)], dim=1
    )
    return torch.linalg.qr(augmented)[0].T.contiguous()


def export_frames(omega: torch.Tensor, m_table: torch.Tensor) -> torch.Tensor:
    """Orthogonal frames ``(layers, groups, K, K)`` from the basis prefixes."""
    omega = omega.double()
    L, groups, _, K = omega.shape
    m = m_table.long().reshape(L, groups, -1).amax(-1)
    frames = torch.empty(L, groups, K, K, dtype=torch.float32)
    for layer in range(L):
        for group in range(groups):
            rows = omega[layer, group, : int(m[layer, group])]
            frames[layer, group] = complete_frame(rows).float()
    return frames


def frames_for_mean_rank(
    c: dict[str, Any], mean_rank: float, window: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frames and per-head ranks of a portable calibration for one mean rank."""
    with torch.device("cpu"), set_default_torch_num_threads(EXPORT_THREADS):
        validate(c)
        ranks = allocate(c, float(mean_rank), window)
        return export_frames(c["omega"], ranks), ranks
