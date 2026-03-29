# SPDX-License-Identifier: Apache-2.0
"""Lloyd-Max optimal scalar quantizer for TurboQuant.

After rotating a d-dimensional unit vector by a random orthogonal matrix,
each coordinate approximately follows N(0, 1/d) for d >= 64.
We solve the Lloyd-Max conditions to find optimal centroids.

Based on: turboquant-pytorch/lloyd_max.py (Zandieh et al.)
"""

import math
from functools import lru_cache

import torch


def _gaussian_pdf(x: float, sigma2: float) -> float:
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(
        -x * x / (2 * sigma2)
    )


def solve_lloyd_max(
    d: int,
    bits: int,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve Lloyd-Max optimal quantizer for N(0, 1/d) distribution.

    Args:
        d: Vector dimension (determines variance = 1/d).
        bits: Number of quantization bits.
        max_iter: Maximum Lloyd-Max iterations.
        tol: Convergence tolerance.

    Returns:
        centroids: Sorted tensor of 2^bits optimal centroids.
        boundaries: Sorted tensor of 2^bits - 1 decision boundaries.
    """
    from scipy import integrate

    n_levels = 2**bits
    sigma2 = 1.0 / d
    sigma = math.sqrt(sigma2)

    def pdf(x):
        return _gaussian_pdf(x, sigma2)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            den, _ = integrate.quad(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])

        if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < tol:
            break
        centroids = new_centroids

    boundaries = [
        (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
    ]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


@lru_cache(maxsize=32)
def get_centroids(d: int, bits: int) -> torch.Tensor:
    """Get precomputed Lloyd-Max centroids (cached)."""
    centroids, _ = solve_lloyd_max(d, bits)
    return centroids


@lru_cache(maxsize=32)
def get_boundaries(d: int, bits: int) -> torch.Tensor:
    """Get precomputed Lloyd-Max boundaries (cached)."""
    _, boundaries = solve_lloyd_max(d, bits)
    return boundaries
