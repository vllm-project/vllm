# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant rotation matrix generation.

Pre-computes a fixed random orthogonal rotation matrix used by
``--kv-cache-dtype tq4`` to spread coordinate energy uniformly before
scalar quantization.  The rotation is data-oblivious (depends only on
head_dim and a fixed seed) and shared across all layers / heads.

Reference: Zandieh et al., "TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate", arXiv:2504.19874, 2025.
"""

from functools import lru_cache

import torch


@lru_cache(maxsize=4)
def _compute_rotation_cpu(head_dim: int, seed: int = 42) -> torch.Tensor:
    """Compute and cache a random orthogonal matrix on CPU.

    Uses QR decomposition of a random Gaussian matrix with a fixed seed
    for reproducibility.  Cached on CPU to avoid multi-GPU cross-device
    issues — callers move the result to the target device.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    gaussian = torch.randn(
        head_dim, head_dim, generator=gen, dtype=torch.float32
    )
    q, r = torch.linalg.qr(gaussian)
    # Ensure deterministic sign (Haar measure requires this correction).
    # Use torch.where instead of sign() to guarantee strictly ±1
    # (sign() returns 0 for zero inputs, which would zero out a column).
    d = torch.diag(r)
    ph = torch.where(d >= 0, 1.0, -1.0)
    q = q * ph.unsqueeze(0)
    return q


def get_tq4_rotation(
    head_dim: int,
    device: str = "cuda",
    seed: int = 42,
) -> torch.Tensor:
    """Return a [head_dim, head_dim] random orthogonal matrix on *device*.

    The rotation is computed and cached on CPU, then moved to the requested
    device on each call.  This avoids cross-device errors in multi-GPU
    setups where ``lru_cache`` would otherwise pin the tensor to whichever
    GPU was active on the first call.
    """
    cpu_rotation = _compute_rotation_cpu(head_dim, seed)
    return cpu_rotation.to(device=device, dtype=torch.float32)
