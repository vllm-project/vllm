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
def get_tq4_rotation(
    head_dim: int,
    device: str = "cuda",
    seed: int = 42,
) -> torch.Tensor:
    """Return a [head_dim, head_dim] random orthogonal matrix.

    Uses QR decomposition of a random Gaussian matrix with a fixed seed
    for reproducibility.  The result is cached so it is computed only once
    per (head_dim, device, seed) combination.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    gaussian = torch.randn(
        head_dim, head_dim, generator=gen, dtype=torch.float32
    )
    q, r = torch.linalg.qr(gaussian)
    # Ensure deterministic sign (Haar measure requires this correction)
    d = torch.diag(r)
    ph = d.sign()
    q = q * ph.unsqueeze(0)
    return q.to(device=device, dtype=torch.float32)
