# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence oracle for the q-side Hadamard rotation trick used by the
TurboQuant MLA backend.

Math
----
Store path produces packed indices for ``y = x_hat @ Pi^T`` (Pi is a
Walsh-Hadamard rotation, symmetric self-inverse). Decode reconstructs
``kv_c ≈ (y_hat @ Pi) * vec_norm``. The attention score for a query ``q`` is
therefore::

    score = q · kv_c = q · ((y_hat @ Pi) * vec_norm)
                     = ((q @ Pi) · y_hat) * vec_norm

The right-hand form lets us rotate ``q`` once per forward and feed the raw
``y_hat * vec_norm`` directly into the fused decode kernel (no per-token
K-side GEMM). This test pins that equivalence under the same bf16 tolerance
the MLA backend already uses, before we cut over production code.
"""

from __future__ import annotations

import pytest
import torch

from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard


def _quantize_to_indices(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    # Nearest-centroid lookup (Lloyd-Max); shape preserves x.
    diff = (x.unsqueeze(-1) - centroids.view(1, 1, -1)).abs()
    return diff.argmin(dim=-1)


@pytest.mark.parametrize("L", [128, 256, 512, 1024])
@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("norm_correction", [False, True])
def test_q_side_rotation_equivalence(L: int, bits: int, norm_correction: bool) -> None:
    """``q · ((y_hat @ Pi) * vn) == ((q @ Pi) · y_hat) * vn`` in bf16."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    device = torch.device("cuda")

    T = 64  # tokens
    H = 8  # heads (q only — kv_c is shared across heads in MLA)

    Pi = _build_hadamard(L, str(device)).to(torch.float32)
    centroids = get_centroids(L, bits).to(device=device, dtype=torch.float32)

    # Synthetic latent kv_c of shape (T, L), gaussian.
    kv_c = torch.randn(T, L, device=device, dtype=torch.float32)

    # Store-side: rotate, optionally norm-correct, then quantize.
    y = kv_c @ Pi  # Pi is symmetric, so Pi == Pi^T
    if norm_correction:
        vec_norm = y.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        y_unit = y / vec_norm
    else:
        vec_norm = torch.ones(T, 1, device=device, dtype=torch.float32)
        y_unit = y

    idx = _quantize_to_indices(y_unit, centroids)
    y_hat = centroids[idx]  # (T, L)

    # Random query.
    q = torch.randn(H, L, device=device, dtype=torch.float32)

    # k-side reference: dequant, rotate, then dot with q.
    k_recovered = (y_hat @ Pi) * vec_norm  # (T, L)
    score_kside = (q @ k_recovered.T).to(torch.bfloat16).to(torch.float32)

    # q-side fused: rotate q once, dot with raw y_hat, scale by vec_norm.
    q_rot = q @ Pi
    score_qside = ((q_rot @ y_hat.T) * vec_norm.T).to(torch.bfloat16).to(torch.float32)

    torch.testing.assert_close(score_qside, score_kside, atol=5e-3, rtol=5e-3)
    cos = torch.nn.functional.cosine_similarity(
        score_qside.flatten(), score_kside.flatten(), dim=0
    )
    assert cos.item() >= 0.999, f"cosine={cos.item():.6f}"
