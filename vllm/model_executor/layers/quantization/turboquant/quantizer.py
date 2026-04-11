# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant quantizer utilities.

Serving path uses generate_wht_signs() for WHT rotation sign buffers.
generate_rotation_matrix() is retained for standalone benchmarks only.
Triton kernels handle all quantization, packing, and dequantization on GPU.
"""

import torch


def generate_rotation_matrix(
    d: int, seed: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Generate Haar-distributed random orthogonal matrix via QR decomposition."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen, device="cpu", dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    # Fix sign ambiguity for determinism
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


def generate_wht_signs(
    d: int, seed: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Generate deterministic random ±1 signs for WHT rotation.

    Used with Walsh-Hadamard Transform for per-layer rotation randomization.
    Same seed derivation as QR (per-layer via seed + layer_idx * stride).
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    bits = torch.randint(0, 2, (d,), generator=gen, device="cpu")
    signs = bits.float() * 2 - 1
    return signs.to(device)
