# SPDX-License-Identifier: Apache-2.0
"""TurboQuant quantizer utilities.

The only function used by vLLM's serving path is generate_rotation_matrix().
Triton kernels (triton_tq_store.py / triton_tq_decode.py) handle all
quantization, packing, and dequantization on GPU.
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
