# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant quantizer utilities.

Serving path uses generate_wht_signs() for WHT rotation sign buffers.
Triton kernels handle all quantization, packing, and dequantization on GPU.
"""

import torch

_CPU = torch.device("cpu")


def generate_wht_signs(d: int, seed: int, device: torch.device = _CPU) -> torch.Tensor:
    """Generate deterministic random ±1 signs for WHT rotation.

    Used with Walsh-Hadamard Transform for per-layer rotation randomization.
    Same seed derivation as QR (per-layer via seed + layer_idx * stride).
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    bits = torch.randint(0, 2, (d,), generator=gen, device="cpu")
    signs = bits.float() * 2 - 1
    return signs.to(device)
