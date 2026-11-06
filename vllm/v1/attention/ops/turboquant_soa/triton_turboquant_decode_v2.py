# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optimized Triton TurboQuant decode attention (v2).

FLUTE-paper optimizations applied:
  1. Grouped Q heads: grid over (B, Hk, splits) instead of (B, Hq, splits).
     Each program loads BLOCK_M Q heads sharing a KV head into a 2D tile,
     enabling tl.dot on tensor cores (MFMA/WMMA) for both Q·K and P·V.
  2. Vectorized pair LUT: precompute `pair_table[i][j] = (T[i], T[j])`
     offline. At runtime, extract adjacent index pairs and fetch two
     dequantized centroids with a single gather, halving LUT lookups.
  3. exp2 instead of exp: scores pre-scaled by log2(e) so the hardware-
     native exp2 instruction replaces the more expensive exp.
  4. Wider index extraction: for 4-bit MSE, two adjacent 4-bit indices share
     a byte. One byte load yields both, eliminating redundant loads.
  5. Centroids pre-warmed in L1 at kernel start.
  6. BLOCK_KV = TILE_SIZE raised to 16-32 (from 4), reducing loop iterations
     and softmax rescaling overhead.

Stage 2 is reused unchanged from triton_decode_attention.py.
"""

import math

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# ROCm prefers num_stages=1 in attention-like kernels to reduce shared-memory
# pressure (mirrors the pattern used in triton_decode_attention.py and
# triton_turboquant_decode.py).
_is_hip = current_platform.is_rocm()

# On ROCm, bf16 has the same MFMA throughput as fp16 but wider dynamic range
# (8-bit exponent vs 5-bit), which is safer for attention scores.  On CUDA,
# fp16 tensor cores may be faster than bf16 for some shapes.
_DOT_DTYPE = tl.bfloat16 if _is_hip else tl.float16


# ---------------------------------------------------------------------------
# Pair LUT construction (FLUTE) — called once at launcher time
# ---------------------------------------------------------------------------


def build_pair_lut(centroids: torch.Tensor) -> torch.Tensor:
    """Build vectorized pair lookup table.

    For N centroids, returns a [N, N, 2] float32 tensor where
    pair_lut[i, j] = (centroids[i], centroids[j]).
    Flattened to [N*N, 2] for kernel indexing: pair_lut[i*N + j].

    For 4-bit MSE (N=16): 16*16*2*4 = 2048 bytes — fits in L1/smem.
    For 3-bit MSE (N=8):   8*8*2*4  = 512 bytes.
    """
    N = centroids.shape[0]
    # pair_lut[i,j,0] = centroids[i], pair_lut[i,j,1] = centroids[j]
    c = centroids.float()
    lut = torch.empty(N, N, 2, dtype=torch.float32, device=centroids.device)
    lut[:, :, 0] = c[:, None]
    lut[:, :, 1] = c[None, :]
    return lut.reshape(N * N, 2).contiguous()


# ---------------------------------------------------------------------------
# Launcher v2
# ---------------------------------------------------------------------------

_layout_cache: dict = {}


def _get_layout(D, mse_bits, value_quant_bits, key_packed_size):
    key = (D, mse_bits, value_quant_bits, key_packed_size)
    cfg = _layout_cache.get(key)
    if cfg is None:
        val_data_bytes = math.ceil(D * value_quant_bits / 8)
        cfg = {
            "mse_bytes": math.ceil(D * mse_bits / 8),
            "val_data_bytes": val_data_bytes,
            "mse_bits": mse_bits,
            "n_centroids": 2**mse_bits,
            "BLOCK_D": triton.next_power_of_2(D),
        }
        _layout_cache[key] = cfg
    return cfg


def _get_pair_lut(centroids: torch.Tensor) -> torch.Tensor:
    """Return a fresh pair-LUT for ``centroids`` on each call.

    The LUT is tiny (N*N*2 fp32, e.g. 2KB for 4-bit MSE) so the build cost
    is negligible compared to attention. We avoid caching by data_ptr()
    because CUDA allocator memory reuse across different centroid tensors
    can silently return a stale LUT (subtle correctness bug). If this ever
    shows up on a profile, cache by a hash-of-values fingerprint instead.
    """
    return build_pair_lut(centroids)
