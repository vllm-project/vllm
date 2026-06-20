# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OSCAR: INT2 KV-cache quantization for vLLM.

OSCAR (Offline Spectral Covariance-Aware Rotation) quantizes the KV cache
to ~2 bits per element using *calibrated*, data-dependent per-layer
rotations followed by clipped INT2 scalar quantization. Unlike data-free
Hadamard schemes (e.g. QuaRot, TurboQuant), the rotation is fit offline
from attention-aware K/V covariance estimates so that quantization noise
lands in the directions attention is least sensitive to.

The rotation matrices are an auxiliary, weight-preserving calibration
artifact (per-layer ``[head_dim, head_dim]`` orthogonal matrices) — the
model weights themselves are loaded unchanged from the original
checkpoint.

Reference: Zhou et al., "OSCAR: Offline Spectral Covariance-Aware
Rotation for 2-bit KV Cache Quantization" (arXiv:2605.17757). The
reference implementation targets SGLang; this is a vLLM-native port that
reuses the TurboQuant KV-cache-backend integration pattern.
"""

from vllm.model_executor.layers.quantization.oscar.config import OscarConfig

__all__ = ["OscarConfig"]
