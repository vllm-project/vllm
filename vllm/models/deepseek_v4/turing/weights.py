# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Turing (SM75) DeepSeek-V4 weight and KV-cache sizing helpers."""

from __future__ import annotations


def kv_bytes_per_token_fp16(nope_head_dim: int, rope_head_dim: int) -> int:
    """Bytes per token in the FP16 MLA KV cache (plain-row layout).

    SM75 has no FP8 tensor cores, so the compressed KV row is stored as
    ``nope_head_dim + rope_head_dim`` FP16 elements per token (the
    compressor output plus the rotary part).
    """
    return (nope_head_dim + rope_head_dim) * 2
