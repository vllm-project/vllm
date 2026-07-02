# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Arch-gating contract for the AOT DeepSeek-V4 compressor ops.

Platform-agnostic — runs on any host (including non-ROCm CI). It asserts that
``hip_compressor_supported`` is True exactly when the device is gfx950 (CDNA4)
with the _rocm_C ops built in, and always False otherwise (wrong arch, an
unsupported/default-disabled shape, or a non-uint8 cache layout). The model
relies on this gate to choose between the HIP op and the Triton fallback.
"""

import pytest
import torch

from tests.kernels.attention.dsv4_compress_utils import detect_gfx950
from vllm.models.deepseek_v4.amd.ops.hip_compress_dispatch import (
    ALL_HIP_COMPRESSOR_SHAPES,
    DEFAULT_HIP_COMPRESSOR_SHAPES,
    hip_compressor_supported,
    parse_hip_compressor_modes,
)


def test_parse_hip_compressor_modes():
    assert parse_hip_compressor_modes(None) == frozenset()
    assert parse_hip_compressor_modes("0") == frozenset()
    assert parse_hip_compressor_modes("false") == frozenset()
    assert parse_hip_compressor_modes("1") == frozenset({"csa", "hca"})
    assert parse_hip_compressor_modes("true") == frozenset({"csa", "hca"})
    assert parse_hip_compressor_modes("csa,indexer") == frozenset({"csa", "indexer"})

    with pytest.raises(ValueError, match="VLLM_ROCM_DSV4_HIP_COMPRESSOR"):
        parse_hip_compressor_modes("csa,unknown")


def test_compressor_gated_on_gfx950():
    on = detect_gfx950()
    if on:
        import vllm._rocm_C  # noqa: F401  (ensure the ops are registered)

    u8 = torch.empty(1, 16, dtype=torch.uint8)

    assert (128, 4) not in DEFAULT_HIP_COMPRESSOR_SHAPES
    assert (128, 4) in ALL_HIP_COMPRESSOR_SHAPES

    # Default enabled mode supports CSA/HCA iff on gfx950.
    for head_dim, ratio in DEFAULT_HIP_COMPRESSOR_SHAPES:
        assert hip_compressor_supported(head_dim, ratio, u8) == on

    # Indexer is HIP-supported only when explicitly included by the caller.
    assert not hip_compressor_supported(128, 4, u8)
    assert (
        hip_compressor_supported(128, 4, u8, allowed_shapes=ALL_HIP_COMPRESSOR_SHAPES)
        == on
    )

    # Unsupported (head_dim, compress_ratio) -> never supported.
    assert not hip_compressor_supported(256, 4, u8)
    # Non-uint8 (e.g. FlashInfer full-cache) layout -> never supported.
    bf16 = torch.empty(1, 16, dtype=torch.bfloat16)
    assert not hip_compressor_supported(512, 4, bf16)
