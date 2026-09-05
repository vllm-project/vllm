# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Architecture and opt-in gating for the gfx942 compressor ops."""

import torch

from tests.kernels.attention.dsv4_compress_utils import detect_gfx942
from vllm.models.deepseek_v4.amd.ops import hip_compress_dispatch as dispatch
from vllm.models.deepseek_v4.amd.ops.hip_compress_dispatch import (
    SUPPORTED_SHAPES,
    hip_compressor_enabled,
    hip_compressor_selected,
    hip_compressor_supported,
)


def test_compressor_opt_in(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_DSV4_HIP_COMPRESSOR", "0")
    assert not hip_compressor_selected(512, 4)
    monkeypatch.setenv("VLLM_ROCM_DSV4_HIP_COMPRESSOR", "1")
    assert hip_compressor_selected(512, 4)
    assert hip_compressor_selected(512, 128)
    assert not hip_compressor_selected(128, 4)


def test_compressor_construction_gates(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_DSV4_HIP_COMPRESSOR", "1")
    monkeypatch.setattr(dispatch, "hip_compressor_available", lambda *_: True)

    assert hip_compressor_enabled(512, 64, 4, "fp8_ds_mla")
    assert not hip_compressor_enabled(512, 32, 4, "fp8_ds_mla")
    assert not hip_compressor_enabled(512, 64, 4, "fp8")

    monkeypatch.setattr(dispatch, "hip_compressor_available", lambda *_: False)
    assert not hip_compressor_enabled(512, 64, 4, "fp8_ds_mla")


def test_compressor_gated_on_gfx942():
    on = detect_gfx942()
    if on:
        import vllm._rocm_C  # noqa: F401  (ensure the ops are registered)

    u8 = torch.empty(1, 16, dtype=torch.uint8)

    for head_dim, ratio in SUPPORTED_SHAPES:
        assert hip_compressor_supported(head_dim, ratio, u8) == on

    assert not hip_compressor_supported(128, 4, u8)
    assert not hip_compressor_supported(256, 4, u8)
    bf16 = torch.empty(1, 16, dtype=torch.bfloat16)
    assert not hip_compressor_supported(512, 4, bf16)
