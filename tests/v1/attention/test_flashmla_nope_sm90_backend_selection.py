# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-selection matrix for the rope-free NoPE zero-padded envelope on
SM90 (fp8_ds_mla NoPE-512 -> FLASHMLA_SPARSE).

These tests are CPU-eligible: they exercise the static backend acceptance
gates and the SM90 priority ordering without touching GPU kernels. On hosts
without the compiled CUDA extension, the priority test skips.
"""

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    QUANTIZED_DS_MLA_CACHE_FORMATS,
    FlashMLASparseBackend,
)

SM90 = DeviceCapability(major=9, minor=0)


def test_supported_head_sizes_include_nope_512():
    assert FlashMLASparseBackend.get_supported_head_sizes() == [576, 512]


@pytest.mark.parametrize(
    "head_size,kv_cache_dtype,accepted",
    [
        # NoPE-512 rides the zero-padded envelope only for quantized DS-MLA.
        (512, "fp8_ds_mla", True),
        # nvfp4_ds_mla is SM100-only, so it is rejected on SM90 regardless.
        (512, "nvfp4_ds_mla", False),
        # bf16/auto and plain fp8 NoPE-512 must keep flowing to
        # FlashInfer/TRITON backends (existing traffic unchanged).
        (512, "auto", False),
        (512, "bfloat16", False),
        (512, "fp8", False),
        (512, "fp8_e4m3", False),
        (512, None, False),
        # 576 traffic (DeepSeek family) is unchanged for every dtype.
        (576, "auto", True),
        (576, "bfloat16", True),
        (576, "fp8", True),
        (576, "fp8_ds_mla", True),
    ],
)
def test_supports_combination_nope_gate(head_size, kv_cache_dtype, accepted):
    reason = FlashMLASparseBackend.supports_combination(
        head_size,
        torch.bfloat16,
        kv_cache_dtype,
        64,
        use_mla=True,
        has_sink=False,
        use_sparse=True,
        use_mm_prefix=False,
        device_capability=SM90,
    )
    if accepted:
        assert reason is None, reason
    else:
        assert reason is not None


def test_nope_gate_only_quantized_ds_mla_formats():
    assert frozenset({"fp8_ds_mla", "nvfp4_ds_mla"}) == QUANTIZED_DS_MLA_CACHE_FORMATS


def test_supports_combination_nope_gate_sm100_nvfp4():
    """On SM100 the nvfp4_ds_mla NoPE-512 envelope is accepted (the SM90
    rejection above comes from the SM100-only check, not the NoPE gate)."""
    reason = FlashMLASparseBackend.supports_combination(
        512,
        torch.bfloat16,
        "nvfp4_ds_mla",
        64,
        use_mla=True,
        has_sink=False,
        use_sparse=True,
        use_mm_prefix=False,
        device_capability=DeviceCapability(major=10, minor=0),
    )
    assert reason is None, reason


def _sparse_order(kv_cache_dtype, head_size):
    try:
        from vllm.platforms.cuda import _get_backend_priorities
    except ImportError as e:
        pytest.skip(f"CUDA platform unavailable on this host: {e}")

    priorities = _get_backend_priorities(
        use_mla=True,
        device_capability=SM90,
        num_heads=32,
        kv_cache_dtype=kv_cache_dtype,
        head_size=head_size,
    )
    sparse = {
        "FLASH_ATTN_MLA_SPARSE",
        "FLASHMLA_SPARSE",
        "FLASHINFER_MLA_SPARSE_SM90",
    }
    return [b.name for b in priorities if b.name in sparse]


def test_sm90_nope_512_fp8_ds_mla_prefers_flashmla():
    order = _sparse_order("fp8_ds_mla", 512)
    assert order[0] == "FLASHMLA_SPARSE"
    assert order[1] == "FLASHINFER_MLA_SPARSE_SM90"


def test_sm90_nope_512_plain_fp8_keeps_flashinfer():
    order = _sparse_order("fp8", 512)
    assert order[0] == "FLASHINFER_MLA_SPARSE_SM90"
    assert "FLASHMLA_SPARSE" in order


def test_sm90_nope_512_bf16_unchanged():
    order = _sparse_order("auto", 512)
    assert order[0] == "FLASHINFER_MLA_SPARSE_SM90"


def test_sm90_576_unchanged_for_all_dtypes():
    for kv in ("auto", "fp8", "fp8_ds_mla"):
        order = _sparse_order(kv, 576)
        assert order == [
            "FLASH_ATTN_MLA_SPARSE",
            "FLASHMLA_SPARSE",
            "FLASHINFER_MLA_SPARSE_SM90",
        ], kv
