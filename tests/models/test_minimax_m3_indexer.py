# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 indexer admission without allocating GPU tensors."""

import pytest

from vllm.models.minimax_m3.common import indexer
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("platform_enum", "capability", "topk", "fp8_supported", "fp8_impl"),
    [
        (PlatformEnum.CUDA, None, 16, False, None),
        (PlatformEnum.CUDA, (8, 0), 16, False, None),
        (PlatformEnum.CUDA, (8, 6), 16, False, None),
        (PlatformEnum.CUDA, (8, 9), 16, True, "triton"),
        (PlatformEnum.CUDA, (9, 0), 16, True, "triton"),
        (PlatformEnum.CUDA, (10, 0), 16, True, "msa"),
        (PlatformEnum.CUDA, (10, 3), 16, True, "msa"),
        (PlatformEnum.CUDA, (10, 0), 8, True, "triton"),
        (PlatformEnum.CUDA, (12, 0), 16, True, "triton"),
        (PlatformEnum.CUDA, (12, 1), 8, True, "triton"),
        (PlatformEnum.ROCM, (9, 4), 16, True, None),
        (PlatformEnum.CPU, None, 16, False, None),
        (PlatformEnum.XPU, None, 16, True, None),
        (PlatformEnum.OOT, None, 16, True, None),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        "bf16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "mxfp4",
        "nvfp4",
        "auto",
        "fp16",
        "unknown",
    ],
)
def test_indexer_admission_preserves_fallback_and_rejects_unsupported_fp8(
    monkeypatch, platform_enum, capability, topk, fp8_supported, fp8_impl, dtype
):
    platform = Platform()
    platform._enum = platform_enum
    monkeypatch.setattr(
        Platform,
        "get_device_capability",
        classmethod(
            lambda cls, device_id=0: DeviceCapability(*capability)
            if capability is not None
            else None
        ),
    )
    monkeypatch.setattr(
        Platform, "supports_fp8", classmethod(lambda cls: fp8_supported)
    )
    monkeypatch.setattr(indexer, "current_platform", platform)

    expected = fp8_impl
    if dtype == "bf16" and expected is None:
        expected = "triton"
    elif dtype not in ("bf16", "fp8", "fp8_e4m3"):
        expected = None
    if expected is None:
        reason = (
            "CuteDSL indexer impl"
            if dtype in ("mxfp4", "nvfp4")
            else "Triton indexer impl"
        )
        with pytest.raises(NotImplementedError, match=reason):
            indexer.select_indexer_impl_cls(topk_blocks=topk, indexer_kv_dtype=dtype)
    else:
        expected_cls = indexer.MiniMaxM3IndexerTritonImpl
        if expected == "msa":
            pytest.importorskip("cutlass")
            pytest.importorskip("cuda.bindings")
            pytest.importorskip("quack")
            from vllm.models.minimax_m3.nvidia.indexer_msa import (
                MiniMaxM3IndexerMSAImpl,
            )

            expected_cls = MiniMaxM3IndexerMSAImpl
        assert (
            indexer.select_indexer_impl_cls(topk_blocks=topk, indexer_kv_dtype=dtype)
            is expected_cls
        )
