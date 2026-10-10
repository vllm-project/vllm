# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validation wiring in `XPUPlatform.get_attn_backend_cls`.

Unlike CUDA and ROCm, the XPU platform used to return the resolved attention
backend without calling `validate_configuration`, leaving every capability
gate the backends declare unenforced on XPU. These tests pin the wired-up
behavior. They run on CPU: the `vllm_xpu_kernels` modules that the XPU
platform imports at module load are stubbed out.
"""

import importlib
import sys
import types

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig

FA_PATH = AttentionBackendEnum.FLASH_ATTN.get_path()
TRITON_PATH = AttentionBackendEnum.TRITON_ATTN.get_path()

XPU_KERNELS_MODULES = (
    "vllm_xpu_kernels",
    "vllm_xpu_kernels._C",
    "vllm_xpu_kernels._moe_C",
    "vllm_xpu_kernels._xpu_C",
)


@pytest.fixture()
def xpu_platform():
    """Import vllm.platforms.xpu with stubbed vllm_xpu_kernels modules."""
    sentinel = object()
    saved_mem_info = getattr(torch.accelerator, "get_memory_info", sentinel)
    fake_pkg = types.ModuleType("vllm_xpu_kernels")
    fake_pkg.__path__ = []
    sys.modules["vllm_xpu_kernels"] = fake_pkg
    for name in XPU_KERNELS_MODULES[1:]:
        sys.modules[name] = types.ModuleType(name)
    try:
        mod = importlib.import_module("vllm.platforms.xpu")
        yield mod.XPUPlatform
    finally:
        current = getattr(torch.accelerator, "get_memory_info", sentinel)
        if saved_mem_info is sentinel:
            if current is not sentinel:
                del torch.accelerator.get_memory_info
        else:
            torch.accelerator.get_memory_info = saved_mem_info
        for name in XPU_KERNELS_MODULES:
            sys.modules.pop(name, None)


def cfg(**overrides) -> AttentionSelectorConfig:
    defaults = dict(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype=None,
        block_size=16,
    )
    defaults.update(overrides)
    return AttentionSelectorConfig(**defaults)


def test_default_backend_passes_validation(xpu_platform):
    assert xpu_platform.get_attn_backend_cls(None, cfg()) == FA_PATH


def test_float32_falls_back_to_triton(xpu_platform):
    path = xpu_platform.get_attn_backend_cls(None, cfg(dtype=torch.float32))
    assert path == TRITON_PATH


def test_batch_invariant_routes_to_triton(xpu_platform):
    path = xpu_platform.get_attn_backend_cls(None, cfg(use_batch_invariant=True))
    assert path == TRITON_PATH


def test_mm_prefix_routes_to_triton(xpu_platform):
    path = xpu_platform.get_attn_backend_cls(None, cfg(use_mm_prefix=True))
    assert path == TRITON_PATH


def test_explicit_flash_attn_with_batch_invariant_raises(xpu_platform):
    with pytest.raises(ValueError, match="batch invariance"):
        xpu_platform.get_attn_backend_cls(
            AttentionBackendEnum.FLASH_ATTN, cfg(use_batch_invariant=True)
        )


def test_explicit_flash_attn_with_mm_prefix_raises(xpu_platform):
    with pytest.raises(ValueError, match="prefix"):
        xpu_platform.get_attn_backend_cls(
            AttentionBackendEnum.FLASH_ATTN, cfg(use_mm_prefix=True)
        )


def test_validation_rejects_unsupported_head_size(xpu_platform):
    # 60 is not a multiple of 8, which FlashAttentionBackend rejects. The
    # gate used to be dead code on XPU and the config was accepted silently.
    with pytest.raises(ValueError, match="head_size"):
        xpu_platform.get_attn_backend_cls(
            AttentionBackendEnum.FLASH_ATTN, cfg(head_size=60)
        )


def test_auto_selection_falls_back_when_flash_attn_invalid(xpu_platform):
    # head_size=60 is valid for TritonAttentionBackend (>= 32), so the
    # auto-selected FlashAttention backend falls back instead of raising.
    path = xpu_platform.get_attn_backend_cls(None, cfg(head_size=60))
    assert path == TRITON_PATH


def test_explicit_triton_backend(xpu_platform):
    path = xpu_platform.get_attn_backend_cls(AttentionBackendEnum.TRITON_ATTN, cfg())
    assert path == TRITON_PATH


def test_unsupported_selected_backend_still_raises(xpu_platform):
    with pytest.raises(ValueError, match="Invalid attention backend"):
        xpu_platform.get_attn_backend_cls(AttentionBackendEnum.FLASH_ATTN_DIFFKV, cfg())


def test_validate_configuration_tolerates_missing_capability(xpu_platform):
    # XPUPlatform.get_device_capability() returns None because Intel GPUs
    # have no CUDA-style capability. The FlashAttention capability gates must
    # treat it as "not applicable" instead of raising, and must not report
    # spurious reasons for a well-formed configuration.
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

    assert FlashAttentionBackend.supports_compute_capability(None) is True
    assert (
        FlashAttentionBackend.supports_compute_capability(DeviceCapability(8, 0))
        is True
    )
    assert (
        FlashAttentionBackend.supports_compute_capability(DeviceCapability(7, 5))
        is False
    )
    reasons = FlashAttentionBackend.validate_configuration(
        device_capability=None, **cfg(has_sink=True)._asdict()
    )
    # Depending on the flash-attn build, has_sink may legitimately be
    # reported as unsupported, but a missing capability must not raise a
    # TypeError and must not be reported as an unsupported capability.
    assert "compute capability not supported" not in reasons


def test_routing_backends_are_validated(xpu_platform, monkeypatch):
    # The sparse/MLA/TurboQuant routes go through the same validation; stub
    # the backend resolution to keep the heavy backend modules out of the
    # CPU test run.
    monkeypatch.setattr(
        xpu_platform,
        "_validate_backend",
        classmethod(lambda cls, backend, attn_selector_config: []),
    )
    assert (
        xpu_platform.get_attn_backend_cls(None, cfg(kv_cache_dtype="turboquant_fp8"))
        == AttentionBackendEnum.TURBOQUANT.get_path()
    )
    assert (
        xpu_platform.get_attn_backend_cls(None, cfg(use_sparse=True))
        == AttentionBackendEnum.XPU_MLA_SPARSE.get_path()
    )
    assert (
        xpu_platform.get_attn_backend_cls(None, cfg(use_mla=True))
        == AttentionBackendEnum.TRITON_MLA.get_path()
    )


def test_invalid_routing_backend_raises(xpu_platform, monkeypatch):
    monkeypatch.setattr(
        xpu_platform,
        "_validate_backend",
        classmethod(lambda cls, backend, attn_selector_config: ["some reason"]),
    )
    with pytest.raises(ValueError, match="some reason"):
        xpu_platform.get_attn_backend_cls(None, cfg(use_mla=True))
