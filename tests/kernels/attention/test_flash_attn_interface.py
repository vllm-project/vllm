# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the FlashAttention wrapper's input layout contract."""

import importlib.util
import sys
from importlib.metadata import distribution
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def flash_attn_interface():
    # Load the wrapper without requiring the package's CUDA extensions.
    root = Path(__file__).resolve().parents[3]
    relative_path = Path("vllm/vllm_flash_attn/flash_attn_interface.py")
    path = root / relative_path
    if not path.is_file():
        # CI images move the source tree under src/ to test the installed wheel.
        path = root / "src" / relative_path
    if not path.is_file():
        # Native ROCm CI ships only the installed wheel and test workspace.
        path = Path(distribution("vllm").locate_file(relative_path))
    spec = importlib.util.spec_from_file_location("flash_attn_interface", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize("layout", ["scalar", "per_head", "none"])
@pytest.mark.parametrize(
    "dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16]
)
def test_fa4_descale_layout(
    flash_attn_interface, monkeypatch, batch_size, num_kv_heads, layout, dtype
):
    """FA4 receives unit-stride descales, including scalar-expanded (1, 1)."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    shape = (batch_size, num_kv_heads)
    descales: dict[str, torch.Tensor | None] = {}
    for i, name in enumerate(("q_descale", "k_descale", "v_descale")):
        if layout == "none":
            descales[name] = None
        else:
            scale = torch.tensor(i + 0.5, device="cpu")
            if layout == "per_head":
                scale = scale + torch.arange(num_kv_heads, device="cpu")
            descales[name] = scale.expand(shape)

    q = torch.empty(batch_size, 4, 128, dtype=dtype, device="cpu")
    kv = torch.empty(batch_size, num_kv_heads, 128, dtype=dtype, device="cpu")
    out = torch.empty_like(q, dtype=torch.bfloat16)
    fwd = Mock(return_value=(out, None, None, None))
    cute_interface = ModuleType("vllm.vllm_flash_attn.cute.interface")
    monkeypatch.setattr(cute_interface, "_flash_attn_fwd", fwd, raising=False)
    monkeypatch.setitem(sys.modules, cute_interface.__name__, cute_interface)
    cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int32, device="cpu")

    flash_attn_interface.flash_attn_varlen_func(
        q,
        kv,
        kv,
        max_seqlen_q=1,
        cu_seqlens_q=cu_seqlens,
        max_seqlen_k=1,
        cu_seqlens_k=cu_seqlens,
        fa_version=4,
        **descales,
    )

    fwd.assert_called_once()
    for name, original in descales.items():
        actual = fwd.call_args.kwargs[name]
        if dtype == torch.bfloat16 or original is None:
            assert actual is None
        else:
            assert actual.stride(1) == 1
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
            if original.stride(1) == 1:
                assert actual is original


def test_fa_utils_import_and_backend_validation_without_fa_extensions(monkeypatch):
    """CUDA builds without _vllm_fa2_C/_vllm_fa3_C must import fa_utils cleanly
    and reject FlashAttentionBackend during backend validation."""
    import vllm._custom_ops as custom_ops
    from vllm.platforms.interface import DeviceCapability, Platform

    class _FakeCudaPlatform(Platform):
        device_type = "cuda"

        def is_cuda(self) -> bool:
            return True

        def is_xpu(self) -> bool:
            return False

        def is_rocm(self) -> bool:
            return False

        @classmethod
        def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
            return DeviceCapability(11, 0)

    fake_platform = _FakeCudaPlatform()
    monkeypatch.setattr("vllm.platforms.current_platform", fake_platform)
    monkeypatch.setattr(custom_ops, "reshape_and_cache_flash", Mock(), raising=False)
    monkeypatch.delitem(sys.modules, "vllm.vllm_flash_attn", raising=False)
    monkeypatch.delitem(
        sys.modules, "vllm.vllm_flash_attn.flash_attn_interface", raising=False
    )

    root = Path(__file__).resolve().parents[3]
    fa_utils_path = root / "vllm/v1/attention/backends/fa_utils.py"
    spec = importlib.util.spec_from_file_location("_test_fa_utils_no_fa", fa_utils_path)
    assert spec is not None and spec.loader is not None
    fa_utils_mod = importlib.util.module_from_spec(spec)
    # Must not raise ImportError at module import time when FA extensions are absent.
    spec.loader.exec_module(fa_utils_mod)

    assert not fa_utils_mod.is_flash_attn_varlen_func_available()
    assert fa_utils_mod.get_flash_attn_version(head_size=128) is None
    with pytest.raises(ImportError, match="requires the CUDA flash attention"):
        fa_utils_mod.flash_attn_varlen_func()

    from vllm.v1.attention.backends import flash_attn as flash_attn_mod

    monkeypatch.setattr(flash_attn_mod, "current_platform", fake_platform)
    monkeypatch.setattr(
        flash_attn_mod,
        "is_flash_attn_varlen_func_available",
        fa_utils_mod.is_flash_attn_varlen_func_available,
    )
    monkeypatch.setattr(
        flash_attn_mod,
        "get_flash_attn_version",
        fa_utils_mod.get_flash_attn_version,
    )

    reason = flash_attn_mod.FlashAttentionBackend.supports_combination(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype=None,
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        device_capability=DeviceCapability(11, 0),
    )
    assert reason is not None and "FlashAttention" in reason
