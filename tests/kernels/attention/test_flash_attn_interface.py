# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the FlashAttention wrapper's input layout contract."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def flash_attn_interface():
    # Load the wrapper without requiring the package's CUDA extensions.
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm/vllm_flash_attn/flash_attn_interface.py"
    )
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
