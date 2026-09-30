# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.platforms import current_platform

if not current_platform.is_rocm() or not is_aiter_found_and_supported():
    pytest.skip("Requires ROCm with AITER.", allow_module_level=True)

from vllm._aiter_ops import rocm_aiter_ops
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse_mod


@pytest.mark.parametrize(
    "on_gfx942,on_gfx950,flag,expected",
    [
        (True, False, False, "flydsl"),
        (False, True, True, "flydsl"),
        (False, True, False, "triton"),
    ],
)
def test_rocm_fp8_mqa_logits_dispatch(
    monkeypatch, on_gfx942, on_gfx950, flag, expected
):
    called = []
    out = torch.empty((1, 1))

    def fake_flydsl(*args, **kwargs):
        called.append("flydsl")
        return out

    def fake_triton(*args, **kwargs):
        called.append("triton")
        return out

    monkeypatch.setattr(sparse_mod, "_ON_GFX942", on_gfx942)
    monkeypatch.setattr(sparse_mod, "_ON_GFX950", on_gfx950)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FLYDSL_MQA_LOGITS", "1" if flag else "0")
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl",
        SimpleNamespace(flydsl_fp8_mqa_logits=fake_flydsl),
    )
    monkeypatch.setattr(
        sparse_mod,
        "mqa_logits_module",
        lambda: SimpleNamespace(fp8_mqa_logits=fake_triton),
    )

    result = sparse_mod.rocm_fp8_mqa_logits(
        torch.empty((1, 1, 1)),
        (torch.empty((1, 1)), torch.empty((1,))),
        torch.empty((1, 1)),
        torch.zeros((1,), dtype=torch.int32),
        torch.ones((1,), dtype=torch.int32),
    )

    assert result is out
    assert called == [expected]
