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
    "on_gfx942,on_gfx950,aiter_enabled,has_tuned,expected",
    [
        (True, False, True, True, "flydsl"),
        (False, True, True, True, "aiter"),
        (False, True, True, False, "triton"),
        (False, False, True, True, "triton"),
        (False, True, False, True, "torch"),
    ],
)
def test_rocm_fp8_mqa_logits_dispatch(
    monkeypatch, on_gfx942, on_gfx950, aiter_enabled, has_tuned, expected
):
    called = []
    out = torch.empty((1, 1))

    def fake(name):
        def fn(*args, **kwargs):
            called.append(name)
            return out

        return fn

    monkeypatch.setattr(sparse_mod, "_ON_GFX942", on_gfx942)
    monkeypatch.setattr(sparse_mod, "_ON_GFX950", on_gfx950)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: aiter_enabled)
    monkeypatch.setattr(rocm_aiter_ops, "is_rdna_aiter_enabled", lambda: False)
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl",
        SimpleNamespace(flydsl_fp8_mqa_logits=fake("flydsl")),
    )
    monkeypatch.setattr(
        sparse_mod,
        "aiter_tuned_mqa_logits",
        lambda: fake("aiter") if has_tuned else None,
    )
    monkeypatch.setattr(
        sparse_mod,
        "mqa_logits_module",
        lambda: SimpleNamespace(fp8_mqa_logits=fake("triton")),
    )
    monkeypatch.setattr(sparse_mod, "fp8_mqa_logits_torch", fake("torch"))

    result = sparse_mod.rocm_fp8_mqa_logits(
        torch.empty((1, 1, 1)),
        (torch.empty((1, 1)), torch.empty((1,))),
        torch.empty((1, 1)),
        torch.zeros((1,), dtype=torch.int32),
        torch.ones((1,), dtype=torch.int32),
    )

    assert result is out
    assert called == [expected]
