# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    "on_gfx950,available,heads,expected",
    [
        (True, True, 32, "flydsl"),
        (True, False, 32, "gluon"),
        (True, True, 64, "gluon"),
        (False, False, 32, "gluon"),
    ],
)
def test_rocm_fp8_paged_mqa_logits_dispatch(
    monkeypatch, on_gfx950, available, heads, expected
):
    batch_size, next_n, max_model_len = 2, 3, 256
    called = {}

    def fake_flydsl(*args, **kwargs):
        called["kernel"] = "flydsl"
        called["context_lens"] = args[4]
        called["split_kv"] = kwargs["SplitKV"]

    def fake_gluon(*args, **kwargs):
        called["kernel"] = "gluon"

    workspace = SimpleNamespace(
        get_simultaneous=lambda *specs: tuple(
            torch.empty(shape, dtype=dtype) for shape, dtype in specs
        )
    )

    monkeypatch.setattr(sparse_mod, "_ON_GFX942", not on_gfx950)
    monkeypatch.setattr(sparse_mod, "_ON_GFX950", on_gfx950)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(
        sparse_mod,
        "_flydsl_paged_mqa_logits_kernel",
        lambda: fake_flydsl if available else None,
    )
    monkeypatch.setattr(
        sparse_mod,
        "paged_mqa_logits_module",
        lambda: SimpleNamespace(deepgemm_fp8_paged_mqa_logits=fake_gluon),
    )
    monkeypatch.setattr(sparse_mod, "current_workspace_manager", lambda: workspace)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=256),
    )

    seq_lens = torch.tensor([200, 90], dtype=torch.int32)
    per_row = seq_lens.unsqueeze(1) + torch.arange(next_n, dtype=torch.int32) - 2

    sparse_mod.rocm_fp8_paged_mqa_logits(
        torch.empty((batch_size, next_n, heads, 128)),
        torch.empty((8, 64, 1, 132), dtype=torch.uint8),
        torch.empty((batch_size * next_n, heads)),
        per_row,
        torch.zeros((batch_size, 4), dtype=torch.int32),
        None,
        max_model_len,
    )

    assert called["kernel"] == expected
    if expected == "flydsl":
        assert torch.equal(called["context_lens"], seq_lens)
        assert called["split_kv"] == 4
