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


@pytest.mark.parametrize("aiter_has_op", [True, False])
def test_rocm_fp8_paged_mqa_logits_calls_aiter_op(monkeypatch, aiter_has_op):
    batch_size, next_n, heads, max_model_len = 2, 3, 32, 256
    called = {}

    def fake_aiter_op(*args, **kwargs):
        called["kernel"] = "aiter"
        called["args"] = args
        called["kwargs"] = kwargs

    def fake_gluon(*args, **kwargs):
        called["kernel"] = "gluon"

    workspace = SimpleNamespace(
        get_simultaneous=lambda *specs: tuple(
            torch.empty(shape, dtype=dtype) for shape, dtype in specs
        )
    )
    monkeypatch.setattr(sparse_mod, "_ON_GFX950", True)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(
        sparse_mod,
        "_aiter_paged_mqa_logits",
        lambda: fake_aiter_op if aiter_has_op else None,
    )
    monkeypatch.setattr(
        sparse_mod,
        "paged_mqa_logits_module",
        lambda: SimpleNamespace(deepgemm_fp8_paged_mqa_logits=fake_gluon),
    )
    monkeypatch.setattr(sparse_mod, "current_workspace_manager", lambda: workspace)

    q = torch.empty((batch_size, next_n, heads, 128))
    kv_cache = torch.empty((8, 64, 1, 132), dtype=torch.uint8)
    weights = torch.empty((batch_size * next_n, heads))
    seq_lens = torch.tensor([[198, 199, 200], [88, 89, 90]], dtype=torch.int32)
    block_tables = torch.zeros((batch_size, 4), dtype=torch.int32)

    out = sparse_mod.rocm_fp8_paged_mqa_logits(
        q, kv_cache, weights, seq_lens, block_tables, None, max_model_len
    )

    assert out.shape == (batch_size * next_n, max_model_len)
    if not aiter_has_op:
        assert called["kernel"] == "gluon"
        return
    assert called["kernel"] == "aiter"
    assert called["args"][0] is q
    assert called["args"][1] is kv_cache
    assert called["args"][3] is out
    assert called["args"][4] is seq_lens
    assert called["args"][6] == max_model_len
    assert called["kwargs"] == {"Preshuffle": True, "KVBlockSize": 64}
