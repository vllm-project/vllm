# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.platforms import current_platform

if not current_platform.is_rocm() or not is_aiter_found_and_supported():
    pytest.skip(
        "ROCm AITER paged-MQA logits test requires ROCm with AITER.",
        allow_module_level=True,
    )

from vllm._aiter_ops import rocm_aiter_ops
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse_mod


@pytest.mark.parametrize("next_n", [1, 2])
def test_rocm_fp8_paged_mqa_logits_uses_aiter_dispatch(monkeypatch, next_n):
    calls = []
    out = torch.empty((2 * next_n, 128), dtype=torch.float32)

    def fake_gluon(*args, **kwargs):
        calls.append(("gluon", args, kwargs))

    def fake_dispatch(*args, **kwargs):
        calls.append(("dispatch", args, kwargs))

    fake_aiter_module = ModuleType("aiter.paged_mqa_logits")
    fake_aiter_module.paged_mqa_logits = fake_dispatch  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aiter.paged_mqa_logits", fake_aiter_module)
    monkeypatch.setattr(sparse_mod, "_USE_FUSED_PAGED_MQA_LOGITS", True)
    monkeypatch.setattr(sparse_mod, "_ON_RDNA", False)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(
        sparse_mod,
        "paged_mqa_logits_module",
        lambda: SimpleNamespace(
            __name__="fake_pa_mqa_logits", deepgemm_fp8_paged_mqa_logits=fake_gluon
        ),
    )
    monkeypatch.setattr(sparse_mod, "_decode_logits_workspace", lambda *_: out)

    batch_size, heads, head_dim, block_size = 2, 32, 128, 64
    q_fp8 = torch.empty((batch_size, next_n, heads, head_dim))
    kv_cache = torch.empty((4, block_size, 1, head_dim + 4), dtype=torch.uint8)
    weights = torch.empty((batch_size * next_n, heads))
    context_lens = torch.ones((batch_size, next_n), dtype=torch.int32)
    block_tables = torch.zeros((batch_size, 2), dtype=torch.int32)

    result = sparse_mod.rocm_fp8_paged_mqa_logits(
        q_fp8,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        torch.empty(0),
        max_model_len=128,
    )

    assert result is out
    assert len(calls) == 1
    backend, args, kwargs = calls[0]
    assert backend == "dispatch"
    assert args[3] is out
    assert kwargs["Preshuffle"] is True
    assert kwargs["KVBlockSize"] == block_size
    assert "ChunkK" not in kwargs
    assert "WavePerEU" not in kwargs
