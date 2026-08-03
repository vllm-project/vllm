# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip(
        "ROCm AITER sparse indexer test requires ROCm.",
        allow_module_level=True,
    )

from vllm._aiter_ops import rocm_aiter_ops
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse_mod


def test_rocm_fp8_mqa_logits_disables_clean_logits(monkeypatch):
    captured = {}
    expected = torch.empty((1, 1))

    def fake_fp8_mqa_logits(*args, clean_logits):
        captured["clean_logits"] = clean_logits
        return expected

    monkeypatch.setattr(sparse_mod, "_ON_GFX942", False)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(
        sparse_mod,
        "mqa_logits_module",
        lambda: SimpleNamespace(fp8_mqa_logits=fake_fp8_mqa_logits),
    )

    q = torch.empty((1, 1, 1))
    k_fp8 = torch.empty((1, 1))
    scale = torch.empty((1,))
    weights = torch.empty((1, 1))
    cu_seqlen_ks = torch.zeros((1,), dtype=torch.int32)
    cu_seqlen_ke = torch.ones((1,), dtype=torch.int32)

    result = sparse_mod.rocm_fp8_mqa_logits(
        q,
        (k_fp8, scale),
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )

    assert result is expected
    assert captured["clean_logits"] is False
