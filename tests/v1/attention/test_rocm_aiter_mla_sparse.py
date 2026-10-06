# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.platforms import current_platform

if not current_platform.is_rocm() or not is_aiter_found_and_supported():
    pytest.skip(
        "ROCm AITER sparse indexer test requires ROCm with AITER.",
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
    monkeypatch.setattr(sparse_mod, "_ON_GFX950", False)
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


@pytest.mark.parametrize("seq_len,seq_len_kv", [(256, 4096), (2048, 16384)])
@torch.inference_mode()
def test_aiter_mqa_logits_matches_triton_call(monkeypatch, seq_len, seq_len_kv):
    """On gfx950, aiter.mqa_logits picks the prefill kernel per shape; inside
    each row's key window it must give the logits of the Triton/Gluon call it
    replaces."""
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("aiter.mqa_logits is only called on gfx950.")
    pytest.importorskip("aiter.mqa_logits")
    from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()
    try:
        assert sparse_mod._aiter_mqa_logits() is not None
        torch.manual_seed(0)
        heads, dim = 32, 128
        fp8 = current_platform.fp8_dtype()
        q = torch.randn(seq_len, heads, dim, device="cuda").to(fp8)
        k = torch.randn(seq_len_kv, dim, device="cuda").to(fp8)
        # Callers pass the per-key scales as [N, 1], a view of the uint8 cache.
        scale = torch.rand(seq_len_kv, 1, device="cuda") + 0.5
        weights = torch.randn(seq_len, heads, device="cuda")
        # The causal windows of the last seq_len tokens of one sequence.
        ks = torch.zeros(seq_len, device="cuda", dtype=torch.int32)
        ke = (
            seq_len_kv
            - seq_len
            + 1
            + torch.arange(seq_len, device="cuda", dtype=torch.int32)
        )

        out = sparse_mod.rocm_fp8_mqa_logits(q, (k, scale), weights, ks, ke)
        ref = fp8_mqa_logits(q, k, scale, weights, ks, ke, clean_logits=False)

        cols = torch.arange(seq_len_kv, device="cuda")
        valid = (cols >= ks[:, None]) & (cols < ke[:, None])
        torch.testing.assert_close(out[valid], ref[valid], rtol=1e-5, atol=1e-4)
    finally:
        monkeypatch.undo()
        rocm_aiter_ops.refresh_env_variables()
