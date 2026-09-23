# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-only aiter path"
)


def _rows_from_ragged(indices: torch.Tensor, indptr: torch.Tensor) -> list[list[int]]:
    ends = indptr.tolist()
    return [indices[s:e].tolist() for s, e in zip(ends, ends[1:])]


def test_triton_sparse_mla_gate(monkeypatch) -> None:
    import vllm._aiter_ops as aiter_ops
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.platforms.rocm import on_gfx950

    warnings: list[str] = []
    monkeypatch.setattr(
        aiter_ops.logger,
        "warning_once",
        lambda msg, *args, **kwargs: warnings.append(msg % args),
    )

    def enabled(aiter: bool, flag: bool) -> bool:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", str(int(aiter)))
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA", str(int(flag)))
        rocm_aiter_ops.refresh_env_variables()
        return bool(rocm_aiter_ops.is_triton_sparse_mla_enabled())

    try:
        assert not enabled(aiter=True, flag=False) and not warnings
        assert not enabled(aiter=False, flag=True)
        assert "VLLM_ROCM_USE_AITER is off" in warnings.pop()
        # Without the kernel the flag warns and falls back, whatever the arch.
        monkeypatch.setattr(aiter_ops, "_has_aiter_triton_sparse_mla", lambda: False)
        assert not enabled(aiter=True, flag=True)
        assert ("sparse_mla" if on_gfx950() else "gfx950") in warnings.pop()
    finally:
        monkeypatch.undo()
        rocm_aiter_ops.refresh_env_variables()


@pytest.mark.parametrize(
    "rope_dim",
    [0, 64],
    ids=["rope_free", "appended_rope"],  # GLM-5.3-Flash; GLM-5.1/5.2, V3.2
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@torch.inference_mode()
def test_forward_mqa_prepares_triton_sparse_mla_inputs(
    monkeypatch, rope_dim: int, kv_cache_dtype: str
) -> None:
    """What forward_mqa hands aiter: global slots built from the request-local
    top-k rows, q quantized with the layer's scale, and the cache as stored."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseImpl,
    )

    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(0)
    kv_lora_rank, num_heads, topk, block_size = 512, 16, 128, 16
    head_size = kv_lora_rank + rope_dim
    fp8 = kv_cache_dtype == "fp8"

    # A decode row, then a short prefill.
    query_lens, seq_lens = [1, 5], [40, 20]
    num_tokens = sum(query_lens)
    block_table = torch.randperm(8, generator=gen).view(2, 4).to(device, torch.int32)
    topk_indices = torch.full((num_tokens, topk), -1, dtype=torch.int32)
    req_id_per_token: list[int] = []
    rows: list[list[int]] = []
    for req, (query_len, seq_len) in enumerate(zip(query_lens, seq_lens)):
        for pos in range(seq_len - query_len, seq_len):
            local = torch.randperm(pos + 1, generator=gen)
            topk_indices[len(rows), : pos + 1] = local.int()
            slots = block_table[req, local // block_size].cpu() * block_size
            rows.append((slots + local % block_size).tolist())
            req_id_per_token.append(req)
    indptr = torch.tensor([0] + [len(row) for row in rows]).cumsum(0)

    impl = ROCMAiterMLASparseImpl.__new__(ROCMAiterMLASparseImpl)
    impl.num_heads = num_heads
    impl.kv_lora_rank = kv_lora_rank
    impl.qk_rope_head_dim = rope_dim
    impl.scale = head_size**-0.5
    impl.sinks = torch.randn(num_heads, generator=gen).to(device)
    impl.kv_cache_dtype = kv_cache_dtype
    impl.topk_indices_buffer = topk_indices.to(device)
    impl.use_aiter_sparse_mla = True
    layer = SimpleNamespace(
        _q_scale=torch.tensor([0.02], device=device),
        _k_scale=torch.tensor([0.01], device=device),
    )
    metadata = SimpleNamespace(
        num_actual_tokens=num_tokens,
        topk_tokens=topk,
        req_id_per_token=torch.tensor(req_id_per_token, dtype=torch.int32).to(device),
        block_table=block_table,
        block_size=block_size,
        paged_kv_indptr=indptr.to(device, torch.int32),
        paged_kv_indices=torch.zeros(num_tokens * topk, dtype=torch.int32).to(device),
        attn_out_dtype=torch.bfloat16,
    )
    cache_dtype = current_platform.fp8_dtype() if fp8 else torch.bfloat16
    kv_cache = torch.zeros(8, block_size, head_size, device=device).to(cache_dtype)
    # Two rows of cudagraph padding past the real tokens.
    q = torch.randn(num_tokens + 2, num_heads, head_size, generator=gen).to(
        device, torch.bfloat16
    )
    calls = []
    monkeypatch.setattr(
        rocm_aiter_ops, "triton_sparse_mla_fwd", lambda *a, **k: calls.append((a, k))
    )

    out, lse = impl.forward_mqa(q, kv_cache, metadata, layer)

    assert len(calls) == 1
    (q_in, kv_in, o, sm_scale, kv_indptr, kv_indices), kwargs = calls[0]
    assert _rows_from_ragged(kv_indices, kv_indptr) == rows
    assert q_in.shape[0] == num_tokens
    if fp8:
        assert q_in.dtype == cache_dtype
        torch.testing.assert_close(
            q_in.float() * layer._q_scale, q[:num_tokens].float(), atol=1e-3, rtol=0.07
        )
    else:
        assert q_in.data_ptr() == q.data_ptr()
    assert kv_in.shape == (8 * block_size, 1, 1, head_size)
    assert kv_in.data_ptr() == kv_cache.data_ptr()
    assert o.data_ptr() == out.data_ptr() and o.shape[0] == num_tokens
    assert sm_scale == impl.scale
    assert kwargs["kv_lora_rank"] == kv_lora_rank
    assert kwargs["qk_rope_head_dim"] == rope_dim
    assert kwargs["q_scale"] is layer._q_scale
    assert kwargs["kv_scale"] is layer._k_scale
    assert kwargs["attn_sink"] is impl.sinks
    assert lse is None
