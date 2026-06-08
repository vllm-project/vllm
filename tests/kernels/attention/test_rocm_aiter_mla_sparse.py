# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-level tests for ROCm AITER sparse MLA.

This file owns the ROCm sparse-MLA paths that are not already covered by the
generic or metadata-focused siblings:
- helper kernels that rewrite sparse indices into FlashMLA's expected format
- direct BF16 sparse-forward execution against the backend reference
- sparse-forward determinism on MI300/MI350 hardware

Related siblings:
- ``tests/kernels/rocm/aiter/test_rocm_aiter_mla_variants.py`` covers backend
  metadata and import/name contracts
- ``tests/v1/attention/test_sparse_mla_backends.py`` covers the sparse MLA
  backend behavior
"""

from types import SimpleNamespace

import pytest
import torch
from tests.kernels.rocm.utils import _assert_accurate, _assert_deterministic

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_mi3xx
from vllm.utils.torch_utils import set_random_seed

pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-specific tests"),
    pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only"),
]


# Sparse MLA helper tests -------------------------------------------------


def _assert_aiter_supported() -> None:
    from vllm._aiter_ops import is_aiter_found_and_supported

    assert is_aiter_found_and_supported(), (
        "aiter is required on supported ROCm hardware for this test"
    )


def _print_close_stats(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    pass_rate: float = 0.99999,
    max_atol: float | None = None,
) -> None:
    abs_diff = (actual - expected).abs().float().flatten()
    expected_abs = expected.abs().float().flatten()
    allowed = atol + rtol * expected_abs
    within = abs_diff <= allowed

    total = abs_diff.numel()
    passed = int(within.sum().item())
    failed = total - passed
    allowed_fail_rate = 1.0 - pass_rate

    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    p99_abs = torch.quantile(abs_diff, 0.99).item()
    p999_abs = torch.quantile(abs_diff, 0.999).item()
    worst_ratio = (abs_diff / allowed.clamp_min(1e-12)).max().item()

    msg = (
        "[rocm_aiter_mla_sparse] "
        f"{label}: "
        f"pass={passed / total:.4%} ({passed}/{total}) "
        f"fail={failed / total:.4%} ({failed}/{total}) "
        f"allowed_fail={allowed_fail_rate:.4%} "
        f"atol={atol:g} "
        f"rtol={rtol:g} "
    )
    if max_atol is not None:
        above_max = int((abs_diff > max_atol).sum().item())
        msg += f"abs>{max_atol:g}={above_max / total:.4%} ({above_max}/{total}) "
    msg += (
        f"max_abs={max_abs:.6g} "
        f"mean_abs={mean_abs:.6g} "
        f"p99_abs={p99_abs:.6g} "
        f"p999_abs={p999_abs:.6g} "
        f"worst_ratio={worst_ratio:.6g}"
    )
    print(msg)


def _make_sparse_metadata(batch_size: int, topk: int, device: torch.device):
    """Create minimal ROCMAiterMLASparseMetadata for direct kernel tests."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseMetadata,
    )

    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    paged_kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)
    # These buffers are populated by _forward_bf16_kv internally
    paged_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    paged_kv_indices = torch.zeros(batch_size * topk, dtype=torch.int32, device=device)
    paged_kv_indptr_rest = torch.zeros(0, dtype=torch.int32, device=device)

    return ROCMAiterMLASparseMetadata(
        num_reqs=batch_size,
        max_query_len=1,
        max_seq_len=1,
        num_actual_tokens=batch_size,
        query_start_loc=qo_indptr,
        slot_mapping=torch.zeros(batch_size, dtype=torch.long, device=device),
        block_table=torch.zeros((batch_size, 1), dtype=torch.int32, device=device),
        req_id_per_token=torch.zeros(batch_size, dtype=torch.int32, device=device),
        qo_indptr=qo_indptr,
        paged_kv_last_page_len=paged_kv_last_page_len,
        paged_kv_indices=paged_kv_indices,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indptr_rest=paged_kv_indptr_rest,
        block_size=1,
        topk_tokens=topk,
    )


def _make_sparse_impl(nhead: int, q_head_dim: int, v_head_dim: int):
    """Create minimal ROCMAiterMLASparseImpl for direct forward calls."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseImpl,
    )

    impl = SimpleNamespace(
        num_heads=nhead,
        scale=q_head_dim**-0.5,
        kv_lora_rank=v_head_dim,
    )
    impl._forward_bf16_kv = ROCMAiterMLASparseImpl._forward_bf16_kv.__get__(
        impl, ROCMAiterMLASparseImpl
    )
    return impl


def _reference_index_convert(
    req_ids: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Pure-Python reference for triton_convert_req_index_to_global_index."""
    num_tokens, num_topk = token_indices.shape
    max_blocks = block_table.shape[1]
    result = torch.empty_like(token_indices)
    for t in range(num_tokens):
        req = req_ids[t].item()
        for k in range(num_topk):
            idx = token_indices[t, k].item()
            if idx == -1:
                result[t, k] = -1
            else:
                block_id = idx // block_size
                if block_id >= max_blocks:
                    result[t, k] = -1
                else:
                    result[t, k] = (
                        block_table[req, block_id].item() * block_size
                        + idx % block_size
                    )
    return result


def _reference_fetch_id_to_ragged(
    topk_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = topk_indices != -1
    seq_len = valid_mask.sum(dim=-1, dtype=torch.int32)
    cumsum = torch.zeros(
        topk_indices.shape[0] + 1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    torch.cumsum(seq_len, dim=0, out=cumsum[1:])
    out = torch.zeros(
        topk_indices.numel(),
        dtype=torch.int32,
        device=topk_indices.device,
    )
    out[: int(seq_len.sum().item())] = topk_indices[valid_mask]
    return cumsum, out


@pytest.mark.parametrize("block_size", [1, 16, 64])
@pytest.mark.parametrize("num_topk", [128, 256])  # must be divisible by BLOCK_N=128
def test_rocm_sparse_mla_triton_index_conversion(block_size, num_topk):
    """The ROCm sparse index rewrite should match the Python reference exactly."""
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        triton_convert_req_index_to_global_index,
    )

    device = torch.device("cuda")
    num_tokens = 8
    num_reqs = 4
    max_blocks = 10

    req_ids = torch.randint(
        0, num_reqs, (num_tokens,), dtype=torch.int32, device=device
    )
    block_table = torch.randint(
        0, 50, (num_reqs, max_blocks), dtype=torch.int32, device=device
    )
    token_indices = torch.randint(
        0,
        block_size * max_blocks,
        (num_tokens, num_topk),
        dtype=torch.int32,
        device=device,
    )
    # Insert some -1 masked entries
    token_indices[0, :5] = -1
    token_indices[3, num_topk // 2 :] = -1

    result = triton_convert_req_index_to_global_index(
        req_ids,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk,
    )
    ref = _reference_index_convert(req_ids, block_table, token_indices, block_size)

    torch.testing.assert_close(result, ref, rtol=0, atol=0)


def test_rocm_sparse_mla_fetch_id_to_ragged():
    """The ROCm ragged gather helper should pack valid top-k ids contiguously."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        fetch_id_to_ragged_triton,
    )

    device = torch.device("cuda")
    num_tokens = 4
    topk = 8

    # Indices: some valid (>=0), some masked (-1)
    topk_indices = torch.tensor(
        [
            [5, 2, -1, -1, -1, -1, -1, -1],  # 2 valid
            [0, 3, 7, 1, -1, -1, -1, -1],  # 4 valid
            [-1, -1, -1, -1, -1, -1, -1, -1],  # 0 valid
            [4, 6, -1, -1, -1, -1, -1, -1],  # 2 valid
        ],
        dtype=torch.int32,
        device=device,
    )
    seq_len = (topk_indices != -1).sum(dim=-1)  # [2, 4, 0, 2]
    cumsum = torch.zeros(num_tokens + 1, dtype=torch.int32, device=device)
    torch.cumsum(seq_len, dim=0, out=cumsum[1:])

    out = torch.zeros(num_tokens * topk, dtype=torch.int32, device=device)

    fetch_id_to_ragged_triton(topk_indices, cumsum, out, topk)
    ref_cumsum, ref_out = _reference_fetch_id_to_ragged(topk_indices)

    torch.testing.assert_close(cumsum, ref_cumsum, atol=0, rtol=0)
    torch.testing.assert_close(out, ref_out, atol=0, rtol=0)


# Sparse forward tests ----------------------------------------------------


def test_rocm_sparse_mla_forward_smoke():
    """A representative ROCm sparse MLA launch should return finite BF16 output."""
    _assert_aiter_supported()

    device = torch.device("cuda")
    set_random_seed(0)

    batch_size = 4
    nhead = 128  # gfx942 supported
    q_head_dim = 576  # kv_lora_rank + qk_rope_head_dim
    v_head_dim = 512  # kv_lora_rank (output dim)
    num_kv_tokens = 512
    topk = 128  # must be divisible by BLOCK_N=128

    q = torch.randn(batch_size, nhead, q_head_dim, dtype=torch.bfloat16, device=device)
    # block_size=1: shape (num_kv_tokens, 1, q_head_dim)
    kv_cache = torch.randn(
        num_kv_tokens, 1, q_head_dim, dtype=torch.bfloat16, device=device
    )
    # Global block indices (block_size=1 → block_idx == token_idx)
    topk_indices = torch.randint(
        0, num_kv_tokens, (batch_size, topk), dtype=torch.int32, device=device
    )

    impl = _make_sparse_impl(nhead, q_head_dim, v_head_dim)
    metadata = _make_sparse_metadata(batch_size, topk, device)

    output = impl._forward_bf16_kv(q, kv_cache, topk_indices, metadata)

    assert output.shape == (batch_size, nhead, v_head_dim)
    assert output.dtype == torch.bfloat16
    assert not torch.all(output == 0), "output should be non-trivial"
    assert torch.isfinite(output).all(), "output should not contain NaN or Inf"


@pytest.mark.parametrize("nhead", [16, 128])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("topk", [128, 256])  # must be divisible by BLOCK_N=128
def test_rocm_sparse_mla_forward_accuracy(nhead, batch_size, topk):
    """Sparse MLA forward output matches reference_mla_sparse_prefill.

    Parity with test_sparse_mla_backends.py::test_sparse_backend_decode_correctness
    (CUDA, auto/BF16 KV) adapted for ROCMAiterMLASparseBackend.

    Reference: reference_mla_sparse_prefill implements:
      scores[q, h, k] = q[q, h, :] @ kv[topk[q, k], 0, :].T * scale
      out[q, h, :] = softmax(scores[q, h, :]) @ kv[topk[q, k], 0, :v_head_dim]
    which is the absorbed MLA formulation with sparse KV selection.
    """
    _assert_aiter_supported()

    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        reference_mla_sparse_prefill,
    )

    device = torch.device("cuda")
    set_random_seed(nhead * 1000 + batch_size * 10 + topk)

    q_head_dim = 576  # kv_lora_rank(512) + qk_rope_head_dim(64)
    v_head_dim = 512  # kv_lora_rank
    num_kv_tokens = max(512, topk * 2)

    q = torch.randn(batch_size, nhead, q_head_dim, dtype=torch.bfloat16, device=device)
    # block_size=1: shape (num_kv_tokens, 1, q_head_dim)
    kv_cache = torch.randn(
        num_kv_tokens, 1, q_head_dim, dtype=torch.bfloat16, device=device
    )
    # All-valid indices (no -1) for exact reference comparison
    topk_indices = torch.randint(
        0, num_kv_tokens, (batch_size, topk), dtype=torch.int32, device=device
    )

    impl = _make_sparse_impl(nhead, q_head_dim, v_head_dim)
    metadata = _make_sparse_metadata(batch_size, topk, device)

    output = impl._forward_bf16_kv(q, kv_cache, topk_indices, metadata)

    # reference_mla_sparse_prefill expects indices shape [sq, 1, topk]
    ref_out, _ = reference_mla_sparse_prefill(
        q=q,
        kv=kv_cache,
        indices=topk_indices.unsqueeze(1),
        sm_scale=q_head_dim**-0.5,
        d_v=v_head_dim,
    )

    assert output.shape == ref_out.shape, (
        f"Shape mismatch: got {output.shape}, expected {ref_out.shape}"
    )
    _print_close_stats(
        f"forward_accuracy nhead={nhead} batch={batch_size} topk={topk}",
        output.float(),
        ref_out.float(),
        atol=0.01,
        rtol=0.0,
        max_atol=0.03,
    )
    _assert_accurate(output.float(), ref_out.float(), atol=0.01, rtol=0.0)


def test_rocm_sparse_mla_forward_determinism():
    """Sparse MLA forward should stay bitwise deterministic for fixed inputs."""
    _assert_aiter_supported()

    device = torch.device("cuda")
    set_random_seed(42)

    batch_size = 4
    nhead = 128
    q_head_dim = 576
    v_head_dim = 512
    num_kv_tokens = 512
    topk = 128  # must be divisible by BLOCK_N=128

    q = torch.randn(batch_size, nhead, q_head_dim, dtype=torch.bfloat16, device=device)
    kv_cache = torch.randn(
        num_kv_tokens, 1, q_head_dim, dtype=torch.bfloat16, device=device
    )
    topk_indices = torch.randint(
        0, num_kv_tokens, (batch_size, topk), dtype=torch.int32, device=device
    )

    impl = _make_sparse_impl(nhead, q_head_dim, v_head_dim)

    def run():
        metadata = _make_sparse_metadata(batch_size, topk, device)
        return impl._forward_bf16_kv(q, kv_cache, topk_indices, metadata)

    _assert_deterministic(run, n_runs=4)
