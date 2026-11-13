# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for per-request causal/non-causal attention (mixed batches).

Validates that both triton and flash-attention backends correctly handle
batches where some sequences use causal masking and others use non-causal
(bidirectional) masking — needed by DiffusionGemma.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

# Mixed causal/non-causal attention is only validated on a subset of GPUs:
# the Triton path on Hopper (SM90) and B200 (SM100); the FA4 path on Hopper
# (SM90) only.
_device_capability = current_platform.get_device_capability()
_major = _device_capability.major if _device_capability is not None else None

NUM_HEADS = [(4, 4), (8, 2)]
HEAD_SIZES = [128]
BLOCK_SIZES = [16]
DTYPES = [torch.bfloat16]


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    per_seq_causal: list[bool],
    sliding_window: int | None = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q = q * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_np[i, :num_kv_blocks]
        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()

        if per_seq_causal[i]:
            mask = torch.triu(
                torch.ones(query_len, kv_len, device=attn.device),
                diagonal=kv_len - query_len + 1,
            ).bool()
        else:
            mask = torch.zeros(query_len, kv_len, device=attn.device).bool()

        if sliding_window is not None:
            sw_mask = (
                torch.triu(
                    torch.ones(query_len, kv_len, device=attn.device),
                    diagonal=kv_len - (query_len + sliding_window) + 1,
                )
                .bool()
                .logical_not()
            )
            mask |= sw_mask

        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


# ---- Triton backend test ----


@pytest.mark.skipif(
    _major not in (9, 10),
    reason="Triton mixed causal attention requires Hopper (SM90) or B200 (SM100).",
)
@pytest.mark.parametrize(
    "seq_lens",
    [[(1, 128), (5, 64), (1, 256)]],
)
@pytest.mark.parametrize(
    "per_seq_causal",
    [[True, False, True], [False, True, False], [True, True, False]],
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_triton_mixed_causal(
    seq_lens: list[tuple[int, int]],
    per_seq_causal: list[bool],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    set_random_seed(42)
    device = "cuda"

    num_query_heads, num_kv_heads = num_heads
    assert len(seq_lens) == len(per_seq_causal)

    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    num_seqs = len(seq_lens)

    num_query_tokens = sum(query_lens)
    max_kv_len = max(kv_lens)
    max_num_blocks = (max_kv_len + block_size - 1) // block_size
    num_blocks = max_num_blocks * num_seqs + 10

    scale = head_size**-0.5
    query = torch.randn(
        num_query_tokens, num_query_heads, head_size, dtype=dtype, device=device
    )
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    value_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )

    block_tables_list = []
    for i in range(num_seqs):
        n_blocks = (kv_lens[i] + block_size - 1) // block_size
        blocks = list(range(i * max_num_blocks, i * max_num_blocks + n_blocks))
        blocks += [0] * (max_num_blocks - n_blocks)
        block_tables_list.append(blocks)
    block_tables = torch.tensor(block_tables_list, dtype=torch.int32, device=device)

    cu_seqlens_q = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    for i, ql in enumerate(query_lens):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + ql

    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    max_seqlen_q = max(query_lens)
    max_seqlen_k = max(kv_lens)

    causal_tensor = torch.tensor(per_seq_causal, dtype=torch.bool, device=device)

    output = torch.empty_like(query)
    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=scale,
        causal=causal_tensor,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0.0,
        q_descale=None,
        k_descale=1.0,
        v_descale=1.0,
    )

    ref_output = ref_paged_attn(
        query,
        key_cache,
        value_cache,
        query_lens,
        kv_lens,
        block_tables,
        scale,
        per_seq_causal,
    )

    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)


# ---- Flash Attention 4 backend test (native per_seq_causal) ----


@pytest.mark.skipif(
    _major != 9,
    reason="FA4 mixed causal attention requires Hopper (SM90).",
)
@pytest.mark.parametrize(
    "seq_lens",
    [[(1, 128), (5, 64), (1, 256)]],
)
@pytest.mark.parametrize(
    "per_seq_causal",
    [[True, False, True], [False, True, False]],
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_flash_attn4_mixed_causal(
    seq_lens: list[tuple[int, int]],
    per_seq_causal: list[bool],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
):
    if not current_platform.is_cuda():
        pytest.skip("Flash attention requires CUDA")

    try:
        from vllm.vllm_flash_attn import (
            fa_version_unsupported_reason,
            flash_attn_varlen_func,
            is_fa_version_supported,
        )
    except ImportError:
        pytest.skip("vllm_flash_attn not available")

    if not is_fa_version_supported(4):
        reason = fa_version_unsupported_reason(4)
        pytest.skip(f"FA4 not supported: {reason}")

    set_random_seed(42)
    device = "cuda"

    num_query_heads, num_kv_heads = num_heads
    assert len(seq_lens) == len(per_seq_causal)

    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    num_seqs = len(seq_lens)

    num_query_tokens = sum(query_lens)
    max_kv_len = max(kv_lens)
    max_num_blocks = (max_kv_len + block_size - 1) // block_size
    num_blocks = max_num_blocks * num_seqs + 10

    scale = head_size**-0.5
    query = torch.randn(
        num_query_tokens, num_query_heads, head_size, dtype=dtype, device=device
    )
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    value_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )

    block_tables_list = []
    for i in range(num_seqs):
        n_blocks = (kv_lens[i] + block_size - 1) // block_size
        blocks = list(range(i * max_num_blocks, i * max_num_blocks + n_blocks))
        blocks += [0] * (max_num_blocks - n_blocks)
        block_tables_list.append(blocks)
    block_tables = torch.tensor(block_tables_list, dtype=torch.int32, device=device)

    cu_seqlens_q = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    for i, ql in enumerate(query_lens):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + ql

    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    per_seq_causal_tensor = torch.tensor(
        per_seq_causal, dtype=torch.int32, device=device
    )

    ref_output = ref_paged_attn(
        query,
        key_cache,
        value_cache,
        query_lens,
        kv_lens,
        block_tables,
        scale,
        per_seq_causal,
    )

    output = torch.empty_like(query)
    flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(query_lens),
        seqused_k=seqused_k,
        max_seqlen_k=max(kv_lens),
        softmax_scale=scale,
        # The kernel must be compiled causal for `dynamic_causal` to take effect.
        causal=True,
        block_table=block_tables,
        softcap=0.0,
        dynamic_causal=per_seq_causal_tensor,
        fa_version=4,
    )

    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)


# ---- FlashInfer TRTLLM-Gen test (per-token windows, causal+dense merge) ----


@pytest.mark.parametrize("causal_dtype", [torch.bool, torch.int32])
def test_flashinfer_mixed_causal_metadata(causal_dtype: torch.dtype):
    from vllm.v1.attention.backends.flashinfer import _fill_mixed_causal_metadata

    device = "cuda"
    window_left = 3
    # (query_len, seq_len, causal); the empty last request is CUDA graph padding.
    reqs = [(4, 10, True), (6, 6, False), (1, 9, False), (2, 12, True), (0, 0, False)]
    num_tokens = 16  # 13 query tokens + 3 padding tokens
    query_lens, seq_lens, causal = zip(*reqs)
    query_start_loc = torch.tensor([0, *query_lens], device=device).cumsum(0)
    bounds = torch.full((2, num_tokens), -1, dtype=torch.int32, device=device)
    token_causal = torch.ones(num_tokens, dtype=torch.bool, device=device)
    _fill_mixed_causal_metadata(
        query_start_loc.to(torch.int32),
        torch.tensor(seq_lens, dtype=torch.int32, device=device),
        torch.tensor(causal, dtype=causal_dtype, device=device),
        num_tokens,
        {window_left: bounds},
        token_causal,
    )

    expected_bounds, expected_causal = [], []
    for query_len, seq_len, is_causal in reqs:
        for pos in range(seq_len - query_len, seq_len):
            visible = [
                k
                for k in range(seq_len)
                if abs(k - pos) <= window_left and (k <= pos or not is_causal)
            ]
            expected_bounds.append([visible[0], visible[-1]])
            expected_causal.append(is_causal)
    num_padding = num_tokens - len(expected_causal)
    expected_bounds += [[0, 0]] * num_padding
    expected_causal += [False] * num_padding
    assert bounds.T.tolist() == expected_bounds
    assert token_causal.tolist() == expected_causal


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="FlashInfer TRTLLM-Gen attention requires an SM100-family GPU.",
)
@pytest.mark.parametrize(
    "per_seq_causal", [[True, False, True, False], [False, True, False, True]]
)
@pytest.mark.parametrize(
    "head_size, num_heads, sliding_window",
    # Sliding layers use per-token windows; full layers merge causal and dense.
    [(256, (16, 8), 32), (512, (16, 2), None)],
)
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, current_platform.fp8_dtype()])
@torch.inference_mode()
def test_flashinfer_trtllm_mixed_causal(
    per_seq_causal: list[bool],
    head_size: int,
    num_heads: tuple[int, int],
    sliding_window: int | None,
    kv_dtype: torch.dtype,
):
    from flashinfer.prefill import trtllm_batch_context_with_kv_cache

    from vllm.v1.attention.backends.flashinfer import _fill_mixed_causal_metadata

    set_random_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    block_size = 16
    # Bidirectional queries fit in the window; `ref_paged_attn` has no right bound.
    query_lens = [1, 24, 16, 1]
    kv_lens = [128, 300, 200, 256]
    num_seqs = len(query_lens)
    num_tokens = sum(query_lens)
    num_query_heads, num_kv_heads = num_heads
    scale = head_size**-0.5

    num_pages = [(kv_len + block_size - 1) // block_size for kv_len in kv_lens]
    max_num_pages = max(num_pages)
    block_tables = torch.arange(
        num_seqs * max_num_pages, dtype=torch.int32, device=device
    ).view(num_seqs, max_num_pages)
    # FP8 KV also runs the query in FP8, as the backend does.
    query = torch.randn(num_tokens, num_query_heads, head_size, device=device)
    query = query.to(kv_dtype)
    kv_cache = torch.randn(
        num_seqs * max_num_pages, 2, num_kv_heads, block_size, head_size, device=device
    ).to(kv_dtype)
    ref_output = ref_paged_attn(
        query.to(dtype),
        kv_cache[:, 0].transpose(1, 2).to(dtype).contiguous(),
        kv_cache[:, 1].transpose(1, 2).to(dtype).contiguous(),
        query_lens,
        kv_lens,
        block_tables,
        scale,
        per_seq_causal,
        sliding_window,
    )

    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, *query_lens], device=device).cumsum(0)
    query_start_loc = query_start_loc.to(torch.int32)
    kv_indptr = torch.tensor([0, *num_pages], device=device).cumsum(0)
    window_left = -1 if sliding_window is None else sliding_window - 1
    window_bounds = {}
    token_causal = None
    if sliding_window is None:
        token_causal = torch.empty(num_tokens, dtype=torch.bool, device=device)
    else:
        window_bounds[window_left] = torch.empty(
            2, num_tokens, dtype=torch.int32, device=device
        )
    _fill_mixed_causal_metadata(
        query_start_loc,
        seq_lens,
        torch.tensor(per_seq_causal, dtype=torch.int32, device=device),
        num_tokens,
        window_bounds,
        token_causal,
    )

    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.int8, device=device)

    def run(**kwargs) -> torch.Tensor:
        out = torch.empty(
            num_tokens, num_query_heads, head_size, dtype=dtype, device=device
        )
        trtllm_batch_context_with_kv_cache(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_q_len=max(query_lens),
            max_kv_len=max(kv_lens),
            bmm1_scale=scale,
            bmm2_scale=1.0,
            batch_size=num_seqs,
            cum_seq_lens_q=query_start_loc,
            cum_seq_lens_kv=kv_indptr.to(torch.int32),
            out=out,
            **kwargs,
        )
        return out

    if token_causal is not None:
        output = torch.where(token_causal[:, None, None], run(), run(causal=False))
    else:
        starts, ends = window_bounds[window_left]
        output = run(
            variable_window_token_starts=starts, variable_window_token_ends=ends
        )

    atol, rtol = (6e-2, 2e-2) if kv_dtype != dtype else (1e-2, 1e-2)
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)
