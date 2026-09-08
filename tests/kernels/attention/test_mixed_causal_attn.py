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
    if not current_platform.is_cuda():
        pytest.skip("Triton attention requires CUDA")

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
