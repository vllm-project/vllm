# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from tests.kernels.attention.utils import ref_single_query_cached_kv_attention
from vllm.platforms import current_platform
from vllm.v1.attention.backends.cpu_attn import _PagedAttention

NUM_BLOCKS = 4321  # Arbitrary values for testing
BLOCK_SIZE = 16


# TODO: Make this token-granular once paged_attn API/impls support it
def get_block_granular_sliding_window_mask(
    kv_len: int, window: int | None, block_size: int, dtype
) -> torch.Tensor | None:
    # [1,1,1,kv_len] additive mask,
    # masks out the first N full blocks outside the window
    if not window or window >= kv_len:
        return None
    num_blocks_to_skip = (kv_len - window) // block_size
    num_tokens_to_skip = num_blocks_to_skip * block_size
    keep = min(kv_len, kv_len - num_tokens_to_skip)
    m = torch.zeros((1, 1, kv_len), dtype=dtype)
    m[..., : kv_len - keep] = float("-inf")
    return m


@pytest.mark.parametrize(
    "sliding_window", [0, None, BLOCK_SIZE - 1, BLOCK_SIZE, BLOCK_SIZE + 1, 128]
)
@pytest.mark.parametrize(
    "seq_lens",
    [[42], [19, 21, 127, 129], [1234, BLOCK_SIZE - 1, BLOCK_SIZE + 1, BLOCK_SIZE]],
)
@pytest.mark.parametrize("num_heads", [(8, 8), (64, 8)])
@pytest.mark.parametrize("head_size", [32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seed", [42])
def test_paged_attention(
    kv_cache_factory,
    sliding_window: int,
    seq_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    device = "cpu"
    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    num_seqs = len(seq_lens)
    query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    alibi_slopes = None

    max_seq_len = max(seq_lens)
    # Create the block tables.
    max_num_blocks_per_seq = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables_lst: list[list[int]] = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1) for _ in range(max_num_blocks_per_seq)
        ]
        block_tables_lst.append(block_table)

    block_tables = torch.tensor(block_tables_lst, dtype=torch.int)

    kv_cache_dtype = "bfloat16" if dtype == torch.bfloat16 else "float"
    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(
        NUM_BLOCKS,
        BLOCK_SIZE,
        1,
        num_kv_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        seed,
        device,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Using default kv_scale
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Call the paged attention kernel.
    output = torch.empty_like(query)
    _PagedAttention.forward_decode(
        output,
        query,
        key_cache,
        value_cache,
        block_tables,
        torch.tensor(seq_lens, dtype=torch.int),
        max_seq_len,
        sliding_window,
        kv_cache_dtype,
        num_kv_heads,
        scale,
        None,  # alibi_slopes
        k_scale,
        v_scale,
    )
    # Run the reference implementation.
    ref_output = torch.empty_like(query)
    attn_masks = [
        get_block_granular_sliding_window_mask(
            seq_len, sliding_window, BLOCK_SIZE, torch.float32
        )
        for seq_len in seq_lens
    ]
    ref_single_query_cached_kv_attention(
        ref_output,
        query,
        num_queries_per_kv,
        key_cache,
        value_cache,
        block_tables,
        torch.tensor(seq_lens, dtype=torch.int),
        scale,
        alibi_slopes,
        attn_masks,
    )

    atol = get_default_atol(output)
    rtol = get_default_rtol(output)

    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)
