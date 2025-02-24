# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from utils import (BlockDiagonalCausalFromBottomRightMask, ceil_div,
                   get_active_block_tables, pad_to_multiple,
                   pad_to_next_power_of_2, ref_context_attention,
                   sample_inputs)

from vllm.attention.ops.nki_blocksparse_flash_attn import (
    FlashAttentionPlanner, flash_attn_varlen_blocksparse_nkifunc)


@pytest.mark.parametrize(
    "prefill_batch_size,decode_batch_size,tile_size_q,tile_size_kv,block_size",
    [
        (1, 33, 128, 512, 1),  # 512 blocks
        (4, 12, 256, 2048, 256),  # 128 blocks
        (4, 12, 128, 2048, 16),  # 128 blocks
        (4, 12, 256, 1024, 4),  # 256 blocks
        (4, 12, 128, 2048, 32),  # 64 blocks
        (4, 12, 256, 4096, 32),  # 128 blocks
        (4, 12, 128, 8192, 32),  # 256 blocks
        (4, 12, 256, 8192, 64),  # 128 blocks
    ],
)
@pytest.mark.parametrize(
    "num_heads,num_queries_per_kv,head_size",
    [
        (4, 2, 16),
        (32, 8, 64),
        (4, 4, 128),
        (8, 1, 32),
    ],
)
@pytest.mark.parametrize("mixed_precision", [True, False])
@torch.inference_mode()
def test_blocksparse_flash_paged_attention(
    prefill_batch_size: int,
    decode_batch_size: int,
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    block_size: int,
    tile_size_q: int,
    tile_size_kv: int,
    mixed_precision: bool,
) -> None:
    assert tile_size_kv % block_size == 0

    device = xm.xla_device()

    compiler_flags = [
        "-O1",
        "--retry_failed_compilation",
    ]
    compiler_flags_str = " ".join(compiler_flags)
    os.environ["NEURON_CC_FLAGS"] = compiler_flags_str

    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False)
    dtype = torch.float32

    min_ctx_len = 32
    max_ctx_len = 1024
    min_query_len = 16
    max_query_len = 512
    num_kv_heads = num_heads // num_queries_per_kv
    (
        query,
        k_active,
        v_active,
        k_cache,
        v_cache,
        block_table,
        key,
        value,
        query_lens,
        seq_lens,
    ) = sample_inputs(
        prefill_batch_size=prefill_batch_size,
        decode_batch_size=decode_batch_size,
        min_query_len=min_query_len,
        max_query_len=max_query_len,
        min_ctx_len=min_ctx_len,
        max_ctx_len=max_ctx_len,
        block_size=block_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
    )

    output_ref, *_ = ref_context_attention(
        query,
        key,
        value,
        query_lens,
        seq_lens,
        head_size,
        num_queries_per_kv,
        return_max_reduce=True,
    )

    # build neuron program
    B_P_SIZE = 128
    LARGE_KV_TILE_SZ = tile_size_kv
    assert LARGE_KV_TILE_SZ >= B_P_SIZE

    # calculate input shapes
    max_num_queries = pad_to_next_power_of_2(sum(query_lens))
    context_lens = torch.tensor(seq_lens) - torch.tensor(query_lens)
    num_active_blocks = ceil_div(context_lens, block_size).sum().item()
    num_active_blocks = pad_to_multiple(num_active_blocks,
                                        LARGE_KV_TILE_SZ // block_size)
    context_kv_len = num_active_blocks * block_size
    assert (context_kv_len %
            LARGE_KV_TILE_SZ == 0), f"invalid context_kv_len={context_kv_len}"

    # pad QKV tensors
    pad_dims = (
        0,
        0,
        0,
        0,
        0,
        max_num_queries - query.shape[0],
    )
    query = F.pad(query, pad_dims, "constant", 0)
    k = F.pad(k_active, pad_dims, "constant", 0)
    v = F.pad(v_active, pad_dims, "constant", 0)

    # permute QKV tensors
    # query: (1, n_heads, d, seq_q)
    # key:   (1, n_kv_heads, d, seq_k)
    # value: (1, n_kv_heads, seq_v, d)
    query = query.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
    k = k.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
    v = v.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    k_cache = k_cache.permute(0, 2, 1, 3).contiguous()
    v_cache = v_cache.permute(0, 2, 1, 3).contiguous()
    # transform block table
    active_block_table = get_active_block_tables(
        block_table,
        torch.tensor(query_lens),
        torch.tensor(seq_lens),
        block_size,
        num_active_blocks,
    )

    # Build attention masks
    _, active_mask = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
        query_lens, seq_lens, block_size=block_size)
    active_mask = F.pad(
        active_mask,
        (
            0,
            max_num_queries - active_mask.shape[1],
            0,
            max_num_queries - active_mask.shape[0],
        ),
        "constant",
        0,
    ).bool()

    blocksparse_planner = FlashAttentionPlanner(
        np.array(query_lens, dtype=np.int32),
        context_lens.int().numpy(),
        tile_size_q=tile_size_q,
        tile_size_kv=tile_size_kv,
        block_size=block_size,
    )
    ctx_token_plan = blocksparse_planner.plan()
    tile_block_tables = ctx_token_plan.build_tile_block_tables(
        active_block_table)
    tile_masks = ctx_token_plan.build_tile_masks()

    input_args = (
        query.to(device=device),
        k.to(device=device),
        v.to(device=device),
        k_cache.to(device=device),
        v_cache.to(device=device),
        torch.tensor(ctx_token_plan.tile_q_indices).to(device=device),
        torch.tensor(tile_block_tables).to(device=device),
        torch.tensor(tile_masks).to(device=device),
        active_mask.to(device=device),
    )
    input_kwargs = dict(
        n_kv_head=num_kv_heads,
        head_size=head_size,
        mixed_precision=mixed_precision,
    )

    output_nki = flash_attn_varlen_blocksparse_nkifunc(*input_args,
                                                       **input_kwargs)

    num_actual_tokens = sum(query_lens)
    # - o: shape (bs, n_heads, seq_q, d) -> (bs, seq_q, n_heads, d)
    output_nki = output_nki.cpu().permute(0, 2, 1, 3)
    output_nki = output_nki[0, :num_actual_tokens, :, :]
    output_ref_padded = F.pad(
        output_ref,
        (0, 0, 0, 0, 0, 0, 0, max_num_queries - output_ref.shape[0]),
        "constant",
        0,
    )
    output_ref = output_ref_padded.transpose(0, 1)[0, :num_actual_tokens, :, :]

    torch.testing.assert_close(output_nki, output_ref, atol=1e-2, rtol=0)
