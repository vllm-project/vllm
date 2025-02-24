# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from utils import (BlockDiagonalCausalFromBottomRightMask,
                   get_active_block_tables, ref_context_attention,
                   sample_inputs)


@pytest.mark.parametrize(
    "prefill_batch_size,decode_batch_size,block_size,large_tile_size",
    [
        (1, 199, 1, 512),  # 512 blocks
        (4, 12, 256, 2048),  # 128 blocks
        (4, 12, 16, 2048),  # 128 blocks
        (4, 12, 4, 1024),  # 256 blocks
        (4, 12, 32, 2048),  # 64 blocks
        (4, 12, 32, 4096),  # 128 blocks
        (4, 12, 32, 8192),  # 256 blocks
        (4, 12, 64, 8192),  # 128 blocks
    ],
)
@pytest.mark.parametrize(
    "num_heads,num_queries_per_kv,head_size",
    [
        (4, 2, 8),
        (32, 8, 64),
        (4, 4, 128),
        (8, 1, 32),
    ],
)
@pytest.mark.parametrize("mixed_precision", [True, False])
@torch.inference_mode()
def test_contexted_kv_attention(
    prefill_batch_size: int,
    decode_batch_size: int,
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    block_size: int,
    large_tile_size,
    mixed_precision: bool,
) -> None:
    import os

    import torch_xla.core.xla_model as xm

    from vllm.attention.ops.nki_flash_attn import (flash_attn_varlen_nkifunc,
                                                   reorder_context_mask)

    assert large_tile_size % block_size == 0

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

    output_ref = ref_context_attention(
        query,
        key,
        value,
        query_lens,
        seq_lens,
        head_size,
        num_queries_per_kv,
        return_max_reduce=False,
    )

    # build neuron program
    B_P_SIZE = 128
    assert (large_tile_size >= B_P_SIZE
            ), f"Expect {large_tile_size=} to be larger than {B_P_SIZE=}"

    def ceil_div(a, b):
        return (a + b - 1) // b

    def pad_to_multiple(a, b):
        return ceil_div(a, b) * b

    def pad_to_next_power_of_2(a):
        return 2**int(a - 1).bit_length() if a > 0 else 1

    # calculate input shapes
    max_num_queries = pad_to_next_power_of_2(sum(query_lens))
    context_lens = torch.tensor(seq_lens) - torch.tensor(query_lens)
    num_active_blocks = ceil_div(context_lens, block_size).sum().item()
    num_active_blocks = pad_to_multiple(num_active_blocks,
                                        large_tile_size // block_size)
    context_kv_len = num_active_blocks * block_size
    assert (context_kv_len %
            large_tile_size == 0), f"invalid context_kv_len={context_kv_len}"

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
    prior_mask, active_mask = (
        BlockDiagonalCausalFromBottomRightMask.from_seqlens(
            query_lens, seq_lens, block_size=block_size))
    prior_mask_padded = F.pad(
        prior_mask,
        (
            0,
            context_kv_len - prior_mask.shape[1],
            0,
            max_num_queries - prior_mask.shape[0],
        ),
        "constant",
        0,
    ).bool()
    active_mask_padded = F.pad(
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
    attn_mask = torch.concat([prior_mask_padded, active_mask_padded], dim=1)

    attn_mask = reorder_context_mask(attn_mask, large_tile_size, block_size)

    input_args = (
        query.to(device=device),
        k.to(device=device),
        v.to(device=device),
        k_cache.to(device=device),
        v_cache.to(device=device),
        active_block_table.to(device=device),
        attn_mask.to(device=device),
    )
    input_kwargs = dict(
        n_kv_head=num_kv_heads,
        head_size=head_size,
        mixed_precision=mixed_precision,
        LARGE_TILE_SZ=large_tile_size,
    )

    output_nki = flash_attn_varlen_nkifunc(*input_args, **input_kwargs)

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
