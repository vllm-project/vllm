# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import helion
import helion.language as hl
import torch

from .triton_unified_attention import (
    unified_attention as triton_baseline_unified_attention,
)


def _triton_baseline_fn(
    t_output,  # [num_tokens, num_query_heads, head_size]
    t_query,  # [num_tokens, num_query_heads, head_size]
    t_key_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_value_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_block_tables,  # [num_seqs, max_num_blocks_per_seq]
    t_seq_lens,  # [num_seqs]
    scale,
    # k_scale,
    # v_scale,
    t_query_start_lens,  # [num_seqs+1]
    max_query_len,
    num_seqs,
):
    max_seqlen = t_seq_lens.max()
    return triton_baseline_unified_attention(
        q=t_query,
        k=t_key_cache,
        v=t_value_cache,
        out=t_output,
        cu_seqlens_q=t_query_start_lens,
        max_seqlen_q=max_query_len,
        seqused_k=t_seq_lens,
        max_seqlen_k=max_seqlen,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=t_block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )


nv_config = helion.Config(
    block_sizes=[32, 4],
    indexing=[
        "pointer",
        "pointer",
        "pointer",
        "pointer",
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
    ],
    l2_groupings=[2],
    load_eviction_policies=["", "", "", "", "", "last", "last", ""],
    loop_orders=[[1, 2, 0], [1, 0]],
    num_stages=6,
    num_warps=8,
    pid_type="flat",
    range_flattens=[None, True, True, True],
    range_multi_buffers=[None, None, None, False],
    range_num_stages=[],
    range_unroll_factors=[0, 1, 2, 1],
    range_warp_specializes=[],
)
amd_config = helion.Config(
    block_sizes=[32, 8],
    indexing=[
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
        "pointer",
        "pointer",
        "pointer",
        "pointer",
        "pointer",
    ],
    l2_groupings=[1],
    load_eviction_policies=["", "", "", "", "", "", ""],
    loop_orders=[[2, 1, 0], [0, 1]],
    num_stages=1,
    num_warps=4,
    pid_type="flat",
    range_flattens=[None, None, None, None],
    range_multi_buffers=[None, None, None, None],
    range_num_stages=[],
    range_unroll_factors=[0, 0, 0, 0],
    range_warp_specializes=[],
)

config = nv_config if torch.version.cuda else amd_config


@helion.kernel(
    allow_warp_specialize=True,
    # dot_precision='ieee',
    config=config,
    autotune_baseline_fn=_triton_baseline_fn,
    # autotune_effort="quick",
    static_shapes=False,
    print_output_code=False,
    print_repro=False,
    index_dtype=torch.int64,
)
def kernel_helion_v2_attention(
    t_output,  # [num_tokens, num_query_heads, head_size]
    t_query,  # [num_tokens, num_query_heads, head_size]
    t_key_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_value_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_block_tables,  # [num_seqs, max_num_blocks_per_seq]
    t_seq_lens,  # [num_seqs]
    scale,
    # k_scale,
    # v_scale,
    t_query_start_lens,  # [num_seqs+1]
    max_query_len,  # must be on cpu
    num_seqs,  # on cpu?
    # max_used_querylen_padded: hl.constexpr,
):
    head_size = hl.specialize(t_query.size(2))
    num_kv_heads = hl.specialize(t_key_cache.size(2))
    num_query_heads = hl.specialize(t_query.size(1))
    page_size = hl.specialize(t_value_cache.size(1))
    num_queries_per_kv = hl.specialize(num_query_heads // num_kv_heads)
    # max_used_querylen_padded = hl.specialize(max_used_querylen_padded)

    assert page_size == t_key_cache.size(1)
    assert head_size == t_key_cache.size(3)

    q_block_size = hl.register_block_size(4, int(max_query_len))
    max_qblocks = (max_query_len + q_block_size - 1) // q_block_size
    # q_block_size = hl.register_block_size(4, int(max_used_querylen_padded))
    # max_qblocks = (max_used_querylen_padded + q_block_size -1) // q_block_size

    for seq_idx, kv_head_idx, qblock_idx in hl.grid(
        [num_seqs, num_kv_heads, max_qblocks]
    ):
        seq_len = t_seq_lens[seq_idx]
        query_start = t_query_start_lens[seq_idx]
        query_end = t_query_start_lens[seq_idx + 1]
        query_len = query_end - query_start
        context_len = seq_len - query_len

        cur_qblock_start = query_start + qblock_idx * q_block_size
        cur_qblock_end = torch.minimum(
            query_start + (qblock_idx + 1) * q_block_size, query_end
        )

        for tile_q, tile_m in hl.tile(
            [cur_qblock_start, kv_head_idx * num_queries_per_kv],
            [cur_qblock_end, (kv_head_idx + 1) * num_queries_per_kv],
            block_size=[q_block_size, num_queries_per_kv],
        ):
            block_m_size = tile_m.block_size * tile_q.block_size

            # (tile_q, tile_m, HEAD_SIZE)
            # # tile_q is masked here
            q = t_query[tile_q, tile_m, :]
            # (tile_m, HEAD_SIZE)
            q = q.view([block_m_size, head_size])

            M = hl.full(
                [block_m_size], float("-inf"), dtype=torch.float32
            )  # device=q.device)
            L = hl.full([block_m_size], 1.0, dtype=torch.float32)
            acc = hl.zeros(
                [block_m_size, head_size], dtype=torch.float32
            )  # , device=q.device)

            # adjust for causal mask
            max_seq_prefix_len = (
                context_len
                + tile_q.end
                + (tile_m.block_size + num_queries_per_kv - 1) // num_queries_per_kv
            )
            max_seq_prefix_len = torch.minimum(max_seq_prefix_len, seq_len)
            num_blocks = torch.ceil(max_seq_prefix_len / page_size)
            for tile_n in hl.tile(num_blocks, block_size=None):
                block_n_size = tile_n.block_size * page_size
                blk_idxs = t_block_tables[seq_idx, tile_n].view([tile_n.block_size])
                # (tile_n, PAGE_SIZE, 1, HEAD_SIZE)
                k = t_key_cache[blk_idxs, :, kv_head_idx, :]
                # DEBUG: to assert shape
                # k = k.view([tile_n, page_size, head_size])
                # (tile_n, PAGE_SIZE, HEAD_SIZE)
                v = t_value_cache[blk_idxs, :, kv_head_idx, :]
                # (HEAD_SIZE, tile_n)
                k = k.view([block_n_size, head_size]).transpose(0, 1)
                # (tile_m, tile_n)
                qk = torch.mm(q, k) * scale
                # DEBUG: to check the shape...
                # qk = qk.view([block_m_size, block_n_size])
                # (tile_m)
                M_j = torch.maximum(M, torch.amax(qk, 1))
                # (tile_m, tile_n)
                P = torch.exp(qk - M_j[:, None])
                # (tile_m, )
                L_j = torch.sum(P, 1)
                # (tile_m, )
                alpha = torch.exp(M - M_j)
                # (tile_m, HEAD_SIZE)
                acc *= alpha[:, None]
                L *= alpha + L_j
                M = M_j

                # (tile_n, HEAD_SIZE)
                v_view = v.view([block_n_size, head_size])
                # (tile_m, HEAD_SIZE)
                acc += torch.mm(P.to(v.dtype), v_view)

            # epilogue
            acc = acc / L[:, None]
            t_output[tile_q, tile_m, :] = acc.view(
                [tile_q.block_size, tile_m.block_size, head_size]
            )


def helion_unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    max_query_len_int: int,
    num_seqs: int,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    assert alibi_slopes is None, "not supported right now, still experimental"
    assert softcap == 0, "not supported right now, still experimental"
    assert k_descale is None, "not supported right now, still experimental"
    assert v_descale is None, "not supported right now, still experimental"
    assert window_size == (-1, -1), "not supported right now, still experimental"

    block_size = v.shape[1]
    assert q.element_size() >= 2 or block_size >= 32, (
        "Block size must be at least 32 for fp8"
    )

    # max_used_querylen_padded = max_query_len_int if max_query_len_int == 1
    #   else torch._inductor.runtime.runtime_utils.next_power_of_2(
    #     max(16, max_query_len_int))

    kernel_helion_v2_attention(
        t_output=out,
        t_query=q,
        t_key_cache=k,
        t_value_cache=v,
        t_block_tables=block_table,
        t_seq_lens=seqused_k,
        scale=softmax_scale,
        # k_scale=k_descale,
        # v_scale=v_descale,
        t_query_start_lens=cu_seqlens_q,
        max_query_len=max_query_len_int,  # need not to be a tensor
        # max_used_querylen_padded = int(max_used_querylen_padded),
        num_seqs=num_seqs,
    )
