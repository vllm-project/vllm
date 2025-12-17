# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import helion
import helion.language as hl
import torch

from vllm.triton_utils import next_power_of_2

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
    num_seqs,
    q_block_padded_size,
):
    max_seqlen = t_seq_lens.max()
    max_query_len = t_query_start_lens.diff().max()
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


nv_configs = [
    helion.Config(
        block_sizes=[32, 2],
        indexing=[
            "pointer",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "pointer",
            "tensor_descriptor",
        ],
        l2_groupings=[4],
        load_eviction_policies=["first", "last", "first", "first", "", "last", "last"],
        loop_orders=[[1, 0]],
        num_stages=7,
        num_warps=8,
        pid_type="flat",
        range_flattens=[None, False, None],
        range_multi_buffers=[None, True, False],
        range_num_stages=[0, 2, 1],
        range_unroll_factors=[0, 2, 1],
        range_warp_specializes=[],
    ),
    helion.Config(
        block_sizes=[64, 4],
        indexing=[
            "pointer",
            "tensor_descriptor",
            "pointer",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "tensor_descriptor",
            "tensor_descriptor",
        ],
        l2_groupings=[16],
        load_eviction_policies=["", "last", "first", "last", "", "last", "last"],
        loop_orders=[[1, 0]],
        num_stages=7,
        num_warps=8,
        pid_type="flat",
        range_flattens=[None, None, False],
        range_multi_buffers=[None, False, None],
        range_num_stages=[0, 3, 1],
        range_unroll_factors=[0, 3, 0],
        range_warp_specializes=[],
    ),
    helion.Config(
        block_sizes=[32, 4],
        indexing=[
            "pointer",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "pointer",
            "tensor_descriptor",
        ],
        l2_groupings=[4],
        load_eviction_policies=["", "last", "first", "first", "last", "last", "last"],
        loop_orders=[[1, 0]],
        num_stages=7,
        num_warps=8,
        pid_type="persistent_interleaved",
        range_flattens=[False, False, None],
        range_multi_buffers=[False, True, False],
        range_num_stages=[1, 2, 1],
        range_unroll_factors=[2, 1, 1],
        range_warp_specializes=[],
    ),
    helion.Config(
        block_sizes=[32, 8],
        indexing=[
            "pointer",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "pointer",
            "tensor_descriptor",
        ],
        l2_groupings=[4],
        load_eviction_policies=["", "last", "first", "first", "", "last", ""],
        loop_orders=[[0, 1]],
        num_stages=7,
        num_warps=8,
        pid_type="flat",
        range_flattens=[None, False, None],
        range_multi_buffers=[None, True, False],
        range_num_stages=[0, 2, 1],
        range_unroll_factors=[0, 2, 1],
        range_warp_specializes=[],
    ),
    helion.Config(
        block_sizes=[32, 16],
        indexing=[
            "pointer",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "pointer",
            "tensor_descriptor",
        ],
        l2_groupings=[4],
        load_eviction_policies=["", "last", "first", "first", "", "last", "last"],
        loop_orders=[[1, 0]],
        num_stages=7,
        num_warps=8,
        pid_type="flat",
        range_flattens=[None, False, None],
        range_multi_buffers=[None, True, False],
        range_num_stages=[0, 2, 1],
        range_unroll_factors=[0, 1, 1],
        range_warp_specializes=[],
    ),
    helion.Config(
        block_sizes=[1, 2],
        indexing=[
            "pointer",
            "tensor_descriptor",
            "tensor_descriptor",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "tensor_descriptor",
            "tensor_descriptor",
        ],
        l2_groupings=[16],
        load_eviction_policies=["", "last", "first", "last", "last", "last", "last"],
        loop_orders=[[1, 0]],
        num_stages=8,
        num_warps=8,
        pid_type="flat",
        range_flattens=[None, None, False],
        range_multi_buffers=[None, False, None],
        range_num_stages=[0, 3, 1],
        range_unroll_factors=[0, 3, 0],
        range_warp_specializes=[],
    ),
]
amd_configs = [
    helion.Config(
        block_sizes=[32, 2],
        indexing=[
            "tensor_descriptor",
            "tensor_descriptor",
            "pointer",
            "pointer",
            "tensor_descriptor",
            "tensor_descriptor",
            "pointer",
            "pointer",
        ],
        l2_groupings=[4],
        load_eviction_policies=["first", "last", "first", "", "", "", "last"],
        loop_orders=[[0, 1]],
        matrix_instr_nonkdim=0,
        num_stages=5,
        num_warps=2,
        pid_type="persistent_blocked",
        range_flattens=[None, True, True],
        range_multi_buffers=[True, None, False],
        range_num_stages=[1, 0, 0],
        range_unroll_factors=[1, 1, 0],
        range_warp_specializes=[],
        waves_per_eu=1,
    ),
    helion.Config(
        block_sizes=[32, 1],
        indexing=[
            "tensor_descriptor",
            "tensor_descriptor",
            "pointer",
            "pointer",
            "tensor_descriptor",
            "tensor_descriptor",
            "pointer",
            "pointer",
        ],
        l2_groupings=[4],
        load_eviction_policies=["first", "last", "first", "", "", "", "last"],
        loop_orders=[[0, 1]],
        matrix_instr_nonkdim=0,
        num_stages=5,
        num_warps=2,
        pid_type="persistent_blocked",
        range_flattens=[None, True, True],
        range_multi_buffers=[True, None, None],
        range_num_stages=[1, 0, 0],
        range_unroll_factors=[1, 1, 0],
        range_warp_specializes=[],
        waves_per_eu=1,
    ),
    helion.Config(
        block_sizes=[64, 1],
        indexing=[
            "pointer",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "pointer",
            "pointer",
            "tensor_descriptor",
            "tensor_descriptor",
        ],
        l2_groupings=[1],
        load_eviction_policies=["last", "first", "first", "last", "first", "", "last"],
        loop_orders=[[1, 0]],
        matrix_instr_nonkdim=0,
        num_stages=8,
        num_warps=16,
        pid_type="flat",
        range_flattens=[None, None, True],
        range_multi_buffers=[None, None, False],
        range_num_stages=[0, 0, 2],
        range_unroll_factors=[0, 1, 0],
        range_warp_specializes=[],
        waves_per_eu=1,
    ),
    helion.Config(
        block_sizes=[64, 2],
        indexing=[
            "tensor_descriptor",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "pointer",
            "pointer",
            "pointer",
            "tensor_descriptor",
        ],
        l2_groupings=[16],
        load_eviction_policies=["last", "", "last", "", "first", "last", ""],
        loop_orders=[[0, 1]],
        matrix_instr_nonkdim=0,
        num_stages=8,
        num_warps=8,
        pid_type="persistent_blocked",
        range_flattens=[None, True, None],
        range_multi_buffers=[True, None, None],
        range_num_stages=[1, 2, 1],
        range_unroll_factors=[3, 2, 2],
        range_warp_specializes=[],
        waves_per_eu=1,
    ),
    helion.Config(
        block_sizes=[64, 2],
        indexing=[
            "pointer",
            "pointer",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
        ],
        l2_groupings=[32],
        load_eviction_policies=["first", "", "first", "first", "", "last", ""],
        loop_orders=[[0, 1]],
        matrix_instr_nonkdim=0,
        num_stages=2,
        num_warps=8,
        pid_type="flat",
        range_flattens=[None, None, True],
        range_multi_buffers=[None, False, None],
        range_num_stages=[0, 4, 1],
        range_unroll_factors=[0, 2, 3],
        range_warp_specializes=[],
        waves_per_eu=1,
    ),
    helion.Config(
        block_sizes=[1, 8],
        indexing=[
            "tensor_descriptor",
            "pointer",
            "pointer",
            "pointer",
            "pointer",
            "pointer",
            "pointer",
            "tensor_descriptor",
        ],
        l2_groupings=[8],
        load_eviction_policies=["last", "", "first", "last", "last", "last", "last"],
        loop_orders=[[1, 0]],
        matrix_instr_nonkdim=0,
        num_stages=1,
        num_warps=4,
        pid_type="flat",
        range_flattens=[None, True, False],
        range_multi_buffers=[None, None, False],
        range_num_stages=[0, 2, 1],
        range_unroll_factors=[0, 3, 2],
        range_warp_specializes=[],
        waves_per_eu=1,
    ),
    helion.Config(
        block_sizes=[1, 2],
        indexing=[
            "pointer",
            "tensor_descriptor",
            "pointer",
            "pointer",
            "pointer",
            "pointer",
            "pointer",
            "tensor_descriptor",
        ],
        l2_groupings=[8],
        load_eviction_policies=["", "", "first", "last", "last", "last", "last"],
        loop_orders=[[1, 0]],
        matrix_instr_nonkdim=0,
        num_stages=1,
        num_warps=4,
        pid_type="flat",
        range_flattens=[None, True, False],
        range_multi_buffers=[None, True, False],
        range_num_stages=[0, 2, 1],
        range_unroll_factors=[0, 3, 2],
        range_warp_specializes=[],
        waves_per_eu=1,
    ),
]

configs = nv_configs if torch.version.cuda else amd_configs


@helion.kernel(
    allow_warp_specialize=True,
    static_shapes=False,
    index_dtype=torch.int64,
    configs=configs,
    autotune_baseline_fn=_triton_baseline_fn,
    autotune_effort="quick",
    # for in-place autotuning, not recommended for micro-benchmarks
    #  or initial config selection
    autotune_accuracy_check=False,
    autotune_ignore_errors=True,
    print_repro=False,
    print_output_code=False,
    # useful for debugging
    # debug_dtype_asserts=True,
)
def kernel_helion_v5_attention(
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
    num_seqs,  # must be on cpu
    # to trigger re-compilation (and re-tuning) for decodes only
    q_block_padded_size: hl.constexpr,
):
    head_size = hl.specialize(t_query.size(2))
    num_kv_heads = hl.specialize(t_key_cache.size(2))
    num_query_heads = hl.specialize(t_query.size(1))
    page_size = hl.specialize(t_value_cache.size(1))
    num_queries_per_kv = hl.specialize(num_query_heads // num_kv_heads)

    assert page_size == t_key_cache.size(1)
    assert head_size == t_key_cache.size(3)

    q_block_size = hl.register_block_size(1, q_block_padded_size)
    num_pages_at_once = hl.register_block_size(1, 32)

    for seq_tile, tile_m in hl.tile(
        [num_seqs, num_query_heads],
        block_size=[1, num_queries_per_kv],
    ):
        seq_idx = seq_tile.begin  # is scalar
        seq_len = t_seq_lens[seq_idx]
        query_start = t_query_start_lens[seq_idx]
        query_end = t_query_start_lens[seq_idx + 1]
        query_len = query_end - query_start
        context_len = seq_len - query_len

        for tile_q in hl.tile(query_len, block_size=q_block_size):
            block_m_size = num_queries_per_kv * q_block_size
            kv_head_idx = tile_m.begin // num_queries_per_kv

            # cannot use tile_q.index directly, since tile_q.index is dynamic
            adjusted_tile_q_index = query_start + tile_q.begin + hl.arange(q_block_size)
            query_head_offset = tile_m.begin + hl.arange(num_queries_per_kv)
            q_load_mask = adjusted_tile_q_index[:, None, None] < query_end
            # (tile_q, tile_m, HEAD_SIZE)
            q = hl.load(
                t_query,
                [adjusted_tile_q_index, query_head_offset, hl.arange(head_size)],
                extra_mask=q_load_mask,
            )
            # (tile_m, HEAD_SIZE)
            q = q.flatten(start_dim=0, end_dim=1)

            M = hl.full([block_m_size], float("-inf"), dtype=torch.float32)
            L = hl.full([block_m_size], 1.0, dtype=torch.float32)
            acc = hl.zeros([block_m_size, head_size], dtype=torch.float32)

            # adjust for causal mask
            max_seq_prefix_len = context_len + tile_q.begin + block_m_size + 1
            max_seq_prefix_len = torch.minimum(max_seq_prefix_len, seq_len)
            num_blocks = torch.ceil(max_seq_prefix_len / page_size)
            for tile_n in hl.tile(num_blocks, block_size=num_pages_at_once):
                block_n_size = num_pages_at_once * page_size
                # explicit load due to wrong if tile_n is partial
                blk_idxs = hl.load(
                    t_block_tables,
                    [seq_idx, tile_n.begin + hl.arange(num_pages_at_once)],
                )
                blk_idxs = blk_idxs.view([num_pages_at_once]).to(torch.int64)

                # (tile_n, PAGE_SIZE, 1, HEAD_SIZE)
                k_load = t_key_cache[blk_idxs, :, kv_head_idx, :]
                k_load = k_load.flatten(start_dim=0, end_dim=1)
                # (tile_n, HEAD_SIZE)
                k = hl.zeros([block_n_size, head_size], dtype=k_load.dtype)
                absolute_tile_token_offsets = tile_n.begin * page_size + hl.arange(
                    block_n_size
                )
                k = torch.where(
                    absolute_tile_token_offsets[:, None] < seq_len, k_load, k
                )
                # (HEAD_SIZE, tile_n)
                k = k.transpose(0, 1)

                # (tile_n, PAGE_SIZE, HEAD_SIZE)
                v_load = t_value_cache[blk_idxs, :, kv_head_idx, :]
                v_load = v_load.flatten(start_dim=0, end_dim=1)
                # (tile_n, HEAD_SIZE)
                v = hl.zeros([block_n_size, head_size], dtype=v_load.dtype)
                v = torch.where(
                    absolute_tile_token_offsets[:, None] < seq_len, v_load, v
                )

                # (tile_m, tile_n)
                # use S with float32 as acc to enforce higher precision
                #  for the additions of the dot operation?
                S = hl.zeros([block_m_size, block_n_size], dtype=torch.float32)
                S = hl.dot(q, k, out_dtype=torch.float32, acc=S) * scale
                block_m_query_mask = tile_q.begin + hl.arange(
                    q_block_size
                ).repeat_interleave(num_queries_per_kv, dim=0)
                # construct 2d causal mask
                causal_mask = (
                    absolute_tile_token_offsets[None, :]
                    < context_len + block_m_query_mask[:, None] + 1
                )
                S = torch.where(causal_mask, S, float("-inf"))

                # (tile_m)
                M_j = torch.maximum(M, torch.amax(S, 1))
                # (tile_m, tile_n)
                P = torch.exp(S - M_j[:, None])
                # (tile_m, )
                L_j = torch.sum(P, 1)
                # (tile_m, )
                alpha = torch.exp(M - M_j)
                # (tile_m, HEAD_SIZE)
                acc = acc * alpha[:, None]
                L = (L * alpha) + L_j
                M = M_j

                # (tile_m, HEAD_SIZE)
                acc = hl.dot(P.to(v.dtype), v, out_dtype=torch.float32, acc=acc)

            # epilogue
            acc = acc / L[:, None]
            hl.store(
                t_output,
                [adjusted_tile_q_index, tile_m.index, hl.arange(head_size)],
                acc.view([q_block_size, num_queries_per_kv, head_size]),
                extra_mask=q_load_mask,
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

    max_used_querylen_padded = (
        1 if max_seqlen_q == 1 else next_power_of_2(max(64, max_seqlen_q))
    )

    kernel_helion_v5_attention(
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
        num_seqs=num_seqs,
        q_block_padded_size=max_used_querylen_padded,
    )
