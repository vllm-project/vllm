import pytest
import torch

from vllm.attention.backends.flash_attn import BatchLLM_attention
from vllm.platforms import current_platform
from vllm.utils import make_tensor_with_pad
from vllm.vllm_flash_attn import flash_attn_varlen_func

n_heads_q_n_heads_kv = [(4, 4), (8, 2), (16, 2)]
head_sizes = [128, 192, 256]
paged_block_size = [16]
DTYPES = [torch.float16, torch.bfloat16]
NUM_PAGE_BLOCKS = 2048


def create_seq_params(shared_request_count, request_count, shared_degree,
                      query_len_of_each_request, shared_prefix_length,
                      non_shared_context_length):
    # Construct the sequence parameters for Batchllm prefill scenarios.

    mega_seq_params = []
    cur_q_total_len = 0
    i = 0
    # prefix_sharing_group_count = shared_request_count // shared_degree
    while i < shared_request_count:
        # For each requests, there's a metainfo list with 3 elements:
        # inflight_qkv, shared_prefix and non_shared_context
        # Thus, real_q_length = inflight_qkv_length
        # real_kv_length = shared_prefix_length + non_shared_context_length
        # + inflight_qkv_length

        prefix_sharing_group = [[query_len_of_each_request, 
                                 shared_prefix_length, 
                                 non_shared_context_length]] * shared_degree

        mega_seq_params.append(prefix_sharing_group)
        i += shared_degree

    while i < request_count:
        single_request = [[
            query_len_of_each_request, 0, non_shared_context_length
        ]]
        mega_seq_params.append(single_request)
        i += 1
    return mega_seq_params


@pytest.mark.parametrize("n_heads_q_n_heads_kv", n_heads_q_n_heads_kv)
@pytest.mark.parametrize("head_size", head_sizes)
@pytest.mark.parametrize("paged_block_size", paged_block_size)
@pytest.mark.parametrize("request_with_shared_prefix_ratio", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("request_count", [16])
@pytest.mark.parametrize("shared_degree", [-1, 2])
@pytest.mark.parametrize("query_length_of_each_request", [32])
@pytest.mark.parametrize("shared_prefix_length", [64, 128, 256])
@pytest.mark.parametrize("non_shared_context_length", [0, 64, 128])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scenarios", ["decode", "prefill"])
@torch.inference_mode()
def test_batchllm_scenarios(
    n_heads_q_n_heads_kv,
    head_size,
    paged_block_size,
    request_with_shared_prefix_ratio,
    request_count,
    shared_degree,
    query_length_of_each_request,
    shared_prefix_length,
    non_shared_context_length,
    dtype,
    scenarios,
):
    """
    Test the prefill scenarios for the flash attention kernel.
    Args:
        n_heads_q: Number of query heads.
        n_heads_kv: Number of key-value heads.
        head_size: Dimension of attention head.
        paged_block_size: Size of the paged block.
        request_with_shared_prefix_ratio: Ratio of requests with shared prefix.
            if request_with_shared_prefix_ratio is 1.0, all requests have the 
            shared prefix; if request_with_shared_prefix_ratio is 0.5, only 
            half of the requests have the shared prefix, the other half are 
            regarded as the single requests, without shared prefix.
        request_count: Number of requests.
        shared_degree: Degree of sharing.
            When shared_degree is -1, it means that there's only one prefix
            sharing group. Otherwise, it means there're  `m` prefix-sharing 
            groups in this batch. 
            All requests in the same group share the same prefix. 
            `m` = `int(request_with_shared_prefix_ratio * request_count) // 
            shared_degree` 
        query_length_of_each_request: query_length for each request in this 
            batch.
        shared_prefix_length: Length of the shared prefix. 
            The lengths of shared-prefix for all prefix-sharing requests are 
            same. 
        non_shared_context_length: Length of the non-shared context. 
            The lengths of non-shared context for all prefix-sharing requests 
            are same. Note that inflight-kv (the corresponding kv from the 
            current query) is **not** involved in it.

    """
    n_heads_q, n_heads_kv = n_heads_q_n_heads_kv[0], n_heads_q_n_heads_kv[1]
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)
    if scenarios == "prefill":
        f_query_length_of_each_request = query_length_of_each_request
    else:
        f_query_length_of_each_request = 1
    # Check memory constraints, skip if setting not viable
    shared_request_count = int(request_with_shared_prefix_ratio *
                               request_count)

    f_shared_degree = shared_request_count if shared_degree == -1 \
                        else shared_degree

    mega_seq_params = create_seq_params(
        shared_request_count,
        request_count,
        f_shared_degree,
        f_query_length_of_each_request,
        shared_prefix_length,
        non_shared_context_length,
    )
    assert mega_seq_params != []

    # Prepare data
    query_lens = [
        r[0] for prefix_sharing_group in mega_seq_params
        for r in prefix_sharing_group
    ]
    kv_lens = [
        r[0] + r[1] + r[2] for prefix_sharing_group in mega_seq_params
        for r in prefix_sharing_group
    ]
    total_query_tokens = sum(query_lens)
    q_final = torch.randn(total_query_tokens,
                          n_heads_q,
                          head_size,
                          dtype=dtype)
    key_cache = torch.randn(NUM_PAGE_BLOCKS,
                            paged_block_size,
                            n_heads_kv,
                            head_size,
                            dtype=dtype)
    value_cache = torch.randn_like(key_cache).to(dtype)

    cu_seqlens_q = torch.tensor([0] + query_lens,
                                dtype=torch.int32).cumsum(dim=0,
                                                          dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0] + kv_lens,
                                dtype=torch.int32).cumsum(dim=0,
                                                          dtype=torch.int32)
    max_num_blocks_per_seq = (max(kv_lens) + paged_block_size -
                              1) // paged_block_size
    block_table = torch.randint(0, NUM_PAGE_BLOCKS, 
                                (len(query_lens), max_num_blocks_per_seq), 
                                 dtype=torch.int32)

    # rectify block_table
    cur_r = 0
    for prefix_sharing_group in mega_seq_params:
        num_elements = len(prefix_sharing_group)
        if num_elements > 1:
            # replace the block_table of the prefix-sharing group with the 
            # first one, aka the common prefix
            common_prefix_len = prefix_sharing_group[0][1]
            assert common_prefix_len > 0
            assert common_prefix_len % paged_block_size == 0
            num_common_kv_blocks = common_prefix_len // paged_block_size
            block_table[cur_r:cur_r + num_elements, :num_common_kv_blocks] = \
                block_table[cur_r, :num_common_kv_blocks]
        cur_r += num_elements

    # Prepare metadata for BatchLLM
    fixed_shared_q_lens = []
    fixed_shared_kv_lens = []
    fixed_shared_block_tables = []
    shared_q_lens = []
    shared_q_start_loc = []
    non_shared_q_lens = []
    non_shared_kv_lens = []
    non_shared_block_tables = []

    cur_r = 0
    cur_q_start_loc = 0
    for prefix_sharing_group in mega_seq_params:
        if len(prefix_sharing_group) > 1:
            fixed_shared_q_lens.append(
                sum([r[0] for r in prefix_sharing_group]))
            fixed_shared_kv_lens.append(prefix_sharing_group[0][1])
            common_prefix_len = prefix_sharing_group[0][1]
            assert common_prefix_len > 0
            assert common_prefix_len % paged_block_size == 0
            num_common_kv_blocks = common_prefix_len // paged_block_size
            fixed_shared_block_tables.append(
                block_table[cur_r][:num_common_kv_blocks].tolist())

            for i, req in enumerate(prefix_sharing_group):
                non_shared_q_lens.append(req[0])
                non_shared_kv_lens.append(req[0] + req[2])
                non_shared_block_tables.append(
                    block_table[cur_r + i][num_common_kv_blocks:].tolist())

            shared_q_lens.append(sum([r[0] for r in prefix_sharing_group]))
            shared_q_start_loc.append(cur_q_start_loc)
        else:
            non_shared_q_lens.append(prefix_sharing_group[0][0])
            non_shared_kv_lens.append(prefix_sharing_group[0][0] + 
                                      prefix_sharing_group[0][1] +
                                      prefix_sharing_group[0][2])
            non_shared_block_tables.append(block_table[cur_r].tolist())

            if len(fixed_shared_kv_lens) > 0 and fixed_shared_kv_lens[-1] == 0:
                fixed_shared_q_lens[-1] += prefix_sharing_group[0][0]
            else:
                fixed_shared_q_lens.append(prefix_sharing_group[0][0])
                fixed_shared_kv_lens.append(0)
                fixed_shared_block_tables.append([])
        cur_r += len(prefix_sharing_group)
        cur_q_start_loc += sum([r[0] for r in prefix_sharing_group])

    cu_shared_q_lens_tensor = torch.tensor([0] + fixed_shared_q_lens,
                                           dtype=torch.int32).cumsum(
                                               dim=0, dtype=torch.int32)
    cu_shared_kv_lens_tensor = torch.tensor([0] + fixed_shared_kv_lens,
                                            dtype=torch.int32).cumsum(
                                                dim=0, dtype=torch.int32)
    shared_block_tables_tensor = make_tensor_with_pad(
        fixed_shared_block_tables,
        pad=0,
        dtype=torch.int32,
        device=q_final.device)
    shared_q_lens_tensor = torch.tensor(shared_q_lens, dtype=torch.int32)
    shared_q_start_loc_tensor = torch.tensor(shared_q_start_loc,
                                             dtype=torch.int32)

    cu_non_shared_q_lens_tensor = torch.tensor([0] + non_shared_q_lens,
                                               dtype=torch.int32).cumsum(
                                                   dim=0, dtype=torch.int32)
    cu_non_shared_kv_lens_tensor = torch.tensor([0] + non_shared_kv_lens,
                                                dtype=torch.int32).cumsum(
                                                    dim=0, dtype=torch.int32)
    non_shared_kv_lens_tensor = torch.tensor(non_shared_kv_lens,
                                             dtype=torch.int32)
    non_shared_block_tables_tensor = make_tensor_with_pad(
        non_shared_block_tables,
        pad=0,
        dtype=torch.int32,
        device=q_final.device)
    max_shared_q_len = max(shared_q_lens, default=0)
    max_shared_kv_len = max(fixed_shared_kv_lens, default=0)
    max_non_shared_q_len = max(non_shared_q_lens, default=0)
    max_non_shared_kv_len = max(non_shared_kv_lens, default=0)

    # Perform flash attention call
    flash_attn_out = flash_attn_varlen_func(
        q=q_final,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(query_lens),
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_k=max(kv_lens),
        softmax_scale=1 / (head_size**0.5),
        causal=True,
        block_table=block_table,
    )

    # Perform Batchllm attention call
    batchllm_output = torch.empty_like(flash_attn_out)
    BatchLLM_attention(
        batchllm_output,
        q_final,
        key_cache,
        value_cache,

        # for shared prefix
        cu_shared_q_lens_tensor,
        cu_shared_kv_lens_tensor,
        shared_block_tables_tensor,
        shared_q_lens_tensor,
        shared_q_start_loc_tensor,

        # for non-shared context
        cu_non_shared_q_lens_tensor,
        cu_non_shared_kv_lens_tensor,
        non_shared_kv_lens_tensor,
        non_shared_block_tables_tensor,
        max_shared_q_len,
        max_shared_kv_len,
        max_non_shared_q_len,
        max_non_shared_kv_len,
        softmax_scale=1 / (head_size**0.5),
        alibi_slopes=None,
        sliding_window=(-1, -1),
        logits_soft_cap=0.0,
        is_prefill=(scenarios == "prefill"),
    )

    if dtype == torch.float16:
        torch.testing.assert_close(flash_attn_out,
                                   batchllm_output,
                                   atol=3e-2,
                                   rtol=2e-3)
    else:
        torch.testing.assert_close(flash_attn_out,
                                   batchllm_output,
                                   atol=1e-2,
                                   rtol=1e-2)