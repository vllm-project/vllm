from typing import List, Tuple

import flashinfer
import pytest
import torch


@pytest.mark.parametrize("num_heads", [(16, 16)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("seq_lens", [[(2048, 2048)]])
@pytest.mark.parametrize("num_runs", [1000])
@pytest.mark.parametrize("beam_width", [4])
@pytest.mark.parametrize("max_num_blocks_per_seq", [512])
@torch.inference_mode()
def test_flashinfer_batchprefill_beam_search(
        num_heads: Tuple[int, int],
        head_size: int,
        dtype: torch.dtype,
        soft_cap: float,
        seq_lens: list,
        num_runs: int,
        block_size: int,
        beam_width: int,
        max_num_blocks_per_seq: int,
        key_value_cache: torch.Tensor = None
) -> Tuple[List[torch.Tensor], float]:
    torch.set_default_device("cuda")

    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0

    num_seqs = len(seq_lens)
    kv_lens = [x[1] for x in seq_lens]

    num_blocks = max_num_blocks_per_seq * num_seqs * beam_width

    if key_value_cache is None:
        key_value_cache = torch.randn(num_blocks,
                                      2,
                                      block_size,
                                      num_kv_heads,
                                      head_size,
                                      dtype=dtype,
                                      device='cuda').reshape(
                                          num_blocks, 2, block_size,
                                          num_kv_heads, head_size)

    workspace_size = 128 * 1024 * 1024
    workspace_buffer_decode = torch.empty(workspace_size,
                                          dtype=torch.int8,
                                          device='cuda')
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer_decode, "NHD")

    block_tables = torch.zeros((num_seqs * beam_width, max_num_blocks_per_seq),
                               dtype=torch.int32)
    # This will track the starting point for each sequence's shared blocks
    block_offset = 0

    for start_seq in range(num_seqs):
        shared_len = kv_lens[start_seq] // block_size

        for i in range(start_seq * beam_width, (start_seq + 1) * beam_width):
            block_tables[i, :shared_len] = torch.arange(
                block_offset, block_offset + shared_len)

        block_offset += shared_len

        for i in range(beam_width):
            beam_index = start_seq * beam_width + i
            unique_start = block_offset + i
            block_tables[beam_index,
                         shared_len:max_num_blocks_per_seq] = torch.arange(
                             unique_start, unique_start +
                             (max_num_blocks_per_seq - shared_len) *
                             beam_width, beam_width)
        block_offset += (max_num_blocks_per_seq - shared_len) * beam_width

    cumulative_run_time = 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    outputs = []

    next_block_index = [(x + block_size - 1) // block_size + 1 for x in kv_lens
                        ]  # Index of the next block from block table

    ## FORMAT KV_INDICES FOR DECODE
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []

    for i in range(num_seqs * beam_width):
        seq_len = kv_lens[i // beam_width]
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.append(list(block_tables[i, :num_blocks]))
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)
        kv_indptr.append(kv_indptr[-1] + num_blocks)

    for step in range(num_runs):
        torch.manual_seed(step)

        query = torch.arange(
            num_seqs * beam_width * num_query_heads * head_size,
            dtype=dtype,
            device='cuda').reshape(num_seqs * beam_width, num_query_heads,
                                   head_size)

        kv_indptr_tensor = torch.tensor(kv_indptr, dtype=torch.int32)
        kv_indices_tensor = torch.cat([torch.tensor(x)
                                       for x in kv_indices]).reshape(-1)
        kv_last_page_lens_tensor = torch.tensor(kv_last_page_lens,
                                                dtype=torch.int32)

        decode_wrapper.begin_forward(kv_indptr_tensor,
                                     kv_indices_tensor,
                                     kv_last_page_lens_tensor,
                                     num_query_heads,
                                     num_kv_heads,
                                     head_size,
                                     block_size,
                                     "NONE",
                                     data_type=dtype)

        start_event.record()
        output = decode_wrapper.forward(query, key_value_cache, "NONE", logits_soft_cap=soft_cap)
        end_event.record()
        torch.cuda.synchronize()
        decode_time = start_event.elapsed_time(end_event)
        cumulative_run_time += decode_time

        outputs.append(output)

        if step % block_size == 0:
            for i in range(beam_width * num_seqs):
                kv_indices[i].append(
                    block_tables[i, next_block_index[i // beam_width]])

            for i in range(len(next_block_index)):
                next_block_index[i] += 1

            for i in range(1, beam_width * num_seqs + 1):
                kv_indptr[i] += i
        kv_last_page_lens = [(x + 1) % block_size or block_size
                             for x in kv_last_page_lens]

    return outputs, cumulative_run_time


@pytest.mark.parametrize("num_heads", [(16, 16)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("seq_lens", [[(2048, 2048)]])
@pytest.mark.parametrize("num_runs", [1000])
@pytest.mark.parametrize("beam_width", [4])
@pytest.mark.parametrize("num_levels", [2])
@pytest.mark.parametrize("max_num_blocks_per_seq", [512])
@torch.inference_mode()
def test_multilevel_cascade_attention_wrapper(
        num_heads: Tuple[int, int],
        head_size: int,
        dtype: torch.dtype,
        seq_lens: list,
        num_runs: int,
        block_size: int,
        beam_width: int,
        num_levels: int,
        max_num_blocks_per_seq: int,
        soft_cap: float,
        key_value_cache: torch.Tensor = None,
) -> Tuple[List[torch.Tensor], float]:
    torch.set_default_device("cuda")

    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0

    num_seqs = len(seq_lens)
    kv_lens = [x[1] for x in seq_lens]

    num_blocks = max_num_blocks_per_seq * num_seqs * beam_width

    if key_value_cache is None:
        key_value_cache = torch.randn(num_blocks,
                                      2,
                                      block_size,
                                      num_kv_heads,
                                      head_size,
                                      dtype=dtype,
                                      device='cuda')

    workspace_size = 128 * 1024 * 1024
    workspace_buffer = torch.empty(workspace_size,
                                   dtype=torch.uint8,
                                   device='cuda')
    wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
        num_levels, workspace_buffer, "NHD")

    block_tables = torch.zeros((num_seqs * beam_width, max_num_blocks_per_seq),
                               dtype=torch.int32)

    # Tracks the starting point for each sequence's shared blocks
    block_offset = 0

    for start_seq in range(num_seqs):
        shared_len = kv_lens[start_seq] // block_size

        for i in range(start_seq * beam_width, (start_seq + 1) * beam_width):
            block_tables[i, :shared_len] = torch.arange(
                block_offset, block_offset + shared_len)

        block_offset += shared_len

        for i in range(beam_width):
            beam_index = start_seq * beam_width + i
            unique_start = block_offset + i
            block_tables[beam_index,
                         shared_len:max_num_blocks_per_seq] = torch.arange(
                             unique_start, unique_start +
                             (max_num_blocks_per_seq - shared_len) *
                             beam_width, beam_width)
        block_offset += (max_num_blocks_per_seq - shared_len) * beam_width

    qo_indptr_arr = [
        torch.tensor([0, beam_width * num_seqs],
                     dtype=torch.int32,
                     device='cuda'),
        torch.arange(beam_width * num_seqs + 1,
                     dtype=torch.int32,
                     device="cuda")
    ]

    shared_kv_page_indptr = [0]
    unique_kv_page_indptr = [0]
    shared_kv_page_indices = []
    unique_kv_page_indices: List[List[int]] = []
    shared_kv_last_page_len = []
    unique_kv_last_page_len = []

    query = torch.arange(num_seqs * beam_width * num_query_heads * head_size,
                         dtype=dtype,
                         device='cuda').reshape(num_seqs * beam_width,
                                                num_query_heads, head_size)

    ##Filling the shared metadatas
    for i in range(num_seqs):
        seq_len = kv_lens[i // beam_width]
        num_shared_blocks = (seq_len + block_size - 1) // block_size
        shared_kv_page_indices.append(list(
            block_tables[i, :num_shared_blocks]))
        shared_kv_page_indptr.append(shared_kv_page_indptr[-1] +
                                     num_shared_blocks)
        shared_kv_len = seq_len % block_size
        if shared_kv_len == 0:
            shared_kv_len = block_size
        shared_kv_last_page_len.append(shared_kv_len)

    for i in range(num_seqs * beam_width):
        num_unique_blocks = 0
        unique_kv_page_indices.append([])
        unique_kv_page_indptr.append(unique_kv_page_indptr[-1] +
                                     num_unique_blocks)
        unique_kv_last_page_len.append(block_size)

    shared_kv_page_indptr = torch.tensor(shared_kv_page_indptr,
                                         dtype=torch.int32,
                                         device='cuda')
    shared_kv_page_indices = torch.cat(
        [torch.tensor(x) for x in shared_kv_page_indices]).reshape(-1)
    shared_kv_last_page_len = torch.tensor(shared_kv_last_page_len,
                                           dtype=torch.int32,
                                           device='cuda')

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    cumulative_run_time = 0.0

    outputs = []

    next_block_index = [
        (x + block_size - 1) // block_size + 1 for x in kv_lens
    ]  # index of the next block from block table that we add

    for step in range(num_runs):
        torch.manual_seed(step)

        query = torch.arange(
            num_seqs * beam_width * num_query_heads * head_size,
            dtype=dtype,
            device='cuda').reshape(num_seqs * beam_width, num_query_heads,
                                   head_size)

        wrapper.plan(qo_indptr_arr, [
            shared_kv_page_indptr,
            torch.tensor(
                unique_kv_page_indptr, dtype=torch.int32, device='cuda')
        ], [
            shared_kv_page_indices,
            torch.cat([torch.tensor(x)
                       for x in unique_kv_page_indices]).reshape(-1)
        ], [
            shared_kv_last_page_len,
            torch.tensor(
                unique_kv_last_page_len, dtype=torch.int32, device='cuda')
        ], num_query_heads, num_kv_heads, head_size, block_size, logits_soft_cap=soft_cap)

        start_event.record()
        output = wrapper.run(query, key_value_cache)
        end_event.record()
        torch.cuda.synchronize()

        del query

        cumulative_run_time += start_event.elapsed_time(end_event)

        outputs.append(output)

        if step % block_size == 0:
            for i in range(beam_width * num_seqs):
                unique_kv_page_indices[i].append(
                    block_tables[i, next_block_index[i // beam_width]])
            for i in range(len(next_block_index)):
                next_block_index[i] += 1
            for i in range(1, beam_width * num_seqs + 1):
                unique_kv_page_indptr[i] += i

        unique_kv_last_page_len = [(x + 1) % block_size or block_size
                                   for x in unique_kv_last_page_len]

    return outputs, cumulative_run_time


def initialize_key_value_cache(num_seqs, beam_width, max_num_blocks_per_seq,
                               block_size, num_kv_heads, head_size, dtype):
    num_blocks = max_num_blocks_per_seq * num_seqs * beam_width
    key_value_cache = torch.randn(num_blocks,
                                  2,
                                  block_size,
                                  num_kv_heads,
                                  head_size,
                                  dtype=dtype,
                                  device='cuda')
    return key_value_cache


@pytest.mark.parametrize("test_case", [
    {
        "beam_width": 4,
        "seq_lens": [(4096, 4096)]
    },
    {
        "beam_width": 8,
        "seq_lens": [(4096, 4096)]
    },
    {
        "beam_width": 16,
        "seq_lens": [(4096, 4096)]
    },
    {
        "beam_width": 32,
        "seq_lens": [(4096, 4096)]
    },
])
def test_cascade_speedup(test_case):
    """
    Compares the performance of flashinfer multilevel kernel and batch prefill.
    """
    common_params = {
        "num_heads": (16, 16),
        "head_size": 128,
        "dtype": torch.float16,
        "block_size": 16,
        "num_runs": 1000,
        "max_num_blocks_per_seq": 2560,
        "soft_cap": None
    }

    num_seqs = len(test_case["seq_lens"])
    _, num_kv_heads = common_params["num_heads"]

    key_value_cache = initialize_key_value_cache(
        num_seqs, test_case["beam_width"],
        common_params["max_num_blocks_per_seq"], common_params["block_size"],
        num_kv_heads, common_params["head_size"], common_params["dtype"])

    cascade_outputs, time_taken_cascade = test_multilevel_cascade_attention_wrapper(  # noqa: E501
        **common_params,
        seq_lens=test_case["seq_lens"],
        beam_width=test_case["beam_width"],
        num_levels=2,
        key_value_cache=key_value_cache)

    cascade_outputs_cpu = [output.cpu() for output in cascade_outputs]
    del cascade_outputs
    torch.cuda.empty_cache()

    batchprefill_outputs, time_taken_batchprefill = test_flashinfer_batchprefill_beam_search(  # noqa: E501
        **common_params,
        seq_lens=test_case["seq_lens"],
        beam_width=test_case["beam_width"],
        key_value_cache=key_value_cache)

    batchprefill_outputs_cpu = [
        output.cpu() for output in batchprefill_outputs
    ]
    del batchprefill_outputs
    torch.cuda.empty_cache()

    assert len(cascade_outputs_cpu) == len(
        batchprefill_outputs_cpu
    ), "Output length mismatch between the two methods."

    max_diff = 0
    total_diff = 0
    total_elements = 0

    for cascade_output, batchprefill_output in zip(cascade_outputs_cpu,
                                                   batchprefill_outputs_cpu):
        assert cascade_output.shape == batchprefill_output.shape, "Shape mismatch between outputs."  # noqa: E501

        diff = torch.abs(cascade_output - batchprefill_output)
        max_diff = max(max_diff, torch.max(diff).item())
        total_diff += torch.sum(diff).item()
        total_elements += cascade_output.numel()

    avg_diff = total_diff / total_elements

    speedup = time_taken_batchprefill / time_taken_cascade
    print("MAX DIFF", max_diff)

    # assert speedup > 1.0, f"No speedup with cascade infer: {speedup}"
    # assert max_diff < 1e-2, f"Max difference too large: {max_diff}"
    # assert avg_diff < 1e-3, f"Average difference too large: {avg_diff}"