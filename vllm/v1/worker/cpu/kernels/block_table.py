# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numba ports of the staged write, KV zeroing and DCP Triton kernels."""

from vllm.v1.worker.cpu.kernels.utils import int32_array, numba_kernel


@numba_kernel
def apply_write(
    grid,
    output_ptr,
    output_stride,
    write_indices_ptr,
    write_starts_ptr,
    write_contents_ptr,
    write_cu_lens_ptr,
    MULTI_GROUP,
):
    # CPUBlockTables applies each group separately, so only the single-group
    # form is used.
    if MULTI_GROUP:
        raise NotImplementedError("multi-group staged writes")
    output = output_ptr.reshape(-1)
    for i in range(grid[0]):
        content_start = write_cu_lens_ptr[i - 1] if i > 0 else 0
        content = write_contents_ptr[content_start : write_cu_lens_ptr[i]]
        start = write_indices_ptr[i] * output_stride + write_starts_ptr[i]
        output[start : start + len(content)] = content


@numba_kernel
def zero_kv_blocks(
    grid,
    seg_addrs_ptr,
    seg_block_strides_ptr,
    seg_page_sizes_ptr,
    block_ids_ptr,
):
    # Strides and page sizes are in int32 elements.
    for block_id in block_ids_ptr:
        for seg in range(len(seg_addrs_ptr)):
            start = block_id * seg_block_strides_ptr[seg]
            end = start + seg_page_sizes_ptr[seg]
            int32_array(seg_addrs_ptr[seg], (end,))[start:] = 0


@numba_kernel
def dcp_local_seq_lens(
    grid,
    out_ptr,
    seq_lens_ptr,
    dcp_size,
    dcp_rank,
    cp_interleave,
    num_reqs,
    max_num_reqs,
):
    out_ptr[num_reqs:max_num_reqs] = 0
    for i in range(num_reqs):
        # Distribute KV cache among different ranks, in a round-robin manner.
        seq_len = seq_lens_ptr[i]
        rounds = seq_len // (dcp_size * cp_interleave)
        remainder = seq_len % (dcp_size * cp_interleave) - dcp_rank * cp_interleave
        out_ptr[i] = rounds * cp_interleave + min(max(remainder, 0), cp_interleave)
