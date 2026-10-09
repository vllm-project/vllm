# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch decode build of the DeepSeek V4.1 SWA metadata on ROCm.

DeepseekV4ROCMAiterSparseSWAMetadataBuilder.build runs once per SWA KV cache
group, which is 8 times a step for DeepSeek-V4.1-Flash at TP 4. For a decode
batch that this file does not take, each build launches 8 small kernels:
torch.ge for is_valid_token, the zero fill of the unused decode_swa_lens,
_compute_swa_indices_and_lens_kernel, the ragged row count, torch.cumsum, the
ragged compaction and 2 copies into the CUDA graph buffers. On MI325X each
took about 4.2 us, so at 128k context the 8 builds held the GPU for 281 us of
every step before the decode graph could start.

_decode_swa_meta_kernel writes the same tensors with one program: the dense
window indices and lengths, is_valid_token, the zero lengths after the batch,
and the ragged indices and indptr directly into the graph buffers. It does not
write the ragged entries past indptr[-1]. _copy_ragged_to_graph_buffers copies
them from an uninitialized tensor, and the attention kernels read only up to
indptr.

The DSpark draft's 3 layers build their non-causal block the same way, with
ComputeDSparkNoncausalSWAIndicesKernel in place of the window kernel. That
is 3 more builds of 8 launches after every target step.

On gfx942 with VLLM_ROCM_MONO_DECODE=1 (``dsv41_gfx942.enabled()``), decode
batches of up to MAX_ROWS rows, causal or the draft block, use it. Other
batches take the rest of DeepseekV4ROCMAiterSparseSWAMetadataBuilder.build.
tests/kernels/test_dsv41_gfx942_decode_meta.py compares it with the vLLM ops
that it replaces.
"""

import torch

from vllm.model_executor.layers.dsv41_gfx942 import enabled
from vllm.triton_utils import tl, triton

# One program holds all rows of the batch in registers, as a ROWS x WIDTH
# tile of int32. 32 rows of a 128 wide window are 16 values per thread.
MAX_ROWS = 32
TAIL_BLOCK = 1024


@triton.jit(do_not_specialize=["num_tokens", "lens_size"])
def _decode_swa_meta_kernel(
    slot_mapping_ptr,
    is_valid_ptr,
    indices_ptr,
    indices_stride,
    lens_ptr,
    lens_size,
    ragged_ptr,
    indptr_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_ptr,
    block_table_ptr,
    block_table_stride,
    replay_start_ptr,
    num_tokens,
    window_size,
    block_size,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    TAIL: tl.constexpr,
    NONCAUSAL: tl.constexpr,
):
    rows = tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    in_batch = rows < num_tokens
    # A token is valid when it has a KV slot. Padded tokens have slot -1.
    # The request index is loaded at the same time, and only the loads below
    # that use it wait for the slot.
    slot = tl.load(slot_mapping_ptr + rows, mask=in_batch, other=-1)
    req = tl.load(token_to_req_ptr + rows, mask=in_batch, other=0)
    valid = in_batch & (slot >= 0)
    tl.store(is_valid_ptr + rows, valid, mask=in_batch)

    q_start = tl.load(query_start_loc_ptr + req, mask=valid, other=0)
    q_end = tl.load(query_start_loc_ptr + req + 1, mask=valid, other=0)
    seq_len = tl.load(seq_lens_ptr + req, mask=valid, other=0)
    prefix_len = seq_len - (q_end - q_start)
    if NONCAUSAL:
        # The DSpark draft block, as ComputeDSparkNoncausalSWAIndicesKernel
        # builds it: every token of the block sees the window_size positions
        # before the block and the whole block, including later tokens.
        start = tl.maximum(prefix_len - window_size, 0)
        swa_len = tl.where(valid, seq_len - start, 0)
    else:
        # The same window as _compute_swa_indices_and_lens_kernel without
        # images: the last window_size positions up to the token, but not
        # below the request's replay start.
        pos = prefix_len + rows - q_start
        start = tl.maximum(pos - (window_size - 1), 0)
        start = tl.maximum(start, tl.load(replay_start_ptr + req, mask=valid, other=0))
        swa_len = tl.where(valid, pos + 1 - start, 0)
    tl.store(lens_ptr + rows, swa_len, mask=in_batch)

    p = start[:, None] + cols[None, :]
    inside = (cols[None, :] < swa_len[:, None]) & (cols[None, :] < WIDTH)
    # inside is false in every row that is not valid, so the request index of
    # a padded token is never used for an address.
    block = tl.load(
        block_table_ptr + req[:, None] * block_table_stride + p // block_size,
        mask=inside,
        other=0,
    )
    slots = tl.where(inside, block * block_size + p % block_size, -1)
    tl.store(
        indices_ptr + rows[:, None] * indices_stride + cols[None, :],
        slots,
        mask=in_batch[:, None] & (cols[None, :] < WIDTH),
    )

    # The ragged form keeps, in order, the entries of each row that are inside
    # its length and not negative, as build_ragged_indices_from_dense does.
    # indptr[r + 1] is the number of entries kept in rows 0 to r.
    keep = inside & (slots >= 0)
    counts = tl.sum(keep.to(tl.int32), axis=1)
    ends = tl.cumsum(counts, axis=0)
    tl.store(indptr_ptr, 0)
    tl.store(indptr_ptr + 1 + rows, ends, mask=in_batch)
    rank = tl.cumsum(keep.to(tl.int32), axis=1) - 1
    tl.store(ragged_ptr + (ends - counts)[:, None] + rank, slots, mask=keep)

    # Set the lengths of all rows after the batch to 0, as
    # DeepseekSparseSWAMetadataBuilder.build does.
    for i in range(num_tokens, lens_size, TAIL):
        offs = i + tl.arange(0, TAIL)
        tl.store(lens_ptr + offs, 0, mask=offs < lens_size)


def launch(
    slot_mapping,
    is_valid,
    indices,
    lens,
    ragged,
    indptr,
    query_start_loc,
    seq_lens,
    token_to_req,
    block_table,
    replay_start,
    num_tokens,
    window_size,
    block_size,
    noncausal=False,
):
    """Write is_valid[:num_tokens], indices[:num_tokens], all of lens, and
    the ragged form into ragged and indptr, in one launch. noncausal builds
    the DSpark draft block's rows, whose width is the index tensor's."""
    width = indices.shape[-1]
    _decode_swa_meta_kernel[(1,)](
        slot_mapping,
        is_valid,
        indices,
        indices.stride(0),
        lens,
        lens.shape[0],
        ragged,
        indptr,
        query_start_loc,
        seq_lens,
        token_to_req,
        block_table,
        block_table.stride(0),
        replay_start,
        num_tokens,
        window_size,
        block_size,
        WIDTH=width,
        BLOCK=triton.next_power_of_2(width),
        ROWS=triton.next_power_of_2(num_tokens),
        TAIL=TAIL_BLOCK,
        NONCAUSAL=noncausal,
        num_warps=4,
    )


def build(builder, common_prefix_len, cam, fast_build, replay_start, metadata_cls):
    """Return the ROCm SWA metadata of a decode batch, causal or the DSpark
    draft block, built with one launch, or None when the batch needs the rest
    of DeepseekV4ROCMAiterSparseSWAMetadataBuilder.build."""
    if not enabled():
        return None
    from vllm.v1.attention.backends.mla import sparse_swa
    from vllm.v1.attention.backends.utils import split_decodes_and_prefills

    noncausal = not cam.causal
    if noncausal and not builder.is_dspark:
        return None
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        split_decodes_and_prefills(cam, decode_threshold=builder.decode_threshold)
    )
    slot_mapping = cam.slot_mapping
    if (
        num_prefill_tokens > 0
        or not 0 < num_decode_tokens <= MAX_ROWS
        or slot_mapping.shape[0] != num_decode_tokens
    ):
        return None
    if noncausal:
        # The DSpark draft block's rows have their own wider buffer, which
        # DeepseekSparseSWAMetadataBuilder.build creates the same way on its
        # first non-causal batch.
        width = builder.noncausal_index_width
        if builder.decode_swa_indices_noncausal is None:
            builder.decode_swa_indices_noncausal = torch.zeros(
                builder._max_tokens,
                1,
                width,
                dtype=torch.int32,
                device=builder.device,
            )
        indices = builder.decode_swa_indices_noncausal
    else:
        width = builder.window_size
        indices = builder.decode_swa_indices
    if indices.shape[-1] != width:
        return None

    token_to_req = cam.token_to_req_indices(builder.token_to_req_indices)
    is_valid_token = builder.is_valid_token[:num_decode_tokens]
    if replay_start is None:
        replay_start = builder.no_replay_start
    launch(
        slot_mapping,
        is_valid_token,
        indices,
        builder.decode_swa_lens,
        builder.decode_swa_ragged_indices_buffer,
        builder.decode_swa_ragged_indptr_buffer,
        cam.query_start_loc,
        cam.seq_lens,
        token_to_req,
        cam.block_table_tensor,
        replay_start,
        num_decode_tokens,
        builder.window_size,
        builder.block_size,
        noncausal=noncausal,
    )

    # These are the fields DeepseekSparseSWAMetadataBuilder.build returns for
    # a batch without prefill tokens, and the ragged views
    # _copy_ragged_to_graph_buffers returns.
    tile_sched = builder.build_tile_scheduler(num_decode_tokens)
    base = sparse_swa.DeepseekSparseSWAMetadata(
        seq_lens=cam.seq_lens,
        query_start_loc=cam.query_start_loc,
        query_start_loc_cpu=cam.query_start_loc_cpu,
        block_table=cam.block_table_tensor,
        slot_mapping=slot_mapping,
        is_valid_token=is_valid_token,
        token_to_req_indices=token_to_req,
        decode_swa_indices=indices[:num_decode_tokens],
        decode_swa_lens=builder.decode_swa_lens[:num_decode_tokens],
        decode_swa_width=width,
        prefill_swa_indices=None,
        prefill_swa_lens=None,
        prefill_left_visible=None,
        prefill_right_visible=None,
        replay_start=replay_start,
        block_size=builder.block_size,
        num_decodes=num_decodes,
        num_prefills=num_prefills,
        num_decode_tokens=num_decode_tokens,
        num_prefill_tokens=num_prefill_tokens,
        max_decode_query_len=min(cam.max_query_len, builder.decode_threshold),
        tile_sched_swaonly=tile_sched[sparse_swa._LAYER_TYPE_SWAONLY],
        tile_sched_c4a=tile_sched[sparse_swa._LAYER_TYPE_C4A],
        tile_sched_c128a=tile_sched[sparse_swa._LAYER_TYPE_C128A],
        tile_sched_c1a=tile_sched[sparse_swa._LAYER_TYPE_C1A],
        tile_sched_c2a=tile_sched[sparse_swa._LAYER_TYPE_C2A],
        **builder._build_deepseek_v4_metadata(
            num_decodes,
            num_prefills,
            cam.seq_lens,
            cam.seq_lens_cpu_upper_bound,
            cam.query_start_loc,
            cam.query_start_loc_cpu,
            replay_start,
        ),
    )
    out = metadata_cls(
        **vars(base),
        decode_swa_ragged_indices=builder.decode_swa_ragged_indices_buffer[
            : max(num_decode_tokens * width, 1)
        ],
        decode_swa_ragged_indptr=builder.decode_swa_ragged_indptr_buffer[
            : num_decode_tokens + 1
        ],
        prefill_swa_ragged_indices=None,
        prefill_swa_ragged_indptr=None,
    )
    return out
