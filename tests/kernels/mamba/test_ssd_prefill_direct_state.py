# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.model_executor.layers.mamba.ops.ssd_prefill_dispatch import (
    FlashInferMamba2PrefillRequest,
    run_flashinfer_mamba2_prefill,
)
from vllm.v1.attention.backends.mamba2_attn import (
    compute_flashinfer_ssd_metadata,
)


def _triton_chunk_metadata(
    contexts: list[int], query_lengths: list[int], chunk_size: int
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Build vLLM's sequence-relative Triton chunk traversal."""
    cu_chunk_seqlens = [0]
    last_chunk_indices = []
    chunk_seq_ids = []
    absolute_endpoints = []
    packed_position = 0

    for seq_id, (context, query_length) in enumerate(
        zip(contexts, query_lengths)
    ):
        remaining = query_length
        absolute_position = context
        while remaining:
            capacity = chunk_size - absolute_position % chunk_size
            take = min(remaining, capacity)
            remaining -= take
            packed_position += take
            absolute_position += take
            cu_chunk_seqlens.append(packed_position)
            chunk_seq_ids.append(seq_id)
            absolute_endpoints.append(absolute_position)
        last_chunk_indices.append(len(chunk_seq_ids) - 1)

    return (
        cu_chunk_seqlens,
        last_chunk_indices,
        chunk_seq_ids,
        absolute_endpoints,
    )


def _make_strided_state_cache(
    num_slots: int,
    nheads: int,
    headdim: int,
    dstate: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_stride = 589824
    storage_offset = 18432
    compact_row_size = nheads * headdim * dstate
    storage_size = (
        storage_offset
        + (num_slots - 1) * row_stride
        + compact_row_size
        + 4096
    )
    storage = torch.full(
        (storage_size,), -6.5, dtype=torch.float16, device="cuda"
    )
    cache = torch.as_strided(
        storage,
        (num_slots, nheads, headdim, dstate),
        (row_stride, headdim * dstate, dstate, 1),
        storage_offset=storage_offset,
    )
    cache.fill_(1.25)
    return cache, storage


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_initial_states", [False, True])
def test_full_flashinfer_direct_state_mixed_phases_and_checkpoints(
    use_initial_states: bool,
):
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3), (11, 0)}:
        pytest.skip("FlashInfer Mamba2 SSD prefill requires SM100/SM103/SM110")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(1234)
    chunk_size = 128
    mamba_block_size = 1152
    nheads, headdim, dstate, ngroups = 64, 64, 128, 8
    contexts = [0, 1, 87, 127, 127]
    query_lengths = [2433, 127, 1200, 1026, 1025]
    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
    )
    contexts_tensor = torch.tensor(contexts, dtype=torch.int32)

    metadata = compute_flashinfer_ssd_metadata(
        query_start_loc,
        contexts_tensor,
        chunk_size=chunk_size,
        mamba_block_size=mamba_block_size,
        require_triton_state_boundaries=False,
        repack_sequence_chunks=True,
    )
    (
        cu_chunk_seqlens,
        last_chunk_indices,
        chunk_seq_ids,
        absolute_endpoints,
    ) = _triton_chunk_metadata(contexts, query_lengths, chunk_size)
    assert len(metadata.chunk_indices) == len(chunk_seq_ids)
    assert metadata.segment_seq_ids == chunk_seq_ids
    assert [context % chunk_size for context in contexts[:4]] == [0, 1, 87, 127]

    # Deliberately shuffled physical rows prove that FI consumes vLLM's
    # two-dimensional block table mapping rather than treating block columns
    # or logical-segment ordinals as cache rows.
    block_rows = [
        [13, 2, 9],
        [7, 15, 16],
        [11, 1, 14],
        [6, 12, 17],
        [4, 10, 8],
    ]
    direct_indices = [
        block_rows[seq_id][block_idx] if block_idx >= 0 else -1
        for seq_id, block_idx in zip(
            metadata.segment_seq_ids,
            metadata.segment_state_block_indices,
        )
    ]
    selected_segments = [
        segment for segment, row in enumerate(direct_indices) if row >= 0
    ]
    selected_rows = [direct_indices[segment] for segment in selected_segments]
    assert len(selected_rows) == 9
    assert len(set(selected_rows)) == len(selected_rows)
    assert selected_rows != sorted(selected_rows)
    assert [absolute_endpoints[index] for index in selected_segments] == [
        1152,
        2304,
        2433,
        128,
        1152,
        1287,
        1152,
        1153,
        1152,
    ]

    def randn(*shape: int, dtype: torch.dtype, scale: float = 1.0):
        return (
            torch.randn(*shape, device=device, generator=generator)
            .mul_(scale)
            .to(dtype)
        )

    tokens = int(query_start_loc[-1])
    x = randn(tokens, nheads, headdim, dtype=torch.bfloat16, scale=0.1)
    dt = randn(tokens, nheads, dtype=torch.bfloat16, scale=0.1)
    A = -torch.rand(nheads, device=device, generator=generator)
    B = randn(tokens, ngroups, dstate, dtype=torch.bfloat16, scale=0.1)
    C = randn(tokens, ngroups, dstate, dtype=torch.bfloat16, scale=0.1)
    D = randn(nheads, dtype=torch.bfloat16, scale=0.1)
    dt_bias = randn(nheads, dtype=torch.float32, scale=0.1)
    initial_states = None
    if use_initial_states:
        initial_states = torch.zeros(
            len(contexts),
            nheads,
            headdim,
            dstate,
            dtype=torch.float16,
            device=device,
        )
        initial_states[0] = randn(
            nheads, headdim, dstate, dtype=torch.float16, scale=0.01
        )
        initial_states[2] = randn(
            nheads, headdim, dstate, dtype=torch.float16, scale=0.01
        )

    to_int32_cuda = lambda values: torch.tensor(  # noqa: E731
        values, dtype=torch.int32, device=device
    )
    token_dst_indices = torch.tensor(
        metadata.token_dst_indices, dtype=torch.int64, device=device
    )
    fi_out = torch.empty_like(x)
    state_cache, state_storage = _make_strided_state_cache(
        18, nheads, headdim, dstate
    )
    cache_before = state_cache.clone()
    storage_before = state_storage.clone()

    reason = run_flashinfer_mamba2_prefill(
        FlashInferMamba2PrefillRequest(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            out=fi_out,
            state_cache=state_cache,
            initial_states=initial_states,
            seq_idx=to_int32_cuda(metadata.seq_idx).unsqueeze(0),
            token_dst_indices=token_dst_indices,
            chunk_indices=to_int32_cuda(metadata.chunk_indices),
            chunk_offsets=to_int32_cuda(metadata.chunk_offsets),
            seq_chunk_cumsum=to_int32_cuda(metadata.seq_chunk_cumsum),
            intermediate_state_indices=to_int32_cuda(direct_indices),
            chunk_size=chunk_size,
            num_seqs=len(contexts),
            valid_seqlen=metadata.valid_seqlen,
            padded_seqlen=metadata.padded_seqlen,
            requires_repacking=True,
        )
    )
    assert reason is None

    cu_seqlens = query_start_loc.to(device)
    cu_chunk_seqlens_t = to_int32_cuda(cu_chunk_seqlens)
    last_chunk_indices_t = to_int32_cuda(last_chunk_indices)
    chunk_seq_ids_t = to_int32_cuda(chunk_seq_ids)
    triton_out = torch.empty_like(x)
    triton_states = mamba_chunk_scan_combined_varlen(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        cu_seqlens,
        cu_chunk_seqlens_t,
        last_chunk_indices_t,
        chunk_seq_ids_t,
        triton_out,
        D=D,
        dt_bias=dt_bias,
        initial_states=initial_states,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        return_intermediate_states=True,
        state_dtype=torch.float16,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(fi_out, triton_out, atol=8e-2, rtol=7e-2)
    output_mismatch_fraction = (
        (fi_out != triton_out).count_nonzero().item() / fi_out.numel()
    )
    assert output_mismatch_fraction < 2e-3

    selected_segments_t = torch.tensor(
        selected_segments, dtype=torch.int64, device=device
    )
    direct_states = torch.stack([state_cache[row] for row in selected_rows])
    reference_states = triton_states.index_select(0, selected_segments_t)
    state_abs_diff = (direct_states - reference_states).abs()
    print(
        "direct_state_max_abs=",
        state_abs_diff.max().item(),
        "direct_state_mismatches=",
        (direct_states != reference_states).count_nonzero().item(),
        "output_mismatch_fraction=",
        output_mismatch_fraction,
    )
    torch.testing.assert_close(
        direct_states, reference_states, atol=2e-3, rtol=2e-3
    )

    selected_row_set = set(selected_rows)
    for row in range(state_cache.shape[0]):
        if row not in selected_row_set:
            assert torch.equal(state_cache[row], cache_before[row])

    row_stride = state_cache.stride(0)
    compact_row_size = nheads * headdim * dstate
    storage_offset = state_cache.storage_offset()
    assert torch.equal(
        state_storage[:storage_offset], storage_before[:storage_offset]
    )
    for row in range(state_cache.shape[0]):
        gap_start = storage_offset + row * row_stride + compact_row_size
        gap_end = (
            storage_offset + (row + 1) * row_stride
            if row + 1 < state_cache.shape[0]
            else state_storage.numel()
        )
        assert torch.equal(
            state_storage[gap_start:gap_end],
            storage_before[gap_start:gap_end],
        )

    # Consume every FI-written checkpoint/final state in a subsequent Triton
    # continuation. This catches semantically harmful state drift even when a
    # direct FP16 cache comparison is close but not bitwise equal.
    continuation_contexts = [
        absolute_endpoints[index] for index in selected_segments
    ]
    continuation_lengths = [129] * len(selected_segments)
    continuation_starts = torch.tensor(
        [0, *torch.tensor(continuation_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    (
        continuation_cu_chunks,
        continuation_last_chunks,
        continuation_seq_ids,
        _,
    ) = _triton_chunk_metadata(
        continuation_contexts, continuation_lengths, chunk_size
    )
    continuation_tokens = int(continuation_starts[-1])
    continuation_x = randn(
        continuation_tokens,
        nheads,
        headdim,
        dtype=torch.bfloat16,
        scale=0.1,
    )
    continuation_dt = randn(
        continuation_tokens, nheads, dtype=torch.bfloat16, scale=0.1
    )
    continuation_B = randn(
        continuation_tokens,
        ngroups,
        dstate,
        dtype=torch.bfloat16,
        scale=0.1,
    )
    continuation_C = randn(
        continuation_tokens,
        ngroups,
        dstate,
        dtype=torch.bfloat16,
        scale=0.1,
    )
    continuation_out_fi_state = torch.empty_like(continuation_x)
    continuation_out_reference_state = torch.empty_like(continuation_x)

    def run_continuation(initial: torch.Tensor, out: torch.Tensor):
        return mamba_chunk_scan_combined_varlen(
            continuation_x,
            continuation_dt,
            A,
            continuation_B,
            continuation_C,
            chunk_size,
            continuation_starts,
            to_int32_cuda(continuation_cu_chunks),
            to_int32_cuda(continuation_last_chunks),
            to_int32_cuda(continuation_seq_ids),
            out,
            D=D,
            dt_bias=dt_bias,
            initial_states=initial,
            dt_softplus=True,
            dt_limit=(0.0, float("inf")),
            return_intermediate_states=True,
            state_dtype=torch.float16,
        )

    continuation_states_fi = run_continuation(
        direct_states, continuation_out_fi_state
    )
    continuation_states_reference = run_continuation(
        reference_states, continuation_out_reference_state
    )
    torch.cuda.synchronize()
    continuation_mismatch_fraction = (
        (
            continuation_out_fi_state
            != continuation_out_reference_state
        ).count_nonzero().item()
        / continuation_out_fi_state.numel()
    )
    continuation_max_abs = (
        continuation_out_fi_state - continuation_out_reference_state
    ).abs().max().item()
    print(
        "continuation_output_mismatch_fraction=",
        continuation_mismatch_fraction,
        "continuation_output_max_abs=",
        continuation_max_abs,
    )
    torch.testing.assert_close(
        continuation_out_fi_state,
        continuation_out_reference_state,
        atol=8e-3,
        rtol=2e-3,
    )
    assert continuation_mismatch_fraction < 2e-4
    torch.testing.assert_close(
        continuation_states_fi,
        continuation_states_reference,
        atol=2e-3,
        rtol=2e-3,
    )
