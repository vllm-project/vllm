# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.model_executor.layers.mamba.ops.ssd_prefill_dispatch import (
    FlashInferMamba2PrefillRequest,
    get_mamba2_prefill_dispatch_stats,
    reset_mamba2_prefill_dispatch_stats,
    run_flashinfer_mamba2_prefill,
)
from vllm.v1.attention.backends.mamba2_attn import (
    compute_flashinfer_ssd_metadata,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flashinfer_repacked_mixed_phases_matches_triton_output_and_direct_state():
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3), (11, 0)}:
        pytest.skip("FlashInfer Mamba2 SSD prefill requires SM100/SM103/SM110")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    chunk_size = 128
    nheads, headdim, dstate, ngroups = 64, 64, 128, 8
    query_start_loc = torch.tensor([0, 257, 471, 640], dtype=torch.int32)
    num_computed_tokens = torch.tensor([128, 128, 128], dtype=torch.int32)

    metadata = compute_flashinfer_ssd_metadata(
        query_start_loc,
        num_computed_tokens,
        chunk_size=chunk_size,
        mamba_block_size=1152,
        require_triton_state_boundaries=False,
        repack_sequence_chunks=True,
    )
    assert metadata.requires_repacking
    assert metadata.valid_seqlen == metadata.padded_seqlen == 896
    assert [metadata.token_dst_indices[i] for i in (0, 257, 471)] == [
        0,
        384,
        640,
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
    initial_states = randn(
        3, nheads, headdim, dstate, dtype=torch.float16, scale=0.01
    )

    to_int32_cuda = lambda values: torch.tensor(  # noqa: E731
        values, dtype=torch.int32, device=device
    )
    token_dst_indices = torch.tensor(
        metadata.token_dst_indices, dtype=torch.int64, device=device
    )
    fi_out = torch.empty_like(x)
    fi_state_cache = torch.full(
        (4, nheads, headdim, dstate),
        1.25,
        dtype=torch.float16,
        device=device,
    )
    fi_state_cache_before = fi_state_cache.clone()
    intermediate_state_indices = to_int32_cuda(
        [-1, -1, 0, -1, 1, -1, 2]
    )
    request = FlashInferMamba2PrefillRequest(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        out=fi_out,
        state_cache=fi_state_cache,
        initial_states=initial_states,
        seq_idx=to_int32_cuda(metadata.seq_idx).unsqueeze(0),
        token_dst_indices=token_dst_indices,
        chunk_indices=to_int32_cuda(metadata.chunk_indices),
        chunk_offsets=to_int32_cuda(metadata.chunk_offsets),
        seq_chunk_cumsum=to_int32_cuda(metadata.seq_chunk_cumsum),
        intermediate_state_indices=intermediate_state_indices,
        chunk_size=chunk_size,
        num_seqs=3,
        valid_seqlen=metadata.valid_seqlen,
        padded_seqlen=metadata.padded_seqlen,
        requires_repacking=True,
    )

    reset_mamba2_prefill_dispatch_stats()
    assert run_flashinfer_mamba2_prefill(request) is None
    torch.cuda.synchronize()
    stats = get_mamba2_prefill_dispatch_stats()
    assert stats.flashinfer_layer_invocations == 1
    assert stats.flashinfer_layer_tokens == tokens
    assert not torch.equal(fi_state_cache[:3], fi_state_cache_before[:3])
    assert torch.equal(fi_state_cache[3], fi_state_cache_before[3])

    cu_seqlens = query_start_loc.to(device)
    cu_chunk_seqlens = to_int32_cuda([0, 128, 256, 257, 385, 471, 599, 640])
    last_chunk_indices = to_int32_cuda([2, 4, 6])
    chunk_seq_idx = to_int32_cuda([0, 0, 0, 1, 1, 2, 2])
    triton_out = torch.empty_like(x)
    triton_states = mamba_chunk_scan_combined_varlen(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        cu_seqlens,
        cu_chunk_seqlens,
        last_chunk_indices,
        chunk_seq_idx,
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
    mismatch_fraction = (fi_out != triton_out).count_nonzero().item() / fi_out.numel()
    assert mismatch_fraction < 2e-3
    torch.testing.assert_close(
        fi_state_cache[:3],
        triton_states[last_chunk_indices],
        atol=2e-3,
        rtol=2e-3,
    )
