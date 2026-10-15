# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static (CUDA graph) grids of the GDN prefill kernels.

A static grid covers the largest batch: padded tokens, zero-length padded
sequences and ``[-1, 0]`` padded chunks for FLA, ``PAD_SLOT_ID`` programs for
causal_conv1d. On the real rows it must give the same bits as the batch-sized
grid.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.gdn_attn import CAUSAL_CONV1D_BLOCK_M
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="GDN Triton kernels need a GPU"
)

DEVICE = "cuda"
CHUNK_SIZE = 64
# Static capacity beyond the real batch.
NUM_PAD_SEQS = 5
NUM_PAD_TOKENS = 100


def _seq_lens(num_seqs: int, generator: torch.Generator) -> list[int]:
    # Chunk-boundary lengths first, then random lengths in [1, 2048].
    edges = [1, 2048, 63, 64, 65, 129]
    random = torch.randint(1, 2049, (num_seqs,), generator=generator).tolist()
    return (edges + random)[:num_seqs]


def _cu_seqlens(seq_lens: list[int], num_padded_seqs: int = 0) -> torch.Tensor:
    lens = torch.tensor(seq_lens + [0] * num_padded_seqs, dtype=torch.int32)
    return F.pad(lens.cumsum(0, dtype=torch.int32), (1, 0)).to(DEVICE)


def _pad_chunk_indices(chunk_indices: torch.Tensor, num_rows: int) -> torch.Tensor:
    pad = chunk_indices.new_tensor((-1, 0)).expand(num_rows - len(chunk_indices), 2)
    return torch.cat([chunk_indices, pad])


def _conv_metadata(nums_dict, batch_ptr, token_chunk_offset_ptr) -> SimpleNamespace:
    return SimpleNamespace(
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )


@pytest.mark.parametrize("with_initial_state", [True, False])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_seqs", [1, 7, 64])
@torch.inference_mode()
def test_chunk_gated_delta_rule_static_grid_is_bitwise_equal(
    num_seqs: int, state_dtype: torch.dtype, with_initial_state: bool
):
    rng_cpu = torch.Generator().manual_seed(num_seqs)
    rng = torch.Generator(DEVICE).manual_seed(num_seqs)
    seq_lens = _seq_lens(num_seqs, rng_cpu)
    num_tokens = sum(seq_lens)
    num_padded_tokens = num_tokens + NUM_PAD_TOKENS
    max_num_seqs = num_seqs + NUM_PAD_SEQS
    num_k_heads, num_v_heads, head_dim = 2, 4, 128

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device=DEVICE, dtype=dtype, generator=rng)

    q = F.normalize(randn(1, num_padded_tokens, num_k_heads, head_dim), dim=-1)
    k = F.normalize(randn(1, num_padded_tokens, num_k_heads, head_dim), dim=-1)
    v = randn(1, num_padded_tokens, num_v_heads, head_dim)
    g = F.logsigmoid(randn(1, num_padded_tokens, num_v_heads, dtype=torch.float32))
    beta = randn(1, num_padded_tokens, num_v_heads, dtype=torch.float32).sigmoid()
    h0 = randn(max_num_seqs, num_v_heads, head_dim, head_dim, dtype=state_dtype)

    cu_seqlens = _cu_seqlens(seq_lens)
    o, final_state = chunk_gated_delta_rule(
        q=q[:, :num_tokens],
        k=k[:, :num_tokens],
        v=v[:, :num_tokens],
        g=g[:, :num_tokens],
        beta=beta[:, :num_tokens],
        initial_state=h0[:num_seqs] if with_initial_state else None,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=prepare_chunk_indices(cu_seqlens, CHUNK_SIZE),
        chunk_offsets=prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE),
    )

    static_cu_seqlens = _cu_seqlens(seq_lens, NUM_PAD_SEQS)
    max_num_chunks = cdiv(num_padded_tokens, CHUNK_SIZE) + max_num_seqs - 1
    static_o, static_final_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0 if with_initial_state else None,
        output_final_state=True,
        cu_seqlens=static_cu_seqlens,
        chunk_indices=_pad_chunk_indices(
            prepare_chunk_indices(cu_seqlens, CHUNK_SIZE), max_num_chunks
        ),
        chunk_offsets=prepare_chunk_offsets(static_cu_seqlens, CHUNK_SIZE),
        use_graph=True,
    )

    assert torch.equal(static_o[:, :num_tokens], o)
    assert torch.equal(static_final_state[:num_seqs], final_state)
    # A zero-length sequence returns its initial state.
    expected_pad_state = (
        h0[num_seqs:].float()
        if with_initial_state
        else torch.zeros_like(static_final_state[num_seqs:])
    )
    assert torch.equal(static_final_state[num_seqs:], expected_pad_state)


@pytest.mark.parametrize("num_seqs", [1, 7, 64])
@torch.inference_mode()
def test_causal_conv1d_static_grid_is_bitwise_equal(num_seqs: int):
    rng_cpu = torch.Generator().manual_seed(num_seqs)
    rng = torch.Generator(DEVICE).manual_seed(num_seqs)
    seq_lens = _seq_lens(num_seqs, rng_cpu)
    num_tokens = sum(seq_lens)
    num_padded_tokens = num_tokens + NUM_PAD_TOKENS
    dim, width, num_cache_lines = 256, 4, 2 * num_seqs + 1
    dtype = torch.bfloat16

    # Channel-last x of the padded token buffer, as in the GDN layer.
    x = torch.randn(num_padded_tokens, dim, device=DEVICE, dtype=dtype, generator=rng)
    weight = torch.randn(dim, width, device=DEVICE, dtype=dtype, generator=rng)
    bias = torch.randn(dim, device=DEVICE, dtype=dtype, generator=rng)
    conv_states = torch.randn(
        num_cache_lines, width - 1, dim, device=DEVICE, dtype=dtype, generator=rng
    ).transpose(1, 2)
    cache_indices = (
        torch.randperm(num_cache_lines - 1, generator=rng_cpu)[:num_seqs] + 1
    ).to(DEVICE, torch.int32)
    has_initial_state = (torch.arange(num_seqs, device=DEVICE) % 2).bool()
    query_start_loc = _cu_seqlens(seq_lens)
    query_start_loc_cpu = query_start_loc.cpu()

    def run(metadata, num_programs=None):
        states = conv_states.clone()
        out = causal_conv1d_fn(
            x.T,
            weight,
            bias,
            conv_states=states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation="silu",
            metadata=metadata,
            num_programs=num_programs,
        )
        return out, states

    out, states = run(
        _conv_metadata(
            *compute_causal_conv1d_metadata(query_start_loc_cpu, device=x.device)
        )
    )

    max_num_programs = (
        cdiv(num_padded_tokens, CAUSAL_CONV1D_BLOCK_M) + num_seqs + NUM_PAD_SEQS - 1
    )
    program_buffers = (
        torch.empty(max_num_programs, device=DEVICE, dtype=torch.int32),
        torch.empty(max_num_programs, device=DEVICE, dtype=torch.int32),
    )
    static_metadata = compute_causal_conv1d_metadata(
        query_start_loc_cpu, device=x.device, program_buffers=program_buffers
    )
    assert static_metadata[1] is program_buffers[0]
    static_out, static_states = run(_conv_metadata(*static_metadata), max_num_programs)

    assert torch.equal(static_out[:, :num_tokens], out[:, :num_tokens])
    assert torch.equal(static_states, states)
