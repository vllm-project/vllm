# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.kimi_k3.amd.ops.third_party.kda.chunk_intra import (
    chunk_kda_fwd_intra,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AMD KDA requires ROCm",
)


@pytest.mark.parametrize("safe_gate", [False, True])
@pytest.mark.parametrize(
    "cu_seqlens",
    [
        pytest.param(None, id="fixed"),
        pytest.param([0, 15, 100, 300], id="varlen"),
    ],
)
@torch.inference_mode()
def test_chunk_kda_intra_zeros_upper_triangle(
    safe_gate: bool,
    cu_seqlens: list[int] | None,
) -> None:
    torch.manual_seed(42)
    chunk_size = 64
    num_heads = 8
    head_dim = 128
    total_tokens = 300

    q = torch.randn(
        1,
        total_tokens,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.rand(
        1,
        total_tokens,
        num_heads,
        device="cuda",
        dtype=torch.bfloat16,
    )

    cu_seqlens_t = None
    chunk_indices = None
    sequence_bounds = [(0, total_tokens)]
    if cu_seqlens is not None:
        cu_seqlens_t = torch.tensor(cu_seqlens, device="cuda", dtype=torch.int64)
        chunk_indices = prepare_chunk_indices(cu_seqlens_t, chunk_size)
        sequence_bounds = list(zip(cu_seqlens[:-1], cu_seqlens[1:]))

    _, A = chunk_kda_fwd_intra(
        q=q,
        k=k,
        gk=g,
        beta=beta,
        scale=head_dim**-0.5,
        cu_seqlens=cu_seqlens_t,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
    )

    columns = torch.arange(chunk_size, device="cuda")
    for start, end in sequence_bounds:
        rows = torch.arange(end - start, device="cuda") % chunk_size
        upper_triangle = columns[None, :] > rows[:, None]
        upper_values = A[0, start:end].masked_select(upper_triangle[:, None, :])
        assert torch.count_nonzero(upper_values) == 0
