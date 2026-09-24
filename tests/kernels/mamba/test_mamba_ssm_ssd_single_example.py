# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.mamba._mamba_ssm_ssd import (
    DEVICE,
    generate_random_inputs,
    ssd_minimal_discrete,
)
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mamba2_attn import compute_varlen_chunk_metadata

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Mamba2 SSD Triton kernels require a CUDA-alike or XPU device.",
)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n_heads", [4, 16, 32])
@pytest.mark.parametrize("d_head", [5, 8, 32, 128])
@pytest.mark.parametrize("seq_len_chunk_size", [(112, 16), (128, 32)])
def test_mamba_chunk_scan_single_example(d_head, n_heads, seq_len_chunk_size, itype):
    # this tests the kernels on a single example (bs=1)

    # TODO: the bfloat16 case requires higher thresholds. To be investigated

    if itype == torch.bfloat16:
        atol, rtol = 5e-2, 5e-2
    else:
        atol, rtol = 8e-3, 5e-3

    # set seed
    batch_size = 1  # batch_size
    # ssd_minimal_discrete requires chunk_size divide seqlen
    # - this is only required for generating the reference seqs,
    #   it is not an operational limitation.
    seqlen, chunk_size = seq_len_chunk_size

    A, dt, X, B, C = generate_random_inputs(batch_size, seqlen, n_heads, d_head, itype)

    Y_min, final_state_min = ssd_minimal_discrete(
        X * dt.unsqueeze(-1), A * dt, B, C, chunk_size
    )

    cu_seqlens = torch.tensor((0, seqlen), device=DEVICE).cumsum(dim=0)
    cu_chunk_seqlens, last_chunk_indices, seq_idx_chunks = (
        compute_varlen_chunk_metadata(cu_seqlens, chunk_size)
    )
    # varlen has implicit batch=1
    X = X.squeeze(0)
    dt = dt.squeeze(0)
    A = A.squeeze(0)
    B = B.squeeze(0)
    C = C.squeeze(0)
    Y = torch.empty_like(X)
    final_state = mamba_chunk_scan_combined_varlen(
        X,
        dt,
        A,
        B,
        C,
        chunk_size,
        cu_seqlens=cu_seqlens.to(torch.int32),
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=last_chunk_indices,
        seq_idx=seq_idx_chunks,
        out=Y,
        D=None,
    )

    # just test the last in sequence
    torch.testing.assert_close(Y[-1], Y_min[0, -1], atol=atol, rtol=rtol)

    # just test the last head
    # NOTE, in the kernel we always cast states to fp32
    torch.testing.assert_close(
        final_state[:, -1].to(torch.float32),
        final_state_min[:, -1].to(torch.float32),
        atol=atol,
        rtol=rtol,
    )
