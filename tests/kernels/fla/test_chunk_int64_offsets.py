# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FLA chunk kernels on batches whose offsets do not fit in int32."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="needs a GPU"
)

BT = 64


def _skip_if_low_memory(need: int) -> None:
    free, _ = current_platform.mem_get_info()
    if free < need + (1 << 30):
        pytest.skip("not enough free device memory")


@torch.inference_mode()
def test_chunk_delta_h_int64_seq_offset():
    """A later sequence's start offsets need int64.

    The second sequence starts at token 2^31 // (H * V), so its v, w and v_new
    offsets (bos * H * V) and its h offset (chunk * H * V * K) pass 2^31.
    """
    H, Hg, K, V = 64, 16, 128, 128
    L1, L2 = 2**31 // (H * V), 2 * BT
    T = L1 + L2
    # k, w, u, v_new and h, all bf16.
    _skip_if_low_memory(T * (Hg * K + 3 * H * V) * 2 + T // BT * H * V * K * 2)

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(0)

    def randn(*shape, scale=1.0):
        x = torch.randn(*shape, device=device, generator=gen, dtype=torch.bfloat16)
        return x.mul_(scale)

    k = randn(1, T, Hg, K, scale=0.09)
    w = randn(1, T, H, K, scale=0.05)
    u = randn(1, T, H, V)
    # Chunk-local cumulative log decay, as chunk_local_cumsum produces it.
    g = torch.rand(1, T // BT, BT, H, device=device, generator=gen) * -0.05
    g = g.cumsum(2).view(1, T, H)
    h0 = torch.randn(2, H, V, K, device=device, generator=gen) * 0.1
    cu_seqlens = torch.tensor([0, L1, T], dtype=torch.int32, device=device)

    h, v_new, ht = chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    # Reference: the second sequence on its own, where its offsets are 0.
    h_ref, v_new_ref, ht_ref = chunk_gated_delta_rule_fwd_h(
        k[:, L1:].contiguous(),
        w[:, L1:].contiguous(),
        u[:, L1:].contiguous(),
        g=g[:, L1:].contiguous(),
        initial_state=h0[1:].contiguous(),
        output_final_state=True,
        cu_seqlens=torch.tensor([0, L2], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(h[:, L1 // BT :], h_ref, atol=0, rtol=0)
    torch.testing.assert_close(v_new[:, L1:], v_new_ref, atol=0, rtol=0)
    torch.testing.assert_close(ht[1:], ht_ref, atol=0, rtol=0)
