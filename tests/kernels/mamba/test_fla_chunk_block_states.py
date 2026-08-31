# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical-parity test for the per-chunk intermediate states (`h`) newly
exposed by chunk_gated_delta_rule(..., return_intermediate_states=True).

Semantics under test (verified from chunk_delta_h.py: h is stored at the START of
each chunk loop iteration, before the chunk update):
    h[:, c] == recurrent state AFTER the first c chunks (i.e. before chunk c),
              == replay(first c*FLA_CHUNK_SIZE tokens).final_state
    h[:, 0] == initial_state
    final_state == state after all tokens.

This pins the index mapping gdn_scatter_block_checkpoints relies on:
    checkpoint after m*block_size tokens == h[:, m*(block_size//FLA_CHUNK_SIZE)].
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

if not current_platform.is_cuda():
    pytest.skip(reason="FLA chunk kernels require CUDA", allow_module_level=True)

DEVICE = "cuda"
DT = torch.bfloat16
ATOL, RTOL = 3e-2, 3e-2


def _mk(B, T, H, K, V, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    q = torch.randn(B, T, H, K, dtype=DT, device=DEVICE, generator=gen)
    k = F.normalize(
        torch.randn(B, T, H, K, dtype=DT, device=DEVICE, generator=gen), p=2, dim=-1
    )
    v = torch.randn(B, T, H, V, dtype=DT, device=DEVICE, generator=gen)
    beta = torch.rand(B, T, H, dtype=DT, device=DEVICE, generator=gen).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=DT, device=DEVICE, generator=gen))
    return q, k, v, g, beta


def _close(a, b):
    a = a.float()
    b = b.float()
    return (a - b).abs().max().item(), torch.allclose(a, b, atol=ATOL, rtol=RTOL)


def test_block_states_equal_len():
    """Single sequence, equal-length path (cu_seqlens=None)."""
    B, H, K, V = 1, 4, 64, 64
    C = FLA_CHUNK_SIZE
    L = 4 * C  # 4 chunks
    NT = L // C
    q, k, v, g, beta = _mk(B, L, H, K, V, seed=1)
    h0 = torch.randn(1, H, V, K, dtype=torch.float32, device=DEVICE)

    o, final_state, h = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=h0,
        output_final_state=True,
        return_intermediate_states=True,
    )
    assert h is not None and h.shape == (B, NT, H, V, K), h.shape

    # h[:,0] == initial_state
    md0, ok0 = _close(h[:, 0], h0)
    assert ok0, f"h[0] != initial_state, maxdiff={md0}"

    # h[:,c] == replay(first c chunks).final_state  for c in 1..NT-1
    for c in range(1, NT):
        _, fs_c = chunk_gated_delta_rule(
            q[:, : c * C],
            k[:, : c * C],
            v[:, : c * C],
            g[:, : c * C],
            beta[:, : c * C],
            initial_state=h0,
            output_final_state=True,
        )
        md, ok = _close(h[:, c], fs_c)
        assert ok, f"h[{c}] mismatch vs replay({c} chunks), maxdiff={md}"
        # discriminating: h[c] must NOT match the adjacent (wrong) checkpoint
        md_off, ok_off = _close(h[:, c - 1], fs_c)
        assert not ok_off or md_off < md, (
            f"off-by-one not discriminated at c={c}: md_off={md_off} md={md}"
        )
    # final_state == replay(all)
    _, fs_all = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=h0,
        output_final_state=True,
    )
    _, okf = _close(final_state, fs_all)
    assert okf
    print("equal_len OK")


def test_block_states_varlen():
    """Varlen path (B=1, cu_seqlens) — the real GDN prefill path."""
    H, K, V = 4, 64, 64
    C = FLA_CHUNK_SIZE
    lens = [3 * C, 5 * C]  # 2 seqs, 3 and 5 chunks
    T = sum(lens)
    cu = torch.tensor([0, lens[0], T], dtype=torch.int32, device=DEVICE)
    ci = prepare_chunk_indices(cu, C)
    co = prepare_chunk_offsets(cu, C)
    q, k, v, g, beta = _mk(1, T, H, K, V, seed=2)
    h0 = torch.randn(len(lens), H, V, K, dtype=torch.float32, device=DEVICE)

    o, final_state, h = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu,
        chunk_indices=ci,
        chunk_offsets=co,
        return_intermediate_states=True,
    )
    # per-sequence: h[0, chunk_offsets[n] + c] == replay(seq n, first c chunks)
    for n, ln in enumerate(lens):
        s = cu[n].item()
        e = cu[n + 1].item()
        base = co[n].item()
        nch = (e - s) // C
        # h[:,0] of the sequence == its initial state
        md0, ok0 = _close(h[:, base], h0[n : n + 1])
        assert ok0, f"seq{n} h[base] != init, md={md0}"
        for c in range(1, nch):
            qs = q[:, s : s + c * C]
            ks = k[:, s : s + c * C]
            vs = v[:, s : s + c * C]
            gs = g[:, s : s + c * C]
            bs = beta[:, s : s + c * C]
            cu_c = torch.tensor([0, c * C], dtype=torch.int32, device=DEVICE)
            _, fs_c = chunk_gated_delta_rule(
                qs,
                ks,
                vs,
                gs,
                bs,
                initial_state=h0[n : n + 1],
                output_final_state=True,
                cu_seqlens=cu_c,
                chunk_indices=prepare_chunk_indices(cu_c, C),
                chunk_offsets=prepare_chunk_offsets(cu_c, C),
            )
            md, ok = _close(h[:, base + c], fs_c)
            assert ok, f"seq{n} h[{base + c}] mismatch, maxdiff={md}"
    print("varlen OK")


if __name__ == "__main__":
    test_block_states_equal_len()
    test_block_states_varlen()
    print("ALL PASS")
