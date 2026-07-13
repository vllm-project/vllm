# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for vllm's chunk_fwd_o Triton operator.

Exercises chunk_fwd_kernel_o via its Python wrapper. For each chunk of BT=64
timesteps and head, it computes:
    o = (Q @ H^T + causal(Q @ K^T) @ V) * scale
where H is a pre-computed recurrent hidden state, with optional gating via a
cumulative log-decay g. Compared against a naive float32 PyTorch reference.

Source: vllm/model_executor/layers/fla/ops/chunk_o.py
"""

import pytest
import torch

from vllm.model_executor.layers.fla.ops.chunk_o import chunk_fwd_o
from vllm.platforms import current_platform

# chunk_fwd_o dispatches a Triton kernel that requires a
# GPU-class backend.
if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
    pytest.skip(
        "chunk_fwd_o Triton kernel requires a CUDA-alike "
        "or XPU device",
        allow_module_level=True,
    )

DEVICE = current_platform.device_type


def chunk_fwd_o_ref(q, k, v, h, g=None, scale=None, chunk_size=64):
    """Naive PyTorch reference for the chunked forward output kernel.

    Args:
        q: (B, T, Hg, K) — query tensor (Hg may be < H for GQA).
        k: (B, T, Hg, K) — key tensor.
        v: (B, T, H, V) — value tensor.
        h: (B*NT, H, V, K) — pre-computed hidden state at each chunk boundary.
        g: (B, T, H) — optional cumulative log-decay (per head).

    Returns:
        o: (B, T, H, V).
    """
    B, T, Hg, K = q.shape
    H, V = v.shape[2], v.shape[3]
    BT = chunk_size
    NT = (T + BT - 1) // BT
    rep = H // Hg  # GQA replication factor (1 when Hg == H)
    if scale is None:
        scale = K**-0.5

    o = torch.zeros(B, T, H, V, dtype=torch.float32, device=q.device)

    for b in range(B):
        for ci in range(NT):
            t0 = ci * BT
            t1 = min(t0 + BT, T)
            bt = t1 - t0
            mask = torch.tril(torch.ones(bt, bt, device=q.device))

            for hh in range(H):
                hg_idx = hh // rep

                q_c = q[b, t0:t1, hg_idx].float()  # (bt, K)
                k_c = k[b, t0:t1, hg_idx].float()  # (bt, K)
                v_c = v[b, t0:t1, hh].float()  # (bt, V)

                h_idx = b * NT + ci
                h_block = h[h_idx, hh].float()  # (V, K)

                inter = q_c @ h_block.t()  # (bt, V)
                attn = q_c @ k_c.t()  # (bt, bt)

                if g is not None:
                    g_c = g[b, t0:t1, hh].float()
                    inter = inter * torch.exp(g_c).unsqueeze(-1)
                    attn = attn * torch.exp(g_c[:, None] - g_c[None, :])

                attn = attn * mask
                intra = attn @ v_c

                o[b, t0:t1, hh] = inter * scale + intra * scale

    return o


def chunk_fwd_o_ref_varlen(q, k, v, h, cu_seqlens, g=None, scale=None, chunk_size=64):
    """Naive PyTorch reference for the varlen (cu_seqlens) code path.

    Mirrors chunk_fwd_o_ref, but chunks each sequence in cu_seqlens locally
    and maps chunks to rows of h via a global counter, matching how
    prepare_chunk_indices enumerates (sequence, local_chunk) pairs.

    Args:
        q, k, v, g: same layout as chunk_fwd_o_ref but with B=1 and T equal
            to the total flattened length (cu_seqlens[-1]).
        h: (sum_i ceil(len_i / chunk_size), H, V, K).
        cu_seqlens: 1D sequence of N+1 offsets for N sequences.
    """
    _, T, Hg, K = q.shape
    H, V = v.shape[2], v.shape[3]
    BT = chunk_size
    rep = H // Hg
    if scale is None:
        scale = K**-0.5

    o = torch.zeros(1, T, H, V, dtype=torch.float32, device=q.device)

    gidx = 0
    for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        seq_len = eos - bos
        nt = (seq_len + BT - 1) // BT
        for ci in range(nt):
            t0 = bos + ci * BT
            t1 = min(t0 + BT, eos)
            bt = t1 - t0
            mask = torch.tril(torch.ones(bt, bt, device=q.device))

            for hh in range(H):
                hg_idx = hh // rep

                q_c = q[0, t0:t1, hg_idx].float()
                k_c = k[0, t0:t1, hg_idx].float()
                v_c = v[0, t0:t1, hh].float()
                h_block = h[gidx, hh].float()

                inter = q_c @ h_block.t()
                attn = q_c @ k_c.t()

                if g is not None:
                    g_c = g[0, t0:t1, hh].float()
                    inter = inter * torch.exp(g_c).unsqueeze(-1)
                    attn = attn * torch.exp(g_c[:, None] - g_c[None, :])

                attn = attn * mask
                intra = attn @ v_c

                o[0, t0:t1, hh] = inter * scale + intra * scale
            gidx += 1

    return o


def _make_inputs(
    B, T, H, Hg, K, V, dtype=torch.float32, use_g=True, chunk_size=64
):
    NT = (T + chunk_size - 1) // chunk_size
    q = torch.randn(B, T, Hg, K, device=DEVICE, dtype=dtype) * 0.1
    k = torch.randn(B, T, Hg, K, device=DEVICE, dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device=DEVICE, dtype=dtype) * 0.1
    h = torch.randn(B * NT, H, V, K, device=DEVICE, dtype=dtype) * 0.1
    # g (cumulative log-decay) is always fp32 in the real pipeline
    # (chunk_local_cumsum defaults output_dtype=torch.float); the kernel's
    # exp() requires fp32/fp64 regardless of q/k/v/h dtype.
    g = (
        torch.randn(B, T, H, device=DEVICE, dtype=torch.float32) * 0.1
        if use_g
        else None
    )
    scale = K**-0.5
    return q, k, v, h, g, scale


# (B, T, H, Hg, K, V, use_g) — T is a multiple of 64 except the dedicated
# partial-tail-chunk config below.
CONFIGS = [
    (1, 64, 2, 2, 64, 64, True),
    (1, 64, 2, 2, 64, 64, False),  # no gating
    (1, 128, 2, 2, 64, 64, True),  # two chunks
    (2, 64, 2, 2, 64, 64, True),  # batch > 1
    (1, 64, 4, 2, 64, 64, True),  # GQA (Hg < H)
    (1, 64, 2, 2, 128, 64, True),  # K = 128 (two blocks)
    (1, 64, 2, 2, 64, 128, True),  # V = 128 (two blocks)
    (1, 192, 2, 2, 64, 64, True),  # three chunks
    (1, 100, 2, 2, 64, 64, True),  # partial final chunk (100 = 64 + 36)
]


@pytest.mark.parametrize(
    "B,T,H,Hg,K,V,use_g",
    CONFIGS,
    ids=[
        f"B{b}_T{t}_H{h}_Hg{hg}_K{k}_V{v}_g{int(g)}"
        for b, t, h, hg, k, v, g in CONFIGS
    ],
)
@torch.inference_mode()
def test_chunk_fwd_o(B, T, H, Hg, K, V, use_g):
    """chunk_fwd_o must match the naive reference (fp32)."""
    torch.manual_seed(0)
    q, k, v, h, g, scale = _make_inputs(B, T, H, Hg, K, V, use_g=use_g)

    o = chunk_fwd_o(q, k, v, h, g=g, scale=scale)
    o_ref = chunk_fwd_o_ref(q, k, v, h, g=g, scale=scale)

    assert o.shape == o_ref.shape
    assert o.dtype == v.dtype
    assert not torch.any(torch.isnan(o))
    torch.testing.assert_close(o.float(), o_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_chunk_fwd_o_low_precision(dtype):
    """bf16/fp16 with scalar gating must match the fp32 reference."""
    torch.manual_seed(0)
    B, T, H, Hg, K, V = 1, 64, 2, 2, 64, 64
    q, k, v, h, g, scale = _make_inputs(B, T, H, Hg, K, V, dtype, use_g=True)

    o = chunk_fwd_o(q, k, v, h, g=g, scale=scale)
    o_ref = chunk_fwd_o_ref(q, k, v, h, g=g, scale=scale)

    torch.testing.assert_close(o.float(), o_ref, atol=5e-2, rtol=5e-2)


@torch.inference_mode()
def test_chunk_fwd_o_varlen():
    """cu_seqlens (IS_VARLEN branch) must match the naive varlen reference.

    Real prefill batches always flatten multiple sequences and drive this
    kernel through cu_seqlens/chunk_indices, exercising the bos/eos/i_tg
    offset math instead of the fixed-batch path above.
    """
    from vllm.model_executor.layers.fla.ops.index import prepare_chunk_indices

    torch.manual_seed(0)
    H, Hg, K, V = 2, 2, 64, 64
    cu_seqlens = torch.tensor([0, 40, 96, 250], device=DEVICE, dtype=torch.int32)
    T = int(cu_seqlens[-1])
    NT = sum((int(e - b) + 63) // 64 for b, e in zip(cu_seqlens[:-1], cu_seqlens[1:]))
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64)

    q = torch.randn(1, T, Hg, K, device=DEVICE) * 0.1
    k = torch.randn(1, T, Hg, K, device=DEVICE) * 0.1
    v = torch.randn(1, T, H, V, device=DEVICE) * 0.1
    h = torch.randn(NT, H, V, K, device=DEVICE) * 0.1
    g = torch.randn(1, T, H, device=DEVICE) * 0.1
    scale = K**-0.5

    o = chunk_fwd_o(
        q, k, v, h, g=g, scale=scale,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    o_ref = chunk_fwd_o_ref_varlen(q, k, v, h, cu_seqlens, g=g, scale=scale)

    assert not torch.any(torch.isnan(o))
    torch.testing.assert_close(o.float(), o_ref, atol=1e-2, rtol=1e-2)


@torch.inference_mode()
def test_chunk_fwd_o_core_attn_out_buffer_reuse():
    """core_attn_out must be written in-place and match a fresh allocation.

    Real GDN layers (Qwen/Kimi) pre-allocate a core_attn_out buffer and pass
    it in to avoid an extra allocation per forward pass; this exercises that
    reuse path in chunk_fwd_o instead of the default torch.empty_like(v).
    """
    torch.manual_seed(0)
    B, T, H, Hg, K, V = 1, 64, 2, 2, 64, 64
    q, k, v, h, g, scale = _make_inputs(B, T, H, Hg, K, V, use_g=True)

    core_attn_out = torch.zeros_like(v)
    o = chunk_fwd_o(q, k, v, h, g=g, scale=scale, core_attn_out=core_attn_out)
    o_ref = chunk_fwd_o_ref(q, k, v, h, g=g, scale=scale)

    assert o.data_ptr() == core_attn_out.data_ptr()
    torch.testing.assert_close(o.float(), o_ref, atol=1e-2, rtol=1e-2)
