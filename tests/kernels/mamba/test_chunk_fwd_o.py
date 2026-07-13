# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for vllm's chunk_fwd_o Triton operator.

Exercises chunk_fwd_kernel_o via its Python wrapper. For each chunk of BT=64
timesteps and head, it computes:
    o = (Q @ H^T + causal(Q @ K^T) @ V) * scale
where H is a pre-computed recurrent hidden state, with optional gating via a
cumulative log-decay g. Compared against a naive float32 PyTorch reference.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.chunk_o import chunk_fwd_o
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
    pytest.skip(
        "chunk_fwd_o Triton kernel requires a CUDA-alike or XPU device",
        allow_module_level=True,
    )

DEVICE = current_platform.device_type

TOL_FP32 = (1e-2, 1e-3)  # (rtol, atol) vs the float32 reference
TOL_FP16 = (1e-2, 1e-3)
TOL_BF16 = (2e-2, 5e-3)  # bf16 has 8 mantissa bits against fp16's 11


def _chunk_fwd_o_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Naive PyTorch reference for the chunked forward output kernel.

    Both code paths share one loop: the kernel treats a fixed batch as a
    degenerate varlen one, so a `None` cu_seqlens becomes [0, T, 2T, ...].

    Args:
        q: (B, T, Hg, K) — query tensor (Hg may be < H for GQA).
        k: (B, T, Hg, K) — key tensor.
        v: (B, T, H, V) — value tensor.
        h: (NT_total, H, V, K) — hidden state per (sequence, local chunk).
        g: (B, T, H) — optional cumulative log-decay (per head).
        cu_seqlens: 1D tensor of N+1 offsets into a flattened B=1 batch;
            None means B equal-length sequences of T timesteps.

    Returns:
        o: same shape as v.
    """
    B, T, Hg, K = q.shape
    H, V = v.shape[2], v.shape[3]
    BT = FLA_CHUNK_SIZE
    T_flat = B * T
    rep = H // Hg  # GQA replication factor (1 when Hg == H)
    if scale is None:
        scale = K**-0.5

    if cu_seqlens is None:
        bounds = [(b * T, b * T + T) for b in range(B)]
    else:
        offsets = cu_seqlens.tolist()
        # The kernel leaves anything past cu_seqlens[-1] uninitialized.
        assert B == 1 and offsets[-1] == T, (
            f"varlen expects a flattened, unpadded batch: got B={B}, T={T}, "
            f"cu_seqlens[-1]={offsets[-1]}"
        )
        bounds = list(zip(offsets[:-1], offsets[1:]))

    # Fold the batch axis into time, as the kernel does via bos.
    q, k, v = (x.reshape(1, T_flat, *x.shape[2:]) for x in (q, k, v))
    if g is not None:
        g = g.reshape(1, T_flat, H)

    o = torch.zeros(1, T_flat, H, V, dtype=torch.float32, device=q.device)

    i_tg = 0
    for bos, eos in bounds:
        for t0 in range(bos, eos, BT):
            t1 = min(t0 + BT, eos)
            bt = t1 - t0
            mask = torch.tril(torch.ones(bt, bt, device=q.device))

            for hh in range(H):
                hg_idx = hh // rep

                q_c = q[0, t0:t1, hg_idx].float()  # (bt, K)
                k_c = k[0, t0:t1, hg_idx].float()  # (bt, K)
                v_c = v[0, t0:t1, hh].float()  # (bt, V)
                h_block = h[i_tg, hh].float()  # (V, K)

                inter = q_c @ h_block.t()  # (bt, V)
                attn = q_c @ k_c.t()  # (bt, bt)

                if g is not None:
                    g_c = g[0, t0:t1, hh].float()
                    inter = inter * torch.exp(g_c).unsqueeze(-1)
                    attn = attn * torch.exp(g_c[:, None] - g_c[None, :])

                attn = attn * mask
                intra = attn @ v_c

                o[0, t0:t1, hh] = inter * scale + intra * scale
            i_tg += 1

    return o.view(B, T, H, V)


def _make_inputs(
    B: int,
    T: int,
    H: int,
    Hg: int,
    K: int,
    V: int,
    dtype: torch.dtype = torch.float32,
    use_g: bool = True,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if chunk_indices is not None:
        # One h row per (sequence, local chunk); B is not part of that count.
        assert B == 1, "varlen inputs must be a flattened B=1 batch"
        NT = len(chunk_indices)
    else:
        NT = B * ((T + FLA_CHUNK_SIZE - 1) // FLA_CHUNK_SIZE)
    q = torch.randn(B, T, Hg, K, device=DEVICE, dtype=dtype) * 0.1
    k = torch.randn(B, T, Hg, K, device=DEVICE, dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device=DEVICE, dtype=dtype) * 0.1
    h = torch.randn(NT, H, V, K, device=DEVICE, dtype=dtype) * 0.1
    # g stays fp32 as in the real pipeline; the kernel's exp() requires it.
    g = (
        torch.randn(B, T, H, device=DEVICE, dtype=torch.float32) * 0.1
        if use_g
        else None
    )
    return q, k, v, h, g


# (B, T, H, Hg, K, V, use_g)
CONFIGS = [
    (1, 64, 2, 2, 64, 64, True),
    (1, 64, 2, 2, 64, 64, False),  # no gating
    (1, 128, 2, 2, 64, 64, True),  # two chunks
    (2, 64, 2, 2, 64, 64, True),  # batch > 1
    (1, 64, 4, 2, 64, 64, True),  # GQA (Hg < H)
    (1, 64, 2, 2, 128, 64, True),  # K = 128 (two blocks)
    (1, 64, 2, 2, 64, 128, True),  # V = 128 (two blocks)
    (1, 100, 2, 2, 64, 64, True),  # partial final chunk (100 = 64 + 36)
]


@pytest.mark.parametrize(("B", "T", "H", "Hg", "K", "V", "use_g"), CONFIGS)
@torch.inference_mode()
def test_chunk_fwd_o(
    B: int, T: int, H: int, Hg: int, K: int, V: int, use_g: bool
) -> None:
    """chunk_fwd_o must match the naive reference (fp32)."""
    torch.manual_seed(0)
    q, k, v, h, g = _make_inputs(B, T, H, Hg, K, V, use_g=use_g)

    o = chunk_fwd_o(q, k, v, h, g=g)
    o_ref = _chunk_fwd_o_ref(q, k, v, h, g=g)

    assert o.shape == o_ref.shape
    assert o.dtype == v.dtype
    assert not torch.any(torch.isnan(o))
    rtol, atol = TOL_FP32
    torch.testing.assert_close(o.float(), o_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("dtype", "tol"),
    [(torch.bfloat16, TOL_BF16), (torch.float16, TOL_FP16)],
)
@torch.inference_mode()
def test_chunk_fwd_o_low_precision(
    dtype: torch.dtype, tol: tuple[float, float]
) -> None:
    """bf16/fp16 inputs must match the fp32 reference (the dtype used in prod)."""
    torch.manual_seed(0)
    q, k, v, h, g = _make_inputs(1, 64, 2, 2, 64, 64, dtype, use_g=True)

    o = chunk_fwd_o(q, k, v, h, g=g)
    o_ref = _chunk_fwd_o_ref(q, k, v, h, g=g)

    assert o.dtype == dtype  # o.float() below would hide a wrong dtype
    rtol, atol = tol
    torch.testing.assert_close(o.float(), o_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("use_g", [True, False])
@torch.inference_mode()
def test_chunk_fwd_o_varlen(use_g: bool) -> None:
    """cu_seqlens (IS_VARLEN branch) must match the naive varlen reference.

    Real prefill batches always flatten multiple sequences and drive this
    kernel through cu_seqlens/chunk_indices, exercising the bos/eos/i_tg
    offset math instead of the fixed-batch path above. Parametrized over
    gating so all four USE_G x IS_VARLEN kernel variants get covered.
    """
    torch.manual_seed(0)
    cu_seqlens = torch.tensor([0, 40, 96, 250], device=DEVICE, dtype=torch.int32)
    T = int(cu_seqlens[-1])
    chunk_indices = prepare_chunk_indices(cu_seqlens, FLA_CHUNK_SIZE)
    q, k, v, h, g = _make_inputs(
        1, T, 2, 2, 64, 64, use_g=use_g, chunk_indices=chunk_indices
    )

    o = chunk_fwd_o(q, k, v, h, g=g, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    o_ref = _chunk_fwd_o_ref(q, k, v, h, g=g, cu_seqlens=cu_seqlens)

    assert o.shape == o_ref.shape
    assert o.dtype == v.dtype
    assert not torch.any(torch.isnan(o))
    rtol, atol = TOL_FP32
    torch.testing.assert_close(o.float(), o_ref, rtol=rtol, atol=atol)


@torch.inference_mode()
def test_chunk_fwd_o_core_attn_out_buffer_reuse() -> None:
    """core_attn_out must be written in-place and match a fresh allocation.

    Real GDN layers (Qwen/Kimi) pre-allocate a core_attn_out buffer and pass
    it in to avoid an extra allocation per forward pass; this exercises that
    reuse path in chunk_fwd_o instead of the default torch.empty_like(v).
    """
    torch.manual_seed(0)
    q, k, v, h, g = _make_inputs(1, 64, 2, 2, 64, 64, use_g=True)

    core_attn_out = torch.zeros_like(v)
    o = chunk_fwd_o(q, k, v, h, g=g, core_attn_out=core_attn_out)
    o_ref = _chunk_fwd_o_ref(q, k, v, h, g=g)

    assert o.data_ptr() == core_attn_out.data_ptr()
    assert not torch.any(torch.isnan(o))
    rtol, atol = TOL_FP32
    torch.testing.assert_close(o.float(), o_ref, rtol=rtol, atol=atol)


@torch.inference_mode()
def test_chunk_fwd_o_non_default_scale() -> None:
    """A non-default scale must be honoured, not silently replaced by K**-0.5."""
    torch.manual_seed(0)
    q, k, v, h, g = _make_inputs(1, 64, 2, 2, 64, 64, use_g=True)

    scale = 0.5  # K**-0.5 would be 0.125
    o = chunk_fwd_o(q, k, v, h, g=g, scale=scale)
    o_ref = _chunk_fwd_o_ref(q, k, v, h, g=g, scale=scale)

    assert not torch.any(torch.isnan(o))
    rtol, atol = TOL_FP32
    torch.testing.assert_close(o.float(), o_ref, rtol=rtol, atol=atol)
