# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the individual Mamba-2 SSD chunk Triton kernels.

Upstream ``test_mamba_ssm_ssd.py`` exercises the high-level
``mamba_chunk_scan_combined_varlen`` pipeline. These tests target the
component kernels in isolation, each against a pure-PyTorch reference:

* ``_chunk_cumsum_fwd``     (ssd_chunk_state.py) -- _chunk_cumsum_fwd_kernel
* ``_chunk_state_fwd``      (ssd_chunk_state.py) -- _chunk_state_fwd_kernel
* ``_bmm_chunk_fwd``        (ssd_bmm.py)         -- _bmm_chunk_fwd_kernel
* ``_chunk_scan_fwd``       (ssd_chunk_scan.py)  -- _chunk_scan_fwd_kernel
* ``_state_passing_fwd``    (ssd_state_passing.py) -- _state_passing_fwd_kernel

All kernels are pure Triton, so they run on any backend the vLLM platform
layer treats as a CUDA-alike device or as XPU.
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.ssd_bmm import _bmm_chunk_fwd
from vllm.model_executor.layers.mamba.ops.ssd_chunk_scan import _chunk_scan_fwd
from vllm.model_executor.layers.mamba.ops.ssd_chunk_state import (
    _chunk_cumsum_fwd,
    _chunk_state_fwd,
)
from vllm.model_executor.layers.mamba.ops.ssd_state_passing import _state_passing_fwd
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Mamba2 SSD Triton kernels require a CUDA-alike or XPU device.",
)


# ============================================================================
# _chunk_cumsum_fwd  (_chunk_cumsum_fwd_kernel)
# ============================================================================


def _softplus(x):
    """Numerically stable softplus: log(1 + exp(x)), with large-x shortcut."""
    return torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)


def _chunk_cumsum_fwd_ref(
    dt,
    A,
    chunk_size,
    cu_chunk_seqlens,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
):
    """Pure PyTorch reference for _chunk_cumsum_fwd.

    dt: (seqlen, nheads); A: (nheads,); cu_chunk_seqlens: (nchunks+1,) int32.
    Returns dA_cumsum, dt_out, both (nheads, nchunks, chunk_size) float32.
    """
    seqlen, nheads = dt.shape
    nchunks = cu_chunk_seqlens.shape[0] - 1
    dt_out = torch.zeros(
        nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    dA_cumsum = torch.zeros(
        nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )

    for c in range(nchunks):
        chunk_start = cu_chunk_seqlens[c].item()
        chunk_end = cu_chunk_seqlens[c + 1].item()
        chunk_len = chunk_end - chunk_start

        for h in range(nheads):
            for m in range(min(chunk_len, chunk_size)):
                val = dt[chunk_start + m, h].float()
                if dt_bias is not None:
                    val = val + dt_bias[h].float()
                if dt_softplus:
                    val = _softplus(val.unsqueeze(0)).squeeze(0)
                val = val.clamp(dt_limit[0], dt_limit[1])
                dt_out[h, c, m] = val

            for m in range(min(chunk_len, chunk_size)):
                dA_val = dt_out[h, c, m] * A[h].float()
                if m == 0:
                    dA_cumsum[h, c, m] = dA_val
                else:
                    dA_cumsum[h, c, m] = dA_cumsum[h, c, m - 1] + dA_val

    return dA_cumsum, dt_out


def _cumsum_make_inputs(seqlen, nheads, chunk_size, itype, has_dt_bias=False):
    nchunks = seqlen // chunk_size
    dt = torch.randn(seqlen, nheads, device=DEVICE, dtype=itype) * 0.5
    A = -torch.rand(nheads, device=DEVICE, dtype=itype) * 0.1  # A is typically negative
    cu_chunk_seqlens = torch.tensor(
        [i * chunk_size for i in range(nchunks + 1)],
        device=DEVICE,
        dtype=torch.int32,
    )
    dt_bias = None
    if has_dt_bias:
        dt_bias = torch.randn(nheads, device=DEVICE, dtype=itype) * 0.1
    return dt, A, cu_chunk_seqlens, dt_bias


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("nheads", [4, 8])
@pytest.mark.parametrize("chunk_size", [32, 64])
@torch.inference_mode()
def test_chunk_cumsum_fwd_basic(chunk_size, nheads, itype):
    """Single chunk, no bias, no softplus."""
    set_random_seed(0)
    seqlen = chunk_size
    dt, A, cu_seqlens, _ = _cumsum_make_inputs(seqlen, nheads, chunk_size, itype)

    dA_cumsum, dt_out = _chunk_cumsum_fwd(dt, A, chunk_size, cu_seqlens)
    dA_ref, dt_ref = _chunk_cumsum_fwd_ref(dt, A, chunk_size, cu_seqlens)

    torch.testing.assert_close(dt_out, dt_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dA_cumsum, dA_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("nheads", [4, 8])
@torch.inference_mode()
def test_chunk_cumsum_fwd_with_bias(nheads, itype):
    """With dt_bias."""
    set_random_seed(0)
    seqlen, chunk_size = 64, 64
    dt, A, cu_seqlens, dt_bias = _cumsum_make_inputs(
        seqlen, nheads, chunk_size, itype, has_dt_bias=True
    )

    dA_cumsum, dt_out = _chunk_cumsum_fwd(
        dt, A, chunk_size, cu_seqlens, dt_bias=dt_bias
    )
    dA_ref, dt_ref = _chunk_cumsum_fwd_ref(
        dt, A, chunk_size, cu_seqlens, dt_bias=dt_bias
    )

    torch.testing.assert_close(dt_out, dt_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dA_cumsum, dA_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("nheads", [4, 8])
@torch.inference_mode()
def test_chunk_cumsum_fwd_softplus(nheads, itype):
    """With softplus activation."""
    set_random_seed(0)
    seqlen, chunk_size = 64, 64
    dt, A, cu_seqlens, _ = _cumsum_make_inputs(seqlen, nheads, chunk_size, itype)

    dA_cumsum, dt_out = _chunk_cumsum_fwd(
        dt, A, chunk_size, cu_seqlens, dt_softplus=True
    )
    dA_ref, dt_ref = _chunk_cumsum_fwd_ref(
        dt, A, chunk_size, cu_seqlens, dt_softplus=True
    )

    torch.testing.assert_close(dt_out, dt_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dA_cumsum, dA_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_chunk_cumsum_fwd_multi_chunk(itype):
    """Multiple chunks."""
    set_random_seed(0)
    seqlen, nheads, chunk_size = 256, 4, 64
    dt, A, cu_seqlens, _ = _cumsum_make_inputs(seqlen, nheads, chunk_size, itype)

    dA_cumsum, dt_out = _chunk_cumsum_fwd(dt, A, chunk_size, cu_seqlens)
    dA_ref, dt_ref = _chunk_cumsum_fwd_ref(dt, A, chunk_size, cu_seqlens)

    torch.testing.assert_close(dt_out, dt_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dA_cumsum, dA_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_chunk_cumsum_fwd_bias_softplus_limit(itype):
    """With bias + softplus + dt_limit clamping."""
    set_random_seed(0)
    seqlen, nheads, chunk_size = 128, 8, 64
    dt_limit = (0.001, 10.0)
    dt, A, cu_seqlens, dt_bias = _cumsum_make_inputs(
        seqlen, nheads, chunk_size, itype, has_dt_bias=True
    )

    dA_cumsum, dt_out = _chunk_cumsum_fwd(
        dt,
        A,
        chunk_size,
        cu_seqlens,
        dt_bias=dt_bias,
        dt_softplus=True,
        dt_limit=dt_limit,
    )
    dA_ref, dt_ref = _chunk_cumsum_fwd_ref(
        dt,
        A,
        chunk_size,
        cu_seqlens,
        dt_bias=dt_bias,
        dt_softplus=True,
        dt_limit=dt_limit,
    )

    torch.testing.assert_close(dt_out, dt_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dA_cumsum, dA_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("nheads", [4, 8, 16])
@torch.inference_mode()
def test_chunk_cumsum_fwd_nheads_sweep(nheads, itype):
    """Sweep nheads to exercise different BLOCK_SIZE_H autotuning."""
    set_random_seed(0)
    seqlen, chunk_size = 128, 64
    dt, A, cu_seqlens, dt_bias = _cumsum_make_inputs(
        seqlen, nheads, chunk_size, itype, has_dt_bias=True
    )

    dA_cumsum, dt_out = _chunk_cumsum_fwd(
        dt, A, chunk_size, cu_seqlens, dt_bias=dt_bias
    )
    dA_ref, dt_ref = _chunk_cumsum_fwd_ref(
        dt, A, chunk_size, cu_seqlens, dt_bias=dt_bias
    )

    torch.testing.assert_close(dt_out, dt_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dA_cumsum, dA_ref, rtol=1e-3, atol=1e-3)


# ============================================================================
# _chunk_state_fwd  (_chunk_state_fwd_kernel)
# ============================================================================


def _chunk_state_fwd_ref(B, x, dt, dA_cumsum, cu_chunk_seqlens, states_in_fp32=True):
    """Pure PyTorch reference for _chunk_state_fwd.

    B: (seqlen, ngroups, dstate); x: (seqlen, nheads, headdim);
    dt, dA_cumsum: (nheads, nchunks, chunk_size) float32.
    Returns states: (nchunks, nheads, headdim, dstate).
    """
    seqlen, nheads, headdim = x.shape
    _, ngroups, dstate = B.shape
    nheads_ngroups_ratio = nheads // ngroups
    nchunks = cu_chunk_seqlens.shape[0] - 1

    states_dtype = torch.float32 if states_in_fp32 else B.dtype
    states = torch.zeros(
        nchunks, nheads, headdim, dstate, device=x.device, dtype=states_dtype
    )

    for c in range(nchunks):
        chunk_start = cu_chunk_seqlens[c].item()
        chunk_end = cu_chunk_seqlens[c + 1].item()
        chunk_len = chunk_end - chunk_start

        for h in range(nheads):
            g = h // nheads_ngroups_ratio
            dA_cs_last = dA_cumsum[h, c, chunk_len - 1].float()

            for k in range(chunk_len):
                dA_cs_k = dA_cumsum[h, c, k].float()
                dt_k = dt[h, c, k].float()
                scale = torch.exp(torch.clamp(dA_cs_last - dA_cs_k, max=0.0)) * dt_k

                x_k = x[chunk_start + k, h, :].float()
                b_k = B[chunk_start + k, g, :].float()
                states[c, h] += x_k[:, None] * (b_k[None, :] * scale)

    return states


def _state_make_inputs(seqlen, nheads, headdim, ngroups, dstate, chunk_size, itype):
    nchunks = seqlen // chunk_size
    x = torch.randn(seqlen, nheads, headdim, device=DEVICE, dtype=itype)
    B = torch.randn(seqlen, ngroups, dstate, device=DEVICE, dtype=itype)
    dt = (
        torch.rand(nheads, nchunks, chunk_size, device=DEVICE, dtype=torch.float32)
        * 0.5
        + 0.01
    )
    dA_cumsum = (dt * -0.1).cumsum(dim=-1)
    cu_chunk_seqlens = torch.tensor(
        [i * chunk_size for i in range(nchunks + 1)],
        device=DEVICE,
        dtype=torch.int32,
    )
    return x, B, dt, dA_cumsum, cu_chunk_seqlens


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("nheads", [4, 8])
@pytest.mark.parametrize("chunk_size", [32, 64])
@torch.inference_mode()
def test_chunk_state_fwd_basic(chunk_size, nheads, itype):
    """Single chunk, ngroups == nheads (MHA)."""
    set_random_seed(0)
    seqlen = chunk_size
    headdim, dstate, ngroups = 64, 64, nheads
    x, B, dt, dA_cumsum, cu_seqlens = _state_make_inputs(
        seqlen, nheads, headdim, ngroups, dstate, chunk_size, itype
    )

    states = _chunk_state_fwd(B, x, dt, dA_cumsum, cu_seqlens)
    states_ref = _chunk_state_fwd_ref(B, x, dt, dA_cumsum, cu_seqlens)

    torch.testing.assert_close(states, states_ref, rtol=5e-2, atol=2e-2)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("nheads", [4, 8])
@torch.inference_mode()
def test_chunk_state_fwd_multi_chunk(nheads, itype):
    """Multiple chunks (4 chunks)."""
    set_random_seed(0)
    chunk_size = 64
    seqlen = chunk_size * 4
    headdim, dstate, ngroups = 64, 64, nheads
    x, B, dt, dA_cumsum, cu_seqlens = _state_make_inputs(
        seqlen, nheads, headdim, ngroups, dstate, chunk_size, itype
    )

    states = _chunk_state_fwd(B, x, dt, dA_cumsum, cu_seqlens)
    states_ref = _chunk_state_fwd_ref(B, x, dt, dA_cumsum, cu_seqlens)

    torch.testing.assert_close(states, states_ref, rtol=5e-2, atol=2e-2)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("nheads,ngroups", [(4, 4), (8, 2), (8, 1)])
@torch.inference_mode()
def test_chunk_state_fwd_gqa(nheads, ngroups, itype):
    """GQA: nheads != ngroups (grouped query attention for B)."""
    set_random_seed(0)
    seqlen, chunk_size = 128, 64
    headdim, dstate = 64, 64
    x, B, dt, dA_cumsum, cu_seqlens = _state_make_inputs(
        seqlen, nheads, headdim, ngroups, dstate, chunk_size, itype
    )

    states = _chunk_state_fwd(B, x, dt, dA_cumsum, cu_seqlens)
    states_ref = _chunk_state_fwd_ref(B, x, dt, dA_cumsum, cu_seqlens)

    torch.testing.assert_close(states, states_ref, rtol=5e-2, atol=2e-2)


@pytest.mark.parametrize("headdim", [32, 64, 128])
@pytest.mark.parametrize("dstate", [32, 64, 128])
@torch.inference_mode()
def test_chunk_state_fwd_dim_sweep(headdim, dstate):
    """Sweep headdim x dstate dimensions (exercises different autotune configs)."""
    set_random_seed(0)
    seqlen, nheads, chunk_size, ngroups = 64, 4, 64, 4
    x, B, dt, dA_cumsum, cu_seqlens = _state_make_inputs(
        seqlen, nheads, headdim, ngroups, dstate, chunk_size, torch.bfloat16
    )

    states = _chunk_state_fwd(B, x, dt, dA_cumsum, cu_seqlens)
    states_ref = _chunk_state_fwd_ref(B, x, dt, dA_cumsum, cu_seqlens)

    torch.testing.assert_close(states, states_ref, rtol=5e-2, atol=2e-2)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_chunk_state_fwd_fp32_output(itype):
    """Verify states output is fp32 when states_in_fp32=True (default)."""
    set_random_seed(0)
    seqlen, nheads, chunk_size = 64, 4, 64
    headdim, dstate, ngroups = 64, 64, 4
    x, B, dt, dA_cumsum, cu_seqlens = _state_make_inputs(
        seqlen, nheads, headdim, ngroups, dstate, chunk_size, itype
    )

    states = _chunk_state_fwd(B, x, dt, dA_cumsum, cu_seqlens, states_in_fp32=True)
    assert states.dtype == torch.float32

    states_ref = _chunk_state_fwd_ref(B, x, dt, dA_cumsum, cu_seqlens)
    torch.testing.assert_close(states, states_ref, rtol=5e-2, atol=2e-2)


# ============================================================================
# _bmm_chunk_fwd  (_bmm_chunk_fwd_kernel)
# ============================================================================


def _bmm_chunk_fwd_ref(a, b, chunk_size, cu_chunk_seqlens, causal=False):
    """Pure PyTorch reference for _bmm_chunk_fwd.

    a, b: (seqlen, ngroups, k). Returns (nchunks, ngroups, chunk_size, chunk_size).
    """
    seqlen, ngroups, k = a.shape
    nchunks = len(cu_chunk_seqlens) - 1
    out = torch.zeros(
        nchunks, ngroups, chunk_size, chunk_size, device=a.device, dtype=a.dtype
    )

    for c in range(nchunks):
        start = cu_chunk_seqlens[c].item()
        end = cu_chunk_seqlens[c + 1].item()
        chunk_len = end - start

        a_chunk = a[start:end]
        b_chunk = b[start:end]
        a_t = a_chunk.permute(1, 0, 2).float()
        b_t = b_chunk.permute(1, 0, 2).float()
        mm = torch.bmm(a_t, b_t.transpose(1, 2))
        out[c, :, :chunk_len, :chunk_len] = mm.to(a.dtype)

    if causal:
        mask = torch.tril(
            torch.ones(chunk_size, chunk_size, device=out.device, dtype=torch.bool)
        )
        out = out * mask.unsqueeze(0).unsqueeze(0)

    return out


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_size", [16, 64])
@pytest.mark.parametrize("ngroups", [1, 4])
@pytest.mark.parametrize("k_dim", [32, 64])
@torch.inference_mode()
def test_bmm_chunk_fwd(k_dim, ngroups, chunk_size, itype):
    """Single chunk, non-causal."""
    # Triton tl.dot may use reduced-precision tensor cores
    # (TF32 on CUDA, similar on XPU).
    rtol, atol = (1e-2, 2e-2) if itype == torch.float32 else (5e-2, 5e-2)
    set_random_seed(0)

    seqlen = chunk_size
    a = torch.randn(seqlen, ngroups, k_dim, device=DEVICE, dtype=itype)
    b = torch.randn(seqlen, ngroups, k_dim, device=DEVICE, dtype=itype)
    cu_chunk_seqlens = torch.tensor([0, seqlen], device=DEVICE, dtype=torch.int32)

    out = _bmm_chunk_fwd(a, b, chunk_size, cu_chunk_seqlens, causal=False)
    out_ref = _bmm_chunk_fwd_ref(a, b, chunk_size, cu_chunk_seqlens, causal=False)

    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_size", [16, 64])
@pytest.mark.parametrize("ngroups", [1, 4])
@torch.inference_mode()
def test_bmm_chunk_fwd_causal(ngroups, chunk_size, itype):
    """causal=True masking."""
    rtol, atol = (1e-2, 2e-2) if itype == torch.float32 else (5e-2, 5e-2)
    set_random_seed(0)

    k_dim = 32
    seqlen = chunk_size
    a = torch.randn(seqlen, ngroups, k_dim, device=DEVICE, dtype=itype)
    b = torch.randn(seqlen, ngroups, k_dim, device=DEVICE, dtype=itype)
    cu_chunk_seqlens = torch.tensor([0, seqlen], device=DEVICE, dtype=torch.int32)

    out = _bmm_chunk_fwd(a, b, chunk_size, cu_chunk_seqlens, causal=True)
    out_ref = _bmm_chunk_fwd_ref(a, b, chunk_size, cu_chunk_seqlens, causal=True)

    # Causal: only the lower-triangular region is defined.
    mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=DEVICE, dtype=torch.bool)
    )
    torch.testing.assert_close(
        out[:, :, mask], out_ref[:, :, mask], rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_size", [32, 64])
@torch.inference_mode()
def test_bmm_chunk_fwd_multi_chunk(chunk_size, itype):
    """Multiple chunks (seqlen > chunk_size)."""
    rtol, atol = (1e-2, 2e-2) if itype == torch.float32 else (5e-2, 5e-2)
    set_random_seed(0)

    ngroups, k_dim, nchunks = 4, 32, 3
    seqlen = chunk_size * nchunks
    a = torch.randn(seqlen, ngroups, k_dim, device=DEVICE, dtype=itype)
    b = torch.randn(seqlen, ngroups, k_dim, device=DEVICE, dtype=itype)
    cu_chunk_seqlens = torch.tensor(
        [i * chunk_size for i in range(nchunks + 1)], device=DEVICE, dtype=torch.int32
    )

    out = _bmm_chunk_fwd(a, b, chunk_size, cu_chunk_seqlens, causal=False)
    out_ref = _bmm_chunk_fwd_ref(a, b, chunk_size, cu_chunk_seqlens, causal=False)

    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_bmm_chunk_fwd_uneven_chunks(itype):
    """Uneven chunk lengths (last chunk shorter)."""
    rtol, atol = (1e-2, 2e-2) if itype == torch.float32 else (5e-2, 5e-2)
    set_random_seed(0)

    ngroups, k_dim, chunk_size = 2, 32, 64
    seqlen = 64 + 48  # full chunk + partial chunk
    a = torch.randn(seqlen, ngroups, k_dim, device=DEVICE, dtype=itype)
    b = torch.randn(seqlen, ngroups, k_dim, device=DEVICE, dtype=itype)
    cu_chunk_seqlens = torch.tensor([0, 64, 112], device=DEVICE, dtype=torch.int32)

    out = _bmm_chunk_fwd(a, b, chunk_size, cu_chunk_seqlens, causal=False)
    out_ref = _bmm_chunk_fwd_ref(a, b, chunk_size, cu_chunk_seqlens, causal=False)

    torch.testing.assert_close(out[0], out_ref[0], rtol=rtol, atol=atol)
    torch.testing.assert_close(
        out[1, :, :48, :48], out_ref[1, :, :48, :48], rtol=rtol, atol=atol
    )


# ============================================================================
# _chunk_scan_fwd  (_chunk_scan_fwd_kernel)
# ============================================================================


def _chunk_scan_fwd_ref(
    cb,
    x,
    dt,
    dA_cumsum,
    C,
    states,
    cu_chunk_seqlens,
    seq_idx,
    D=None,
    z=None,
    initial_states=None,
):
    """Pure PyTorch reference for _chunk_scan_fwd.

    Returns out: (seqlen, nheads, headdim).
    """
    seqlen, nheads, headdim = x.shape
    nchunks, ngroups, chunk_size, _ = cb.shape
    dstate = C.shape[2]
    nheads_ngroups_ratio = nheads // ngroups

    out = torch.zeros_like(x)

    for c in range(nchunks):
        chunk_start = cu_chunk_seqlens[c].item()
        chunk_end = cu_chunk_seqlens[c + 1].item()
        chunk_len = chunk_end - chunk_start

        for h in range(nheads):
            g = h // nheads_ngroups_ratio

            s_idx = seq_idx[c].item()
            s_idx_prev = seq_idx[c - 1].item() if c > 0 else -1
            if initial_states is not None and s_idx != s_idx_prev:
                prev_st = initial_states[s_idx, h]
            elif s_idx != s_idx_prev:
                prev_st = torch.zeros(headdim, dstate, device=x.device, dtype=x.dtype)
            else:
                prev_st = states[c - 1, h]

            for m in range(min(chunk_len, chunk_size)):
                dA_m = dA_cumsum[h, c, m].float()
                scale = torch.exp(dA_m)
                C_m = C[chunk_start + m, g, :].float()
                state_contrib = (prev_st.float() @ C_m) * scale

                intra = torch.zeros(headdim, device=x.device, dtype=torch.float32)
                for k in range(min(m + 1, chunk_len)):  # causal: k <= m
                    cb_mk = cb[c, g, m, k].float()
                    dA_k = dA_cumsum[h, c, k].float()
                    weight = cb_mk * torch.exp(torch.clamp(dA_m - dA_k, max=0.0))
                    weight = weight * dt[h, c, k].float()
                    intra += weight * x[chunk_start + k, h, :].float()

                val = state_contrib + intra
                if D is not None:
                    val += D[h].float() * x[chunk_start + m, h, :].float()
                if z is not None:
                    z_m = z[chunk_start + m, h, :].float()
                    val = val * z_m * torch.sigmoid(z_m)

                out[chunk_start + m, h, :] = val.to(x.dtype)

    return out


def _scan_make_inputs(
    seqlen,
    nheads,
    headdim,
    ngroups,
    dstate,
    chunk_size,
    itype,
    has_D=False,
    has_z=False,
):
    nchunks = seqlen // chunk_size
    # Small magnitudes keep exp() well-conditioned.
    cb = (
        torch.randn(
            nchunks, ngroups, chunk_size, chunk_size, device=DEVICE, dtype=itype
        )
        * 0.1
    )
    x = torch.randn(seqlen, nheads, headdim, device=DEVICE, dtype=itype) * 0.1
    dt = torch.rand(nheads, nchunks, chunk_size, device=DEVICE, dtype=itype) * 0.5
    dA_cumsum = (
        -torch.rand(nheads, nchunks, chunk_size, device=DEVICE, dtype=itype).cumsum(
            dim=-1
        )
        * 0.1
    )
    C = torch.randn(seqlen, ngroups, dstate, device=DEVICE, dtype=itype) * 0.1
    states = (
        torch.randn(nchunks, nheads, headdim, dstate, device=DEVICE, dtype=itype) * 0.1
    )
    out = torch.zeros(seqlen, nheads, headdim, device=DEVICE, dtype=itype)
    seq_idx = torch.zeros(nchunks, device=DEVICE, dtype=torch.int32)
    cu_chunk_seqlens = torch.tensor(
        [i * chunk_size for i in range(nchunks + 1)], device=DEVICE, dtype=torch.int32
    )

    D = (
        torch.randn(nheads, headdim, device=DEVICE, dtype=itype) * 0.1
        if has_D
        else None
    )
    z = (
        torch.randn(seqlen, nheads, headdim, device=DEVICE, dtype=itype) * 0.1
        if has_z
        else None
    )

    return cb, x, dt, dA_cumsum, C, states, cu_chunk_seqlens, out, seq_idx, D, z


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("headdim", [32, 64])
@pytest.mark.parametrize("dstate", [16, 64])
@torch.inference_mode()
def test_chunk_scan_fwd_basic(dstate, headdim, itype):
    """Single chunk, no D, no z."""
    rtol, atol = (1e-2, 2e-2) if itype == torch.float32 else (5e-2, 5e-2)
    set_random_seed(0)

    seqlen, nheads, ngroups, chunk_size = 64, 4, 2, 64
    cb, x, dt, dA_cumsum, C, states, cu_seqlens, out, seq_idx, _, _ = _scan_make_inputs(
        seqlen, nheads, headdim, ngroups, dstate, chunk_size, itype
    )

    _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, cu_seqlens, out, seq_idx)
    out_ref = _chunk_scan_fwd_ref(cb, x, dt, dA_cumsum, C, states, cu_seqlens, seq_idx)

    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("headdim", [32, 64])
@torch.inference_mode()
def test_chunk_scan_fwd_with_D(headdim, itype):
    """With D residual connection."""
    rtol, atol = (1e-2, 2e-2) if itype == torch.float32 else (5e-2, 5e-2)
    set_random_seed(0)

    seqlen, nheads, ngroups, dstate, chunk_size = 64, 4, 2, 16, 64
    cb, x, dt, dA_cumsum, C, states, cu_seqlens, out, seq_idx, D, _ = _scan_make_inputs(
        seqlen, nheads, headdim, ngroups, dstate, chunk_size, itype, has_D=True
    )

    _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, cu_seqlens, out, seq_idx, D=D)
    out_ref = _chunk_scan_fwd_ref(
        cb, x, dt, dA_cumsum, C, states, cu_seqlens, seq_idx, D=D
    )

    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("headdim", [32, 64])
@torch.inference_mode()
def test_chunk_scan_fwd_with_z(headdim, itype):
    """With z gating."""
    rtol, atol = (1e-2, 2e-2) if itype == torch.float32 else (5e-2, 5e-2)
    set_random_seed(0)

    seqlen, nheads, ngroups, dstate, chunk_size = 64, 4, 2, 16, 64
    cb, x, dt, dA_cumsum, C, states, cu_seqlens, out, seq_idx, _, z = _scan_make_inputs(
        seqlen, nheads, headdim, ngroups, dstate, chunk_size, itype, has_z=True
    )

    _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, cu_seqlens, out, seq_idx, z=z)
    out_ref = _chunk_scan_fwd_ref(
        cb, x, dt, dA_cumsum, C, states, cu_seqlens, seq_idx, z=z
    )

    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_chunk_scan_fwd_multi_chunk(itype):
    """Multiple chunks."""
    rtol, atol = (1e-2, 2e-2) if itype == torch.float32 else (5e-2, 5e-2)
    set_random_seed(0)

    seqlen, nheads, headdim, ngroups, dstate, chunk_size = 128, 4, 32, 2, 16, 64
    cb, x, dt, dA_cumsum, C, states, cu_seqlens, out, seq_idx, _, _ = _scan_make_inputs(
        seqlen, nheads, headdim, ngroups, dstate, chunk_size, itype
    )

    _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, cu_seqlens, out, seq_idx)
    out_ref = _chunk_scan_fwd_ref(cb, x, dt, dA_cumsum, C, states, cu_seqlens, seq_idx)

    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_chunk_scan_fwd_with_D_and_z(itype):
    """With both D and z."""
    rtol, atol = (1e-2, 2e-2) if itype == torch.float32 else (5e-2, 5e-2)
    set_random_seed(0)

    seqlen, nheads, headdim, ngroups, dstate, chunk_size = 64, 4, 32, 2, 16, 64
    cb, x, dt, dA_cumsum, C, states, cu_seqlens, out, seq_idx, D, z = _scan_make_inputs(
        seqlen,
        nheads,
        headdim,
        ngroups,
        dstate,
        chunk_size,
        itype,
        has_D=True,
        has_z=True,
    )

    _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, cu_seqlens, out, seq_idx, D=D, z=z)
    out_ref = _chunk_scan_fwd_ref(
        cb, x, dt, dA_cumsum, C, states, cu_seqlens, seq_idx, D=D, z=z
    )

    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("headdim", [32, 64, 128])
@torch.inference_mode()
def test_chunk_scan_fwd_headdim_sweep(headdim, itype):
    """Extended headdim sweep."""
    rtol, atol = (1e-2, 2e-2) if itype == torch.float32 else (5e-2, 5e-2)
    set_random_seed(0)

    seqlen, nheads, ngroups, dstate, chunk_size = 64, 4, 2, 16, 64
    cb, x, dt, dA_cumsum, C, states, cu_seqlens, out, seq_idx, D, z = _scan_make_inputs(
        seqlen,
        nheads,
        headdim,
        ngroups,
        dstate,
        chunk_size,
        itype,
        has_D=True,
        has_z=True,
    )

    _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, cu_seqlens, out, seq_idx, D=D, z=z)
    out_ref = _chunk_scan_fwd_ref(
        cb, x, dt, dA_cumsum, C, states, cu_seqlens, seq_idx, D=D, z=z
    )

    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)


# ============================================================================
# _state_passing_fwd  (_state_passing_fwd_kernel)
# ============================================================================


def _state_passing_fwd_ref(states, dA_cumsum, last_chunk_indices, initial_states=None):
    """Pure PyTorch reference for _state_passing_fwd.

    Returns out: (nchunks, nheads, dim).
    """
    nchunks, nheads, dim = states.shape
    chunk_size = dA_cumsum.shape[-1]
    batch = last_chunk_indices.shape[0]

    out = torch.zeros_like(states, dtype=states.dtype)

    for b in range(batch):
        chunk_end = last_chunk_indices[b].item() + 1
        chunk_start = (last_chunk_indices[b - 1].item() + 1) if b > 0 else 0

        if initial_states is not None:
            state = initial_states[b].float().clone()
        else:
            state = torch.zeros(nheads, dim, dtype=torch.float32, device=states.device)

        for c in range(chunk_start, chunk_end):
            new_states = states[c].float()
            for h in range(nheads):
                dA_cs = dA_cumsum[h, c, chunk_size - 1].float()
                state[h] = torch.exp(dA_cs) * state[h] + new_states[h]
                out[c, h] = state[h]

    return out


def _passing_make_inputs(
    nchunks, nheads, dim, chunk_size, batch, itype, with_initstates=False
):
    states = torch.randn(nchunks, nheads, dim, device=DEVICE, dtype=itype)
    dt = (
        torch.rand(nheads, nchunks, chunk_size, device=DEVICE, dtype=torch.float32)
        * 0.5
        + 0.01
    )
    dA_cumsum = (dt * -0.1).cumsum(dim=-1)
    chunks_per_seq = nchunks // batch
    last_chunk_indices = torch.tensor(
        [(b + 1) * chunks_per_seq - 1 for b in range(batch)],
        device=DEVICE,
        dtype=torch.int64,
    )
    initial_states = None
    if with_initstates:
        initial_states = torch.randn(batch, nheads, dim, device=DEVICE, dtype=itype)
    return states, dA_cumsum, last_chunk_indices, initial_states


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("nheads", [4, 8])
@pytest.mark.parametrize("chunk_size", [32, 64])
@torch.inference_mode()
def test_state_passing_basic(chunk_size, nheads, itype):
    """Single sequence, no initial states."""
    set_random_seed(0)
    nchunks, dim, batch = 4, 64, 1
    states, dA_cumsum, last_idx, _ = _passing_make_inputs(
        nchunks, nheads, dim, chunk_size, batch, itype
    )

    out = _state_passing_fwd(states, dA_cumsum, last_idx)
    out_ref = _state_passing_fwd_ref(states, dA_cumsum, last_idx)

    torch.testing.assert_close(out.float(), out_ref.float(), rtol=5e-2, atol=2e-2)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("nheads", [4, 8])
@torch.inference_mode()
def test_state_passing_with_initstates(nheads, itype):
    """With initial states (cached decoding scenario)."""
    set_random_seed(0)
    nchunks, dim, chunk_size, batch = 4, 64, 64, 1
    states, dA_cumsum, last_idx, initstates = _passing_make_inputs(
        nchunks, nheads, dim, chunk_size, batch, itype, with_initstates=True
    )

    out = _state_passing_fwd(states, dA_cumsum, last_idx, initial_states=initstates)
    out_ref = _state_passing_fwd_ref(
        states, dA_cumsum, last_idx, initial_states=initstates
    )

    torch.testing.assert_close(out.float(), out_ref.float(), rtol=5e-2, atol=2e-2)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [2, 4])
@torch.inference_mode()
def test_state_passing_multi_seq(batch, itype):
    """Multiple sequences packed into one batch."""
    set_random_seed(0)
    chunks_per_seq = 4
    nchunks = chunks_per_seq * batch
    nheads, dim, chunk_size = 4, 64, 64
    states, dA_cumsum, last_idx, _ = _passing_make_inputs(
        nchunks, nheads, dim, chunk_size, batch, itype
    )

    out = _state_passing_fwd(states, dA_cumsum, last_idx)
    out_ref = _state_passing_fwd_ref(states, dA_cumsum, last_idx)

    torch.testing.assert_close(out.float(), out_ref.float(), rtol=5e-2, atol=2e-2)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [2, 4])
@torch.inference_mode()
def test_state_passing_multi_seq_initstates(batch, itype):
    """Multiple sequences with initial states."""
    set_random_seed(0)
    chunks_per_seq = 4
    nchunks = chunks_per_seq * batch
    nheads, dim, chunk_size = 4, 64, 64
    states, dA_cumsum, last_idx, initstates = _passing_make_inputs(
        nchunks, nheads, dim, chunk_size, batch, itype, with_initstates=True
    )

    out = _state_passing_fwd(states, dA_cumsum, last_idx, initial_states=initstates)
    out_ref = _state_passing_fwd_ref(
        states, dA_cumsum, last_idx, initial_states=initstates
    )

    torch.testing.assert_close(out.float(), out_ref.float(), rtol=5e-2, atol=2e-2)


@pytest.mark.parametrize("dim", [64, 128, 256])
@torch.inference_mode()
def test_state_passing_dim_sweep(dim):
    """Sweep dims to exercise different BLOCK_SIZE autotune configs."""
    set_random_seed(0)
    nchunks, nheads, chunk_size, batch = 4, 4, 64, 1
    states, dA_cumsum, last_idx, _ = _passing_make_inputs(
        nchunks, nheads, dim, chunk_size, batch, torch.bfloat16
    )

    out = _state_passing_fwd(states, dA_cumsum, last_idx)
    out_ref = _state_passing_fwd_ref(states, dA_cumsum, last_idx)

    torch.testing.assert_close(out.float(), out_ref.float(), rtol=5e-2, atol=2e-2)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_state_passing_out_dtype(itype):
    """Output inherits input dtype by default."""
    set_random_seed(0)
    nchunks, nheads, dim, chunk_size, batch = 4, 4, 64, 64, 1
    states, dA_cumsum, last_idx, _ = _passing_make_inputs(
        nchunks, nheads, dim, chunk_size, batch, itype
    )

    out = _state_passing_fwd(states, dA_cumsum, last_idx)
    assert out.dtype == states.dtype

    out_ref = _state_passing_fwd_ref(states, dA_cumsum, last_idx)
    torch.testing.assert_close(out.float(), out_ref.float(), rtol=5e-2, atol=2e-2)


@torch.inference_mode()
def test_state_passing_single_chunk():
    """Edge case: single chunk per sequence."""
    set_random_seed(0)
    nchunks, nheads, dim, chunk_size, batch = 1, 4, 64, 64, 1
    states, dA_cumsum, last_idx, initstates = _passing_make_inputs(
        nchunks, nheads, dim, chunk_size, batch, torch.float32, with_initstates=True
    )

    out = _state_passing_fwd(states, dA_cumsum, last_idx, initial_states=initstates)
    out_ref = _state_passing_fwd_ref(
        states, dA_cumsum, last_idx, initial_states=initstates
    )

    torch.testing.assert_close(out.float(), out_ref.float(), rtol=5e-2, atol=2e-2)
