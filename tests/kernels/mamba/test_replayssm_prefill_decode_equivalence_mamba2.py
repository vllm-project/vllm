# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill == decode equivalence for the Mamba2 ReplaySSM kernels.

The SSM recurrence is path-independent: vLLM's chunked prefill (the SSD kernel
``mamba_chunk_scan_combined_varlen``) and step-by-step decode must produce the
same per-position outputs and final state over a sequence. This file feeds one
set of inputs through the production dt flow (raw dt + softplus + a per-head
dt_bias, applied inside each kernel) to:

  * the exact fp32 step recurrence (``selective_state_update_ref``) -- the
    ground truth,
  * the chunked prefill kernel,
  * the baseline decode kernel,
  * both ReplaySSM decode routes,

and checks all of them agree. Prefill (a chunked scan) and decode (a step
recurrence) are different code paths, so they differ numerically: the chunked
scan carries ~2e-2 (fp32) / ~4e-2 (bf16) vs the exact recurrence, far above the
near-exact decode. We therefore anchor every path on the exact recurrence at
SSD-level tolerances (the same regime as ``test_mamba_ssm_ssd.py``), which the
chunked scan sets, keyed off the activation dtype.

State and activation/buffer precision are swept independently, including the
fp32-state + bf16-activation production config.
"""

import pytest
import torch

from tests.kernels.mamba.utils import selective_state_update_ref
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_output_only import (  # noqa: E501
    selective_state_update_replayssm_output_only,
)
from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_state_and_output import (  # noqa: E501
    selective_state_update_replayssm_state_and_output,
)
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.mamba2_attn import compute_varlen_chunk_metadata


def _prefill_tolerances(act_dtype: torch.dtype) -> tuple[float, float]:
    # The chunked prefill scan, not the decode, sets these: it carries ~2e-2
    # (fp32) / ~6e-2 (bf16) vs the exact recurrence, while ReplaySSM decode is
    # near-exact (~2e-6 fp32). Keyed off the activation dtype (the outputs are
    # in act_dtype). Same regime as test_mamba_ssm_ssd.py.
    if act_dtype == torch.float32:
        return 1e-2, 3e-2
    return 6e-2, 1e-1


def _run_prefill_decode_equivalence(
    *,
    route: str,
    state_dtype: torch.dtype,
    act_dtype: torch.dtype,
    nheads: int,
    headdim: int,
    ngroups: int,
    dstate: int,
    seqlen: int,
    chunk_size: int,
    max_cache_len: int,
    seed: int = 0,
) -> None:
    """Prefill the whole sequence and decode it step by step; check both match
    the exact fp32 recurrence (and each other). All paths use the production dt
    flow (raw dt + softplus + a per-head dt_bias), so this also checks prefill
    and decode apply the softplus/bias preprocessing consistently. ``state_dtype``
    is the recurrent-state precision; ``act_dtype`` the activation/buffer one."""
    device = "cuda"
    rtol, atol = _prefill_tolerances(act_dtype)
    set_random_seed(seed)

    # Production dt flow: raw dt + a per-head dt_bias (~-4 keeps softplus(dt +
    # bias) small and well-conditioned). dt_bias is (nheads,) for prefill and
    # (nheads, headdim) for the decode kernels/reference.
    A = -torch.exp(torch.rand(nheads, device=device, dtype=act_dtype))
    dt = torch.randn(seqlen, nheads, device=device, dtype=act_dtype)
    dt_bias = torch.rand(nheads, device=device, dtype=act_dtype) - 4
    dt_bias_hd = dt_bias.view(nheads, 1).expand(nheads, headdim)
    X = torch.randn(seqlen, nheads, headdim, device=device, dtype=act_dtype)
    B = torch.randn(seqlen, ngroups, dstate, device=device, dtype=act_dtype)
    C = torch.randn(seqlen, ngroups, dstate, device=device, dtype=act_dtype)
    A_bcast = A.view(nheads, 1, 1).expand(nheads, headdim, dstate)

    # Chunked prefill over the whole sequence (implicit batch=1, varlen). The
    # kernel always returns the final state in fp32, so no state_dtype plumbing.
    cu_seqlens = torch.tensor((0, seqlen), device=device).cumsum(0).to(torch.int32)
    cu_chunk_seqlens, last_chunk_indices, seq_idx = compute_varlen_chunk_metadata(
        cu_seqlens, chunk_size)
    y_prefill = torch.empty(seqlen, nheads, headdim, device=device, dtype=act_dtype)
    final_state_prefill = mamba_chunk_scan_combined_varlen(
        X, dt, A, B, C, chunk_size, cu_seqlens=cu_seqlens,
        cu_chunk_seqlens=cu_chunk_seqlens, last_chunk_indices=last_chunk_indices,
        seq_idx=seq_idx, out=y_prefill, D=None, dt_bias=dt_bias,
        dt_softplus=True)

    # Step paths: exact fp32 recurrence (ground truth), baseline, ReplaySSM.
    # State follows state_dtype; caches follow act_dtype (dt_cache is fp32).
    state_ref = torch.zeros(
        1, nheads, headdim, dstate, device=device, dtype=torch.float32)
    state_base = torch.zeros(
        1, nheads, headdim, dstate, device=device, dtype=state_dtype)
    state_dec = torch.zeros(
        1, nheads, headdim, dstate, device=device, dtype=state_dtype)
    x_cache = torch.zeros(
        1, nheads, max_cache_len, headdim, device=device, dtype=act_dtype)
    dt_cache = torch.zeros(
        1, nheads, max_cache_len, device=device, dtype=torch.float32)
    B_cache = torch.zeros(
        1, ngroups, max_cache_len, dstate, device=device, dtype=act_dtype)
    bc_pre = torch.empty(
        1, ngroups, max_cache_len, device=device, dtype=torch.float32)
    write_pos = torch.zeros(1, dtype=torch.int32, device=device)
    # No skip connection (D=0) on any path here; the D!=0 path is covered by the
    # standard-decode suite. The baseline kernel needs a D tensor, not None.
    D_zero = torch.zeros(nheads, headdim, device=device)

    y_ref = torch.empty(seqlen, nheads, headdim, device=device, dtype=torch.float32)
    y_base = torch.empty(seqlen, nheads, headdim, device=device, dtype=act_dtype)
    y_dec = torch.empty(seqlen, nheads, headdim, device=device, dtype=act_dtype)
    for t in range(seqlen):
        dt_t = dt[t].view(1, nheads, 1).expand(1, nheads, headdim)
        is_flush = write_pos == max_cache_len - 1

        y_ref[t] = selective_state_update_ref(
            state_ref, X[t:t + 1].float(), dt_t.float(), A_bcast.float(),
            B[t:t + 1].float(), C[t:t + 1].float(), dt_bias=dt_bias_hd.float(),
            dt_softplus=True)[0]

        out_b = torch.empty(1, nheads, headdim, device=device, dtype=act_dtype)
        selective_state_update(
            state_base, X[t:t + 1], dt_t, A_bcast, B[t:t + 1], C[t:t + 1],
            D=D_zero, dt_bias=dt_bias_hd, dt_softplus=True, out=out_b)
        y_base[t] = out_b[0]

        out_d = torch.empty(1, nheads, headdim, device=device, dtype=act_dtype)
        common = dict(
            dt_bias=dt_bias_hd, dt_softplus=True, x_cache=x_cache,
            dt_cache=dt_cache, B_cache=B_cache, write_pos=write_pos,
            is_flush=is_flush, max_cache_len=max_cache_len, out=out_d)
        if route == "output_only":
            selective_state_update_replayssm_output_only(
                state_dec, X[t:t + 1], dt_t, A_bcast, B[t:t + 1], C[t:t + 1],
                bc_pre=bc_pre, **common)
        else:
            selective_state_update_replayssm_state_and_output(
                state_dec, X[t:t + 1], dt_t, A_bcast, B[t:t + 1], C[t:t + 1],
                **common)
        y_dec[t] = out_d[0]

        write_pos = torch.where(
            is_flush, torch.zeros_like(write_pos), write_pos + 1)

    # Every path computes the same recurrence; anchor each on the fp32 truth.
    torch.testing.assert_close(y_prefill.float(), y_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(y_base.float(), y_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(y_dec.float(), y_ref, rtol=rtol, atol=atol)
    # Headline: ReplaySSM decode matches the chunked prefill directly.
    torch.testing.assert_close(y_dec.float(), y_prefill.float(), rtol=rtol, atol=atol)
    # Final state too (the recurrence ends in state_ref after the loop).
    torch.testing.assert_close(
        final_state_prefill[0].float(), state_ref[0], rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("route", ["output_only", "state_and_output"])
@pytest.mark.parametrize(
    "precision",
    # fp32 state is the default; bf16/fp16 are reduced-footprint configs. fp16
    # appears as an activation dtype (s32_afp16, sfp16_afp16) and as a finer-
    # mantissa state under bf16 activations (sfp16_a16).
    [
        pytest.param((torch.float32, torch.float32), id="s32_a32"),
        pytest.param((torch.float32, torch.bfloat16), id="s32_a16"),
        pytest.param((torch.bfloat16, torch.bfloat16), id="s16_a16"),
        pytest.param((torch.float32, torch.float16), id="s32_afp16"),
        pytest.param((torch.float16, torch.float16), id="sfp16_afp16"),
        pytest.param((torch.float16, torch.bfloat16), id="sfp16_a16"),
    ],
)
@pytest.mark.parametrize(
    "geometry",  # (nheads, headdim, dstate, ngroups)
    [
        pytest.param((8, 64, 64, 2), id="small"),
        pytest.param((96, 80, 128, 8), id="nano4b"),
    ],
)
def test_replayssm_prefill_decode_equivalence(
    route: str,
    precision: tuple[torch.dtype, torch.dtype],
    geometry: tuple[int, int, int, int],
):
    state_dtype, act_dtype = precision
    nheads, headdim, dstate, ngroups = geometry
    _run_prefill_decode_equivalence(
        route=route,
        state_dtype=state_dtype,
        act_dtype=act_dtype,
        nheads=nheads,
        headdim=headdim,
        ngroups=ngroups,
        dstate=dstate,
        seqlen=16,
        chunk_size=8,
        max_cache_len=4,
    )
