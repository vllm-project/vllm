# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the ReplaySSM opt-in Mamba2 AR-decode kernels.

ReplaySSM replaces the per-step recurrent-state write-back with a per-layer
input ring buffer that is replayed to recompute the output. These tests prove
the replay output matches:
  1. a pure-PyTorch replay reference, and
  2. the existing stored-state ``selective_state_update`` decode kernel
for both compute routes ("output_only" and "state_and_output"), including the
padded / state-batch-indices path, the unified ``ssu_dispatch`` funnel, and a
kernel-level TP1 == TP2 (head/group sharding) equivalence check.
"""

import pytest
import torch

from tests.kernels.mamba.utils import (
    allocate_update_caches,
    selective_state_update_replayssm_output_only_ref,
    selective_state_update_replayssm_state_and_output_ref,
)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_output_only import (  # noqa: E501
    selective_state_update_replayssm_output_only,
)
from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_state_and_output import (  # noqa: E501
    selective_state_update_replayssm_state_and_output,
)
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    selective_state_update_replayssm,
)
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-3, 1e-2
    return 6e-2, 2e-1


def _tied_A(nheads: int, headdim: int, dstate: int, device: str) -> torch.Tensor:
    A = -torch.rand(nheads, device=device) - 1.0
    return A.view(nheads, 1, 1).expand(nheads, headdim, dstate)


def _tied_dt(
    batch: int, nheads: int, headdim: int, device: str, dtype: torch.dtype
) -> torch.Tensor:
    dt = torch.randn(batch, nheads, device=device, dtype=dtype)
    return dt.unsqueeze(-1).expand(batch, nheads, headdim)


def _tied_dt_bias(nheads: int, headdim: int, device: str) -> torch.Tensor:
    dt_bias = torch.rand(nheads, device=device) - 4.0
    return dt_bias.view(nheads, 1).expand(nheads, headdim)


def _call_replay(
    route,
    state,
    x,
    dt,
    A,
    B,
    C,
    D,
    z,
    dt_bias,
    caches,
    bc_pre,
    write_pos,
    is_flush,
    max_cache_len,
    out,
    state_batch_indices=None,
):
    """Drive a single replay decode step through the chosen route."""
    x_cache, dt_cache, B_cache = caches
    if route == "output_only":
        selective_state_update_replayssm_output_only(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            x_cache=x_cache,
            dt_cache=dt_cache,
            B_cache=B_cache,
            bc_pre=bc_pre,
            write_pos=write_pos,
            is_flush=is_flush,
            max_cache_len=max_cache_len,
            state_batch_indices=state_batch_indices,
            out=out,
        )
    else:
        selective_state_update_replayssm_state_and_output(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            x_cache=x_cache,
            dt_cache=dt_cache,
            B_cache=B_cache,
            write_pos=write_pos,
            is_flush=is_flush,
            max_cache_len=max_cache_len,
            state_batch_indices=state_batch_indices,
            out=out,
        )


def _ref(route):
    if route == "output_only":
        return selective_state_update_replayssm_output_only_ref
    return selective_state_update_replayssm_state_and_output_ref


@pytest.mark.parametrize("route", ["output_only", "state_and_output"])
@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize("ngroups", [1, 4])
@pytest.mark.parametrize("dstate", [16, 64, 128])
@pytest.mark.parametrize("max_cache_len", [1, 4, 8])
def test_replayssm_matches_baseline_decode(
    max_cache_len, dstate, ngroups, has_z, itype, route
):
    """Replay decode == pure-torch ref == stored-state selective_state_update."""
    device = "cuda"
    rtol, atol = _tolerances(itype)
    set_random_seed(0)

    batch, nheads, headdim = 2, 4, 64
    num_steps = 2 * max_cache_len

    state = torch.randn(batch, nheads, headdim, dstate, dtype=itype, device=device)
    state_baseline = state.clone()
    state_cached = state.clone()
    state_ref = state.clone()

    A = _tied_A(nheads, headdim, dstate, device)
    dt_bias = _tied_dt_bias(nheads, headdim, device)
    D = torch.randn(nheads, headdim, device=device)

    x_cache, dt_cache, B_cache, write_pos = allocate_update_caches(
        batch,
        nheads,
        ngroups,
        headdim,
        dstate,
        max_cache_len,
        state.device,
        itype,
        itype,
    )
    xc_r, dtc_r, Bc_r, wp_r = allocate_update_caches(
        batch,
        nheads,
        ngroups,
        headdim,
        dstate,
        max_cache_len,
        state.device,
        itype,
        itype,
    )
    bc_pre = torch.empty(
        batch, ngroups, max_cache_len, device=device, dtype=torch.float32
    )

    for _ in range(num_steps):
        x = torch.randn(batch, nheads, headdim, device=device, dtype=itype)
        dt = _tied_dt(batch, nheads, headdim, device, itype)
        B = torch.randn(batch, ngroups, dstate, device=device, dtype=itype)
        C = torch.randn(batch, ngroups, dstate, device=device, dtype=itype)
        z = torch.randn_like(x) if has_z else None

        out_baseline = torch.empty_like(x)
        selective_state_update(
            state_baseline,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            out=out_baseline,
        )

        out_cached = torch.empty_like(x)
        is_flush = write_pos == max_cache_len - 1
        _call_replay(
            route,
            state_cached,
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            (x_cache, dt_cache, B_cache),
            bc_pre,
            write_pos,
            is_flush,
            max_cache_len,
            out_cached,
        )

        out_ref = _ref(route)(
            state_ref,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            x_cache=xc_r,
            dt_cache=dtc_r,
            B_cache=Bc_r,
            write_pos=wp_r,
            max_cache_len=max_cache_len,
        )

        torch.testing.assert_close(out_cached, out_ref, rtol=rtol, atol=atol)
        torch.testing.assert_close(out_cached, out_baseline, rtol=rtol, atol=atol)
        torch.testing.assert_close(x_cache, xc_r, rtol=rtol, atol=atol)
        torch.testing.assert_close(dt_cache, dtc_r, rtol=rtol, atol=atol)
        torch.testing.assert_close(B_cache, Bc_r, rtol=rtol, atol=atol)

        if bool(is_flush.all()):
            # On flush the rebuilt checkpoint must equal the stored state.
            torch.testing.assert_close(
                state_cached, state_baseline, rtol=rtol, atol=atol
            )

        next_wp = torch.where(is_flush, torch.zeros_like(write_pos), write_pos + 1)
        write_pos.copy_(next_wp)
        wp_r.copy_(next_wp)


@pytest.mark.parametrize("route", ["output_only", "state_and_output"])
@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_padding", [False, True])
def test_replayssm_with_state_batch_indices(with_padding, itype, route):
    """Padded rows (NULL_BLOCK_ID) and scattered state slots are handled."""
    device = "cuda"
    rtol, atol = _tolerances(itype)
    set_random_seed(0)

    batch = 3
    padding = 2 if with_padding else 0
    padded_batch = batch + padding
    total_slots = 16
    nheads, ngroups, headdim, dstate = 4, 2, 64, 16
    max_cache_len = 4
    num_steps = 2 * max_cache_len

    state = torch.randn(
        total_slots, nheads, headdim, dstate, dtype=itype, device=device
    )
    state_baseline = state.clone()
    state_cached = state.clone()
    state_before = state.clone()

    idx = (torch.randperm(total_slots - 1, device=device)[:batch] + 1).to(torch.int32)
    state_batch_indices = torch.cat(
        [
            idx,
            torch.full((padding,), NULL_BLOCK_ID, dtype=torch.int32, device=device),
        ]
    )
    unused = torch.ones(total_slots, dtype=torch.bool, device=device)
    unused[idx] = False

    A = _tied_A(nheads, headdim, dstate, device)
    dt_bias = _tied_dt_bias(nheads, headdim, device)
    D = torch.randn(nheads, headdim, device=device)
    x_cache = torch.zeros(
        total_slots, nheads, max_cache_len, headdim, device=device, dtype=itype
    )
    dt_cache = torch.zeros(
        total_slots, nheads, max_cache_len, device=device, dtype=torch.float32
    )
    B_cache = torch.zeros(
        total_slots, ngroups, max_cache_len, dstate, device=device, dtype=itype
    )
    bc_pre = torch.empty(
        padded_batch, ngroups, max_cache_len, device=device, dtype=torch.float32
    )
    write_pos = torch.zeros(padded_batch, dtype=torch.int32, device=device)

    for _ in range(num_steps):
        x = torch.randn(padded_batch, nheads, headdim, device=device, dtype=itype)
        dt = _tied_dt(padded_batch, nheads, headdim, device, itype)
        B = torch.randn(padded_batch, ngroups, dstate, device=device, dtype=itype)
        C = torch.randn(padded_batch, ngroups, dstate, device=device, dtype=itype)
        z = torch.randn_like(x)

        out_baseline = torch.empty_like(x)
        selective_state_update(
            state_baseline,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=state_batch_indices,
            out=out_baseline,
        )

        out_cached = torch.full_like(x, 42)
        is_flush = write_pos == max_cache_len - 1
        _call_replay(
            route,
            state_cached,
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            (x_cache, dt_cache, B_cache),
            bc_pre,
            write_pos,
            is_flush,
            max_cache_len,
            out_cached,
            state_batch_indices=state_batch_indices,
        )

        torch.testing.assert_close(
            out_cached[:batch], out_baseline[:batch], rtol=rtol, atol=atol
        )
        if with_padding:
            # Padded rows must be left untouched.
            assert torch.equal(
                out_cached[batch:], torch.full_like(out_cached[batch:], 42)
            )

        if bool(is_flush[:batch].all()):
            torch.testing.assert_close(
                state_cached[idx], state_baseline[idx], rtol=rtol, atol=atol
            )

        next_wp = torch.where(is_flush, torch.zeros_like(write_pos), write_pos + 1)
        write_pos.copy_(next_wp)

    # Untouched state slots stay byte-identical.
    assert torch.equal(state_cached[unused], state_before[unused])


@pytest.mark.parametrize("route", ["output_only", "state_and_output"])
def test_replayssm_dispatch_funnel_matches_direct(route):
    """The ssu_dispatch funnel must match a direct kernel call exactly."""
    device = "cuda"
    set_random_seed(0)
    batch, nheads, ngroups, headdim, dstate = 2, 4, 2, 64, 16
    max_cache_len = 4
    itype = torch.float32

    state = torch.randn(batch, nheads, headdim, dstate, dtype=itype, device=device)
    state_direct = state.clone()
    state_funnel = state.clone()
    A = _tied_A(nheads, headdim, dstate, device)
    dt_bias = _tied_dt_bias(nheads, headdim, device)
    D = torch.randn(nheads, headdim, device=device)

    cd, cf = (
        allocate_update_caches(
            batch, nheads, ngroups, headdim, dstate, max_cache_len, device, itype, itype
        )
        for _ in range(2)
    )
    xcd, dtcd, Bcd, wpd = cd
    xcf, dtcf, Bcf, wpf = cf
    bc_d = torch.empty(
        batch, ngroups, max_cache_len, device=device, dtype=torch.float32
    )
    bc_f = torch.empty_like(bc_d)

    for _ in range(2 * max_cache_len):
        x = torch.randn(batch, nheads, headdim, device=device, dtype=itype)
        dt = _tied_dt(batch, nheads, headdim, device, itype)
        B = torch.randn(batch, ngroups, dstate, device=device, dtype=itype)
        C = torch.randn(batch, ngroups, dstate, device=device, dtype=itype)
        z = torch.randn_like(x)

        out_direct = torch.empty_like(x)
        is_flush = wpd == max_cache_len - 1
        _call_replay(
            route,
            state_direct,
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            (xcd, dtcd, Bcd),
            bc_d,
            wpd,
            is_flush,
            max_cache_len,
            out_direct,
        )

        out_funnel = torch.empty_like(x)
        selective_state_update_replayssm(
            state_funnel,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            dt_bias=dt_bias,
            z=z,
            dt_softplus=True,
            x_cache=xcf,
            dt_cache=dtcf,
            B_cache=Bcf,
            write_pos=wpf,
            is_flush=is_flush,
            bc_pre=bc_f if route == "output_only" else None,
            route=route,
            max_cache_len=max_cache_len,
            out=out_funnel,
        )

        assert torch.equal(out_direct, out_funnel)
        next_wp = torch.where(is_flush, torch.zeros_like(wpd), wpd + 1)
        wpd.copy_(next_wp)
        wpf.copy_(next_wp)


@pytest.mark.parametrize("route", ["output_only", "state_and_output"])
def test_replayssm_tp1_equals_tp2(route):
    """Kernel-level TP1 == TP2: sharding heads/groups across two ranks and
    concatenating reproduces the full-width output bitwise (the decode kernel
    has no cross-head reduction, so column-parallel TP is exact)."""
    device = "cuda"
    set_random_seed(0)
    itype = torch.float32
    batch, nheads, ngroups, headdim, dstate = 2, 8, 2, 64, 16
    max_cache_len = 4
    tp = 2
    hpr, gpr = nheads // tp, ngroups // tp  # per-rank heads / groups

    state = torch.randn(batch, nheads, headdim, dstate, dtype=itype, device=device)
    A = _tied_A(nheads, headdim, dstate, device)
    dt_bias = _tied_dt_bias(nheads, headdim, device)
    D = torch.randn(nheads, headdim, device=device)

    # TP1 (full-width) state + caches.
    s1 = state.clone()
    xc1, dtc1, Bc1, wp1 = allocate_update_caches(
        batch, nheads, ngroups, headdim, dstate, max_cache_len, device, itype, itype
    )
    bc1 = torch.empty(batch, ngroups, max_cache_len, device=device, dtype=torch.float32)
    # TP2 (per-rank shards) state + caches.
    shards = []
    for r in range(tp):
        sr = state[:, r * hpr : (r + 1) * hpr].clone()
        xcr, dtcr, Bcr, wpr = allocate_update_caches(
            batch, hpr, gpr, headdim, dstate, max_cache_len, device, itype, itype
        )
        bcr = torch.empty(batch, gpr, max_cache_len, device=device, dtype=torch.float32)
        shards.append([sr, xcr, dtcr, Bcr, wpr, bcr])

    for _ in range(2 * max_cache_len):
        x = torch.randn(batch, nheads, headdim, device=device, dtype=itype)
        dt = _tied_dt(batch, nheads, headdim, device, itype)
        B = torch.randn(batch, ngroups, dstate, device=device, dtype=itype)
        C = torch.randn(batch, ngroups, dstate, device=device, dtype=itype)
        z = torch.randn_like(x)

        out1 = torch.empty_like(x)
        is_flush = wp1 == max_cache_len - 1
        _call_replay(
            route,
            s1,
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            (xc1, dtc1, Bc1),
            bc1,
            wp1,
            is_flush,
            max_cache_len,
            out1,
        )

        out2 = torch.empty_like(x)
        for r in range(tp):
            sr, xcr, dtcr, Bcr, wpr, bcr = shards[r]
            hs, gs = slice(r * hpr, (r + 1) * hpr), slice(r * gpr, (r + 1) * gpr)
            _call_replay(
                route,
                sr,
                x[:, hs],
                dt[:, hs],
                A[hs],
                B[:, gs],
                C[:, gs],
                D[hs],
                z[:, hs],
                dt_bias[hs],
                (xcr, dtcr, Bcr),
                bcr,
                wpr,
                is_flush,
                max_cache_len,
                out2[:, hs],
            )

        assert torch.equal(out1, out2)
        next_wp = torch.where(is_flush, torch.zeros_like(wp1), wp1 + 1)
        wp1.copy_(next_wp)
        for r in range(tp):
            shards[r][4].copy_(next_wp)
