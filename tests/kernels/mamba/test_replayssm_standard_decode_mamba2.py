# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standard (autoregressive) decode correctness for the Mamba2 ReplaySSM kernels.

ReplaySSM caches the recent SSM inputs ``(x, dt, B)`` in a small ring buffer and
reconstructs / reads out the recurrent state on the fly, writing the full state
back to HBM only when the buffer flushes. This file checks that, over a
multi-step decode, the ReplaySSM output_only kernel reproduces the exact SSM
recurrence.

For every step we assert, against trusted oracles driven one token at a time:

  * the ReplaySSM output matches its pure-PyTorch reference, which models the
    kernel's exact arithmetic (including its bf16 reconstruction), at every
    precision,
  * when the state is fp32, the output also matches the baseline decode kernel
    (``selective_state_update``); at bf16 state the baseline legitimately
    differs -- it downcasts the fp32 state to bf16 every step, while ReplaySSM
    accumulates a whole buffer in fp32 and is the more accurate path,
  * when state and activations are both fp32, the output also matches the exact
    elementwise ``selective_state_update_ref``,
  * the cached inputs match the reference cache management,
  * the checkpoint state matches the reference (and the baseline at fp32 state).

State and activation/buffer precision are swept independently. Nemotron-3
defaults to fp32 SSM state (``mamba_ssm_cache_dtype=float32``), but bf16 state is
also supported; the buffer dtype follows the activation dtype and ``dt_cache``
is always fp32.
"""

import pytest
import torch

from tests.kernels.mamba.utils import (
    allocate_update_caches,
    selective_state_update_ref,
    selective_state_update_replayssm_output_only_ref,
)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_output_only import (  # noqa: E501
    selective_state_update_replayssm_output_only,
)
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    # fp32 demands true fp32 parity. A correct fp32 reconstruction (tl.dot with
    # input_precision="tf32x3"/"ieee") matches the elementwise baseline to
    # ~1e-5, while a TF32 reconstruction drifts to ~1e-2. atol=1e-3 sits between,
    # so it flags TF32 degradation yet passes a correct fp32 kernel. bf16 stays
    # loose (bf16 rounding dominates and is unaffected by the matmul precision).
    if dtype == torch.float32:
        return 1e-4, 1e-3
    return 6e-2, 2e-1


def _tied_A(nheads: int, headdim: int, dstate: int, device: str) -> torch.Tensor:
    A = -torch.rand(nheads, device=device) - 1.0
    return A.view(nheads, 1, 1).expand(nheads, headdim, dstate)


def _tied_dt(
    batch: int,
    nheads: int,
    headdim: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    dt = torch.randn(batch, nheads, device=device, dtype=dtype)
    return dt.unsqueeze(-1).expand(batch, nheads, headdim)


def _tied_dt_bias(nheads: int, headdim: int, device: str) -> torch.Tensor:
    dt_bias = torch.rand(nheads, device=device) - 4.0
    return dt_bias.view(nheads, 1).expand(nheads, headdim)


def _run_standard_decode(
    *,
    state_dtype: torch.dtype,
    act_dtype: torch.dtype,
    batch: int,
    nheads: int,
    headdim: int,
    ngroups: int,
    dstate: int,
    max_cache_len: int,
    num_steps: int,
    has_z: bool,
    dt_softplus: bool,
    use_dt_bias: bool,
    desync_write_pos: bool = False,
    seed: int = 0,
) -> None:
    """Drive ``num_steps`` decode steps and check every path against the
    trusted recurrence. ``state_dtype`` is the recurrent-state precision;
    ``act_dtype`` is the activation/buffer precision."""
    device = "cuda"
    both_fp32 = state_dtype == torch.float32 and act_dtype == torch.float32
    rtol, atol = _tolerances(torch.float32 if both_fp32 else torch.bfloat16)
    set_random_seed(seed)

    # One state copy per path; all start identical at the same precision.
    state0 = torch.randn(
        batch, nheads, headdim, dstate, dtype=state_dtype, device=device
    )
    state_anchor = state0.clone()
    state_baseline = state0.clone()
    state_cached = state0.clone()
    state_ref = state0.clone()

    A = _tied_A(nheads, headdim, dstate, device)
    dt_bias = _tied_dt_bias(nheads, headdim, device) if use_dt_bias else None
    D = torch.randn(nheads, headdim, device=device)

    # Caches follow the activation dtype (dt_cache is forced to fp32 inside).
    x_cache, dt_cache, B_cache, _ = allocate_update_caches(
        batch,
        nheads,
        ngroups,
        headdim,
        dstate,
        max_cache_len,
        device,
        act_dtype,
        act_dtype,
    )
    x_cache_ref, dt_cache_ref, B_cache_ref, _ = allocate_update_caches(
        batch,
        nheads,
        ngroups,
        headdim,
        dstate,
        max_cache_len,
        device,
        act_dtype,
        act_dtype,
    )
    bc_pre = torch.empty(
        batch, ngroups, max_cache_len, device=device, dtype=torch.float32
    )

    if desync_write_pos:
        # Rows start at different ring positions so they flush on different
        # steps, exercising per-row write-position handling.
        write_pos = (
            torch.arange(batch, device=device, dtype=torch.int32) % max_cache_len
        )
    else:
        write_pos = torch.zeros(batch, dtype=torch.int32, device=device)

    for _ in range(num_steps):
        x = torch.randn(batch, nheads, headdim, device=device, dtype=act_dtype)
        dt = _tied_dt(batch, nheads, headdim, device, act_dtype)
        B = torch.randn(batch, ngroups, dstate, device=device, dtype=act_dtype)
        C = torch.randn(batch, ngroups, dstate, device=device, dtype=act_dtype)
        z = torch.randn_like(x) if has_z else None
        is_flush = write_pos == max_cache_len - 1

        # Trusted recurrence (mutates state_anchor in place, returns output).
        out_anchor = selective_state_update_ref(
            state_anchor,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
        )

        # Upstream baseline decode kernel.
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
            dt_softplus=dt_softplus,
            out=out_baseline,
        )

        # ReplaySSM kernel under test + its pure-PyTorch reference.
        out_cached = torch.empty_like(x)
        common = dict(
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            x_cache=x_cache,
            dt_cache=dt_cache,
            B_cache=B_cache,
            write_pos=write_pos,
            is_flush=is_flush,
            max_cache_len=max_cache_len,
            out=out_cached,
        )
        selective_state_update_replayssm_output_only(
            state_cached, x, dt, A, B, C, bc_pre=bc_pre, **common
        )
        out_ref = selective_state_update_replayssm_output_only_ref(
            state_ref,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            x_cache=x_cache_ref,
            dt_cache=dt_cache_ref,
            B_cache=B_cache_ref,
            write_pos=write_pos,
            max_cache_len=max_cache_len,
        )

        # The reference models the kernel's exact arithmetic (including its bf16
        # reconstruction), so the kernel must match it tightly at every
        # precision. At fp32 this also flags any TF32 reconstruction drift.
        torch.testing.assert_close(out_cached, out_ref, rtol=rtol, atol=atol)
        # When the STATE is fp32 the baseline decode kernel is a valid oracle (it
        # does not downcast the state per step), so ReplaySSM must match it. At
        # bf16 state the baseline legitimately differs: it downcasts the fp32
        # state to bf16 every step, while ReplaySSM accumulates a whole buffer in
        # fp32 and is the MORE accurate path -- so it is not a tight oracle there.
        if state_dtype == torch.float32:
            torch.testing.assert_close(out_cached, out_baseline, rtol=rtol, atol=atol)
        # The exact elementwise reference is valid only when state AND
        # activations are fp32; otherwise it downcasts the state (at readout for
        # bf16 activations, or per step for bf16 state).
        if both_fp32:
            torch.testing.assert_close(out_cached, out_anchor, rtol=rtol, atol=atol)

        # Cached inputs match the reference cache management.
        torch.testing.assert_close(x_cache, x_cache_ref, rtol=rtol, atol=atol)
        torch.testing.assert_close(dt_cache, dt_cache_ref, rtol=rtol, atol=atol)
        torch.testing.assert_close(B_cache, B_cache_ref, rtol=rtol, atol=atol)

        # Checkpoint state at flush matches the reference (and, when the state is
        # fp32, the baseline kernel).
        if bool(is_flush.any()):
            torch.testing.assert_close(
                state_cached[is_flush], state_ref[is_flush], rtol=rtol, atol=atol
            )
            if state_dtype == torch.float32:
                torch.testing.assert_close(
                    state_cached[is_flush],
                    state_baseline[is_flush],
                    rtol=rtol,
                    atol=atol,
                )

        write_pos = torch.where(is_flush, torch.zeros_like(write_pos), write_pos + 1)


# State/activation precisions. fp32 state is the default; bf16/fp16 are the
# reduced-footprint configs. fp16 appears both as an activation dtype (fully-fp16
# model sfp16_afp16, or fp16 act over fp32 state s32_afp16) and as a state dtype
# under bf16 activations (sfp16_a16): fp16 has a finer mantissa than bf16 at the
# same 2 bytes, so it is a more accurate state at no extra footprint. We still
# skip fp16 state under fp32 activations (the unused low-state/high-act mix).
_PRECISIONS = [
    pytest.param((torch.float32, torch.float32), id="s32_a32"),
    pytest.param((torch.float32, torch.bfloat16), id="s32_a16"),
    pytest.param((torch.bfloat16, torch.bfloat16), id="s16_a16"),
    pytest.param((torch.float32, torch.float16), id="s32_afp16"),
    pytest.param((torch.float16, torch.float16), id="sfp16_afp16"),
    pytest.param((torch.float16, torch.bfloat16), id="sfp16_a16"),
]
# Small synthetic shapes for the full axis sweep (compile fast).
_SMALL_GEOMETRIES = [
    pytest.param((8, 64, 64, 4), id="small"),
    pytest.param((4, 64, 16, 1), id="tiny"),
]
# Production Mamba2 shapes (nheads, headdim, dstate, ngroups), TP=1.
_REAL_GEOMETRIES = [
    pytest.param((96, 80, 128, 8), id="nano4b"),
    pytest.param((128, 64, 128, 8), id="super120b"),
    pytest.param((256, 64, 128, 8), id="ultra550b"),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize("max_cache_len", [1, 4, 16])
@pytest.mark.parametrize("geometry", _SMALL_GEOMETRIES)
@pytest.mark.parametrize("has_z", [False, True])
def test_replayssm_standard_decode_matches_reference(
    precision: tuple[torch.dtype, torch.dtype],
    max_cache_len: int,
    geometry: tuple[int, int, int, int],
    has_z: bool,
):
    state_dtype, act_dtype = precision
    nheads, headdim, dstate, ngroups = geometry
    _run_standard_decode(
        state_dtype=state_dtype,
        act_dtype=act_dtype,
        batch=4,
        nheads=nheads,
        headdim=headdim,
        ngroups=ngroups,
        dstate=dstate,
        max_cache_len=max_cache_len,
        num_steps=2 * max_cache_len + 1,
        has_z=has_z,
        dt_softplus=True,
        use_dt_bias=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize("geometry", _REAL_GEOMETRIES)
def test_replayssm_standard_decode_real_geometry(
    precision: tuple[torch.dtype, torch.dtype],
    geometry: tuple[int, int, int, int],
):
    # Production Mamba2 shapes for the Nemotron-3 family (Nano-4B / Super-120B /
    # Ultra-550B), at the production buffer length (8). All three precisions,
    # including the bf16 state case.
    state_dtype, act_dtype = precision
    nheads, headdim, dstate, ngroups = geometry
    _run_standard_decode(
        state_dtype=state_dtype,
        act_dtype=act_dtype,
        batch=4,
        nheads=nheads,
        headdim=headdim,
        ngroups=ngroups,
        dstate=dstate,
        max_cache_len=8,
        num_steps=17,
        has_z=True,
        dt_softplus=True,
        use_dt_bias=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize(
    "precision",
    [
        pytest.param((torch.float32, torch.float32), id="s32_a32"),
        pytest.param((torch.float32, torch.bfloat16), id="s32_a16"),
        pytest.param((torch.float32, torch.float16), id="s32_afp16"),
    ],
)
def test_replayssm_standard_decode_desync_write_pos(
    precision: tuple[torch.dtype, torch.dtype],
):
    # Rows start at staggered ring positions, so they flush on different steps
    # and hold genuinely different cached histories.
    state_dtype, act_dtype = precision
    _run_standard_decode(
        state_dtype=state_dtype,
        act_dtype=act_dtype,
        batch=4,
        nheads=8,
        headdim=64,
        ngroups=4,
        dstate=64,
        max_cache_len=4,
        num_steps=12,
        has_z=True,
        dt_softplus=True,
        use_dt_bias=True,
        desync_write_pos=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize("with_padding", [False, True])
def test_replayssm_standard_decode_with_batch_indices(
    precision: tuple[torch.dtype, torch.dtype],
    with_padding: bool,
):
    # Sparse state allocation via state_batch_indices, with NULL_BLOCK_ID
    # padding rows. The pure-PyTorch references do not model the sparse
    # gather, so the anchor here is the upstream baseline decode kernel.
    state_dtype, act_dtype = precision
    device = "cuda"
    both_fp32 = state_dtype == torch.float32 and act_dtype == torch.float32
    rtol, atol = _tolerances(torch.float32 if both_fp32 else torch.bfloat16)
    set_random_seed(0)

    batch = 3
    padding = 2 if with_padding else 0
    padded_batch = batch + padding
    total_state_slots = 16
    nheads = 4
    ngroups = 2
    headdim = 64
    dstate = 16
    max_cache_len = 4
    num_steps = 2 * max_cache_len

    state = torch.randn(
        total_state_slots, nheads, headdim, dstate, dtype=state_dtype, device=device
    )
    state_baseline = state.clone()
    state_cached = state.clone()
    state_before = state.clone()

    state_indices = (
        torch.randperm(total_state_slots - 1, device=device)[:batch] + 1
    ).to(torch.int32)
    state_batch_indices = torch.cat(
        [
            state_indices,
            torch.full((padding,), NULL_BLOCK_ID, dtype=torch.int32, device=device),
        ]
    )
    unused_states = torch.ones(total_state_slots, dtype=torch.bool, device=device)
    unused_states[state_indices] = False

    A = _tied_A(nheads, headdim, dstate, device)
    dt_bias = _tied_dt_bias(nheads, headdim, device)
    D = torch.randn(nheads, headdim, device=device)
    x_cache = torch.zeros(
        total_state_slots,
        nheads,
        max_cache_len,
        headdim,
        device=device,
        dtype=act_dtype,
    )
    dt_cache = torch.zeros(
        total_state_slots, nheads, max_cache_len, device=device, dtype=torch.float32
    )
    B_cache = torch.zeros(
        total_state_slots,
        ngroups,
        max_cache_len,
        dstate,
        device=device,
        dtype=act_dtype,
    )
    bc_pre = torch.empty(
        padded_batch, ngroups, max_cache_len, device=device, dtype=torch.float32
    )
    write_pos = torch.zeros(padded_batch, dtype=torch.int32, device=device)

    for _ in range(num_steps):
        x = torch.randn(padded_batch, nheads, headdim, device=device, dtype=act_dtype)
        dt = _tied_dt(padded_batch, nheads, headdim, device, act_dtype)
        B = torch.randn(padded_batch, ngroups, dstate, device=device, dtype=act_dtype)
        C = torch.randn(padded_batch, ngroups, dstate, device=device, dtype=act_dtype)
        z = torch.randn_like(x)
        is_flush = write_pos == max_cache_len - 1

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
        common = dict(
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
            out=out_cached,
        )
        selective_state_update_replayssm_output_only(
            state_cached, x, dt, A, B, C, bc_pre=bc_pre, **common
        )

        torch.testing.assert_close(
            out_cached[:batch], out_baseline[:batch], rtol=rtol, atol=atol
        )
        if with_padding:
            assert torch.equal(
                out_cached[batch:], torch.full_like(out_cached[batch:], 42)
            )

        if bool(is_flush[:batch].all()):
            torch.testing.assert_close(
                state_cached[state_indices],
                state_baseline[state_indices],
                rtol=rtol,
                atol=atol,
            )

        write_pos = torch.where(is_flush, torch.zeros_like(write_pos), write_pos + 1)

    assert torch.equal(state_cached[unused_states], state_before[unused_states])
    assert torch.equal(state_baseline[unused_states], state_before[unused_states])


# Geometries with (nheads, ngroups) both divisible by the tp below.
_TP_GEOMETRIES = [
    pytest.param((8, 64, 64, 4), id="small"),
    pytest.param((96, 80, 128, 8), id="nano4b"),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize("geometry", _TP_GEOMETRIES)
@pytest.mark.parametrize("tp", [2])
def test_replayssm_standard_decode_tp_head_shard_equivalence(
    precision: tuple[torch.dtype, torch.dtype],
    geometry: tuple[int, int, int, int],
    tp: int,
):
    """Tensor-parallel correctness at the kernel boundary (single GPU).

    The kernel has no cross-rank communication, so head sharding must be exactly
    separable: one run over all heads must equal concatenating ``tp`` independent
    per-rank runs on ``nheads // tp`` heads and ``ngroups // tp`` groups. This
    guards the per-rank divisors and group->head mapping the state-shape wiring
    relies on. The real TP1==TP2 engine check lives in the v1/e2e suite."""
    state_dtype, act_dtype = precision
    nheads, headdim, dstate, ngroups = geometry
    assert nheads % tp == 0 and ngroups % tp == 0
    device = "cuda"
    both_fp32 = state_dtype == torch.float32 and act_dtype == torch.float32
    rtol, atol = _tolerances(torch.float32 if both_fp32 else torch.bfloat16)
    set_random_seed(0)

    batch = 4
    max_cache_len = 4
    num_steps = 2 * max_cache_len + 1
    nh_s = nheads // tp
    ng_s = ngroups // tp

    # Shards slice the tied params; never .contiguous() -- that would drop the
    # stride-0 broadcast the kernel's TIE_HDIM asserts require.
    A = _tied_A(nheads, headdim, dstate, device)
    dt_bias = _tied_dt_bias(nheads, headdim, device)
    D = torch.randn(nheads, headdim, device=device)

    state_full = torch.randn(
        batch, nheads, headdim, dstate, dtype=state_dtype, device=device
    )
    x_cache, dt_cache, B_cache, _ = allocate_update_caches(
        batch,
        nheads,
        ngroups,
        headdim,
        dstate,
        max_cache_len,
        device,
        act_dtype,
        act_dtype,
    )
    bc_pre = torch.empty(
        batch, ngroups, max_cache_len, device=device, dtype=torch.float32
    )

    # Per-rank shards, each seeded from the matching head slice so all start equal.
    shard_state = []
    shard_caches = []
    for r in range(tp):
        h0 = r * nh_s
        shard_state.append(state_full[:, h0 : h0 + nh_s].contiguous())
        xc, dtc, bc, _ = allocate_update_caches(
            batch,
            nh_s,
            ng_s,
            headdim,
            dstate,
            max_cache_len,
            device,
            act_dtype,
            act_dtype,
        )
        bcp = torch.empty(
            batch, ng_s, max_cache_len, device=device, dtype=torch.float32
        )
        shard_caches.append((xc, dtc, bc, bcp))

    write_pos = torch.zeros(batch, dtype=torch.int32, device=device)
    for _ in range(num_steps):
        x = torch.randn(batch, nheads, headdim, device=device, dtype=act_dtype)
        dt = _tied_dt(batch, nheads, headdim, device, act_dtype)
        B = torch.randn(batch, ngroups, dstate, device=device, dtype=act_dtype)
        C = torch.randn(batch, ngroups, dstate, device=device, dtype=act_dtype)
        z = torch.randn_like(x)
        is_flush = write_pos == max_cache_len - 1

        out_full = torch.empty_like(x)
        selective_state_update_replayssm_output_only(
            state_full,
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
            out=out_full,
        )

        for r in range(tp):
            h0 = r * nh_s
            g0 = r * ng_s
            xc, dtc, bc, bcp = shard_caches[r]
            out_shard = torch.empty(
                batch, nh_s, headdim, device=device, dtype=act_dtype
            )
            selective_state_update_replayssm_output_only(
                shard_state[r],
                x[:, h0 : h0 + nh_s].contiguous(),
                dt[:, h0 : h0 + nh_s],
                A[h0 : h0 + nh_s],
                B[:, g0 : g0 + ng_s].contiguous(),
                C[:, g0 : g0 + ng_s].contiguous(),
                D=D[h0 : h0 + nh_s].contiguous(),
                z=z[:, h0 : h0 + nh_s].contiguous(),
                dt_bias=dt_bias[h0 : h0 + nh_s],
                dt_softplus=True,
                x_cache=xc,
                dt_cache=dtc,
                B_cache=bc,
                bc_pre=bcp,
                write_pos=write_pos,
                is_flush=is_flush,
                max_cache_len=max_cache_len,
                out=out_shard,
            )

            torch.testing.assert_close(
                out_full[:, h0 : h0 + nh_s], out_shard, rtol=rtol, atol=atol
            )
            if bool(is_flush.any()):
                torch.testing.assert_close(
                    state_full[:, h0 : h0 + nh_s][is_flush],
                    shard_state[r][is_flush],
                    rtol=rtol,
                    atol=atol,
                )

        write_pos = torch.where(is_flush, torch.zeros_like(write_pos), write_pos + 1)
