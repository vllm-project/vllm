# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative-decode correctness for the Mamba2 ReplaySSM kernels.

A spec/verify step takes the checkpoint state ``S_0`` + the committed ring buffer
+ a window of ``max_spec_len = 1 + num_speculative_tokens`` draft tokens, and
computes the recurrence OUTPUT at each window position (causal: draft ``s`` reads
the buffer up to its own position). It does NOT write the state per draft -- only
a flush step folds the *committed* history into the checkpoint, so unaccepted
drafts can be rolled back.

The history window is ``L = B + max_spec_len`` (block ``B`` = ``buffer_len``): the
physical circular buffer is ``next_pow2(L)`` and ``max_cache_len`` passed to the
kernel is the logical ``L``. Verify launches the non-flush kernel; flush launches
the reconstruct kernel; both run every step with device-side row routing.

Oracle (no separate spec reference needed): the already-verified ReplaySSM
STANDARD decode kernel (``selective_state_update_replayssm_output_only``) stepped
one token at a time over the SAME window from the SAME checkpoint+buffer, with
``is_flush=False`` so it reads the SAME fixed checkpoint at every window position.
Its per-position output must equal the spec kernel's. For the multi-step rollback
the ground truth is the baseline ``selective_state_update`` decode of the accepted
token stream (it writes the full state every step).

Three checks per (precision, geometry, base_block, max_spec_len):
  * single spec step output == standard-decode-of-window, swept over the buffer
    fill level up to the max non-flush fill C = B - max_spec_len,
  * a 40-step rollback with a random accepted count k per step: accepted-position
    outputs track the baseline decode, and the committed checkpoint at each flush
    matches the baseline state at the matching folded-token count,
  * continuous-batching realism: sparse state slots via state_batch_indices with
    NULL padding rows and per-row buffer fill levels.

Precision: the checkpoint-readout dot is keyed on the activation dtype -- tf32x3
(~fp32 parity, 3-pass) for fp32 activations, single-pass tf32 for bf16 -- and the
cache GEMM uses tf32x3, so fp32 is near-exact (rel ~1e-6, tight tolerance). At bf16
activations the bf16 ``C``/``x``/``B`` operands dominate (rel ~5e-3), a small
EXPECTED gap that is bounded per step and non-accumulating across the decode.
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_output_only import (  # noqa: E501
    selective_state_update_replayssm_output_only,
)
from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_spec import (  # noqa: E501
    commit_replayssm_spec,
    reset_replayssm_spec_cursors,
    selective_state_update_replayssm_spec,
)
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

DEV = "cuda"


def _lbt(base_block: int, max_spec_len: int) -> tuple[int, int]:
    """Logical flush threshold L = B + max_spec_len and physical pow2 buffer."""
    L = base_block + max_spec_len
    buf = 1 << (L - 1).bit_length()
    return L, buf


def _tolerances(both_fp32: bool) -> tuple[float, float]:
    # fp32 activations use the tf32x3 checkpoint dot, matching the standard-decode
    # oracle to ~1e-6; atol=1e-3 still flags a regression to single-pass tf32. bf16
    # stays loose: the bf16 C/x/B operands dominate (rel ~5e-3), unaffected by the
    # checkpoint dot's precision.
    if both_fp32:
        return 1e-4, 1e-3
    return 6e-2, 2e-1


def _tied_A(nheads: int, headdim: int, dstate: int) -> torch.Tensor:
    # TIE_HDIM: A is scalar per head (stride(-1)==stride(-2)==0).
    A = -torch.rand(nheads, device=DEV) - 1.0
    return A.view(nheads, 1, 1).expand(nheads, headdim, dstate)


def _scatter_packed_history(
    post_conv_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    slot: int,
    x_hist: torch.Tensor,  # (wp, H, P)
    B_hist: torch.Tensor,  # (wp, G, N)
    dt_hist_raw: torch.Tensor,  # (wp, H) raw dt
    d_inner: int,
    ngroups: int,
    dstate: int,
) -> None:
    """Fill the committed history [0, wp) into the packed circular caches at
    post_origin=0 (so logical pos == physical pos). Channel map matches the
    kernel's x_view/B_view: x[h,d]->h*P+d, B[g,n]->d_inner+g*N+n (C is not
    cached; the kernel reads it fresh from conv_out). dt_cache stores RAW dt
    (the kernel applies bias+softplus on read)."""
    wp, H, P = x_hist.shape
    G, N = ngroups, dstate
    for p in range(wp):
        post_conv_cache[slot, p, :d_inner] = x_hist[p].reshape(d_inner)
        post_conv_cache[slot, p, d_inner : d_inner + G * N] = B_hist[p].reshape(G * N)
        # C not cached; only x|B seeded.
        dt_cache[slot, :, p] = dt_hist_raw[p].float()


def _pack_window_conv_out(
    x_w: torch.Tensor,  # (S, H, P)
    B_w: torch.Tensor,  # (S, G, N)
    C_w: torch.Tensor,  # (S, G, N)
    d_inner: int,
    ngroups: int,
    dstate: int,
    act_dtype: torch.dtype,
) -> torch.Tensor:
    S = x_w.shape[0]
    G, N = ngroups, dstate
    conv_dim = d_inner + 2 * G * N
    conv_out = torch.zeros(S, conv_dim, device=DEV, dtype=act_dtype)
    conv_out[:, :d_inner] = x_w.reshape(S, d_inner)
    conv_out[:, d_inner : d_inner + G * N] = B_w.reshape(S, G * N)
    conv_out[:, d_inner + G * N :] = C_w.reshape(S, G * N)
    return conv_out


def _standard_window_oracle(
    *,
    S0_slot: torch.Tensor,  # (H, P, N) checkpoint for the row
    x_all,
    dt_all,
    B_all,
    C_all,
    z_all,  # (T_tot, ...) history+window raw inputs
    A,
    D,
    dt_bias,
    dt_softplus,
    wp: int,
    spec_len: int,
    buffer_len: int,
    act_dtype: torch.dtype,
) -> torch.Tensor:
    """Step the standard output_only decode over history+window (is_flush=False
    throughout, so the checkpoint stays fixed). Return the window outputs
    (spec_len, H, P)."""
    H, P, N = S0_slot.shape
    G = B_all.shape[1]
    state = S0_slot[None].clone()  # (1, H, P, N)
    x_cache = torch.zeros(1, H, buffer_len, P, device=DEV, dtype=act_dtype)
    dt_cache = torch.zeros(1, H, buffer_len, device=DEV, dtype=torch.float32)
    B_cache = torch.zeros(1, G, buffer_len, N, device=DEV, dtype=act_dtype)
    bc_pre = torch.empty(1, G, buffer_len, device=DEV, dtype=torch.float32)
    no_flush = torch.zeros(1, device=DEV, dtype=torch.int8)
    win = []
    for t in range(wp + spec_len):
        out_t = torch.empty(1, H, P, device=DEV, dtype=act_dtype)
        selective_state_update_replayssm_output_only(
            state,
            x_all[t : t + 1],
            dt_all[t : t + 1, :, None].expand(1, H, P),
            A,
            B_all[t : t + 1],
            C_all[t : t + 1],
            D=D,
            z=z_all[t : t + 1] if z_all is not None else None,
            dt_bias=dt_bias[:, None].expand(H, P),
            dt_softplus=dt_softplus,
            x_cache=x_cache,
            dt_cache=dt_cache,
            B_cache=B_cache,
            write_pos=torch.tensor([t], device=DEV, dtype=torch.int32),
            is_flush=no_flush,
            max_cache_len=buffer_len,
            bc_pre=bc_pre,
            out=out_t,
        )
        if t >= wp:
            win.append(out_t.clone())
    return torch.cat(win, dim=0)


def _run_single_step(
    *,
    state_dtype,
    act_dtype,
    nheads,
    headdim,
    dstate,
    ngroups,
    buffer_len,  # history block B
    max_spec_len,
    wp,
    has_z,
    dt_softplus=True,
    seed=0,
    perturb=False,
):
    set_random_seed(seed)
    H, P, N, G = nheads, headdim, dstate, ngroups
    d_inner = H * P
    spec_len = max_spec_len
    L, buf = _lbt(buffer_len, max_spec_len)
    both_fp32 = state_dtype == torch.float32 and act_dtype == torch.float32
    rtol, atol = _tolerances(both_fp32)

    A = _tied_A(H, P, N)
    dt_bias = torch.rand(H, device=DEV) - 4.0
    D = torch.randn(H, P, device=DEV)

    T_tot = wp + spec_len
    x = torch.randn(T_tot, H, P, device=DEV, dtype=act_dtype)
    dt = torch.randn(T_tot, H, device=DEV, dtype=act_dtype)
    B = torch.randn(T_tot, G, N, device=DEV, dtype=act_dtype)
    C = torch.randn(T_tot, G, N, device=DEV, dtype=act_dtype)
    z = torch.randn(T_tot, H, P, device=DEV, dtype=act_dtype) if has_z else None

    num_blocks = 2
    S0 = torch.randn(num_blocks, H, P, N, device=DEV, dtype=state_dtype) * 0.1

    oracle = _standard_window_oracle(
        S0_slot=S0[1],
        x_all=x,
        dt_all=dt,
        B_all=B,
        C_all=C,
        z_all=z,
        A=A,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        wp=wp,
        spec_len=spec_len,
        buffer_len=buf,
        act_dtype=act_dtype,
    )

    # spec path: prefill packed history, one verify call (non-flush).
    state_spec = S0.clone()
    cache_conv_dim = d_inner + G * N  # x|B only (no C)
    post_conv_cache = torch.zeros(
        num_blocks, buf, cache_conv_dim, device=DEV, dtype=act_dtype
    )
    dt_cache = torch.zeros(num_blocks, H, buf, device=DEV, dtype=torch.float32)
    _scatter_packed_history(
        post_conv_cache, dt_cache, 1, x[:wp], B[:wp], dt[:wp], d_inner, G, N
    )
    if perturb:
        # teeth: corrupt one history slot -> outputs must diverge from the oracle.
        post_conv_cache[1, max(0, wp - 1), 0] += 5.0

    conv_out = _pack_window_conv_out(x[wp:], B[wp:], C[wp:], d_inner, G, N, act_dtype)
    dt_spec = dt[wp:].float()
    z_spec = z[wp:] if has_z else None
    write_pos = torch.zeros(num_blocks, dtype=torch.int32, device=DEV)
    write_pos[1] = wp
    post_origin = torch.zeros(num_blocks, dtype=torch.int32, device=DEV)
    is_flush = torch.zeros(num_blocks, dtype=torch.int8, device=DEV)
    qsl = torch.tensor([0, spec_len], device=DEV, dtype=torch.int32)
    sbi = torch.tensor([1], device=DEV, dtype=torch.int32)
    out_spec = torch.empty(spec_len, H, P, device=DEV, dtype=act_dtype)
    selective_state_update_replayssm_spec(
        state_spec,
        post_conv_cache,
        dt_cache,
        conv_out,
        dt_spec,
        A,
        write_pos=write_pos,
        post_conv_state_pos=post_origin,
        is_flush=is_flush,
        query_start_loc=qsl,
        state_batch_indices=sbi,
        max_cache_len=L,
        max_spec_len=max_spec_len,
        d_inner=d_inner,
        ngroups=G,
        dstate=N,
        D=D,
        z=z_spec,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        out=out_spec,
    )

    torch.testing.assert_close(out_spec, oracle, rtol=rtol, atol=atol)
    # non-flush verify must not touch the checkpoint.
    torch.testing.assert_close(state_spec, S0, rtol=0, atol=0)


def _wp_set(base_block: int, max_spec_len: int) -> list[int]:
    # Verify-path fills: 0 (empty buffer = pure checkpoint readout), the max
    # non-flush fill C = B - max_spec_len (the tightest edge), and a midpoint.
    C = base_block - max_spec_len
    return sorted({0, max(0, C // 2), max(0, C)})


_PRECISIONS = [
    pytest.param((torch.float32, torch.float32), id="s32_a32"),
    pytest.param((torch.float32, torch.bfloat16), id="s32_a16"),
    pytest.param((torch.bfloat16, torch.bfloat16), id="s16_a16"),
]
# (nheads, headdim, dstate, ngroups). The full precision x block x max_spec_len
# sweep runs on the small shapes (cheap compiles); the production Nemotron-3
# Mamba2 shapes are exercised by test_spec_step_real_geometry.
_SMALL = pytest.param((8, 64, 64, 4), id="small")
_TINY = pytest.param((4, 64, 16, 1), id="tiny")
_REAL_GEOMETRIES = [
    pytest.param((96, 80, 128, 8), id="nano4b"),
    pytest.param((128, 64, 128, 8), id="super120b"),
    pytest.param((256, 64, 128, 8), id="ultra550b"),
]
_BASE_BLOCKS = [16, 32]  # history block B (replayssm_buffer_len)
_MAX_SPEC_LENS = [2, 4, 6, 8]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize("geometry", [_SMALL, _TINY])
@pytest.mark.parametrize("base_block", _BASE_BLOCKS)
@pytest.mark.parametrize("max_spec_len", _MAX_SPEC_LENS)
@pytest.mark.parametrize("has_z", [False, True])
def test_spec_step_matches_standard_decode(
    precision, geometry, base_block, max_spec_len, has_z
):
    # base_block >= max_spec_len always holds here. Sweep the non-flush fill level.
    state_dtype, act_dtype = precision
    nheads, headdim, dstate, ngroups = geometry
    for wp in _wp_set(base_block, max_spec_len):
        _run_single_step(
            state_dtype=state_dtype,
            act_dtype=act_dtype,
            nheads=nheads,
            headdim=headdim,
            dstate=dstate,
            ngroups=ngroups,
            buffer_len=base_block,
            max_spec_len=max_spec_len,
            wp=wp,
            has_z=has_z,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize("geometry", _REAL_GEOMETRIES)
@pytest.mark.parametrize("max_spec_len", [2, 8])
def test_spec_step_real_geometry(precision, geometry, max_spec_len):
    # Production Nemotron-3 Mamba2 shapes (Nano-4B / Super-120B / Ultra-550B) at
    # the deployable block B=16, both ends of the max_spec_len range.
    state_dtype, act_dtype = precision
    nheads, headdim, dstate, ngroups = geometry
    for wp in _wp_set(16, max_spec_len):
        _run_single_step(
            state_dtype=state_dtype,
            act_dtype=act_dtype,
            nheads=nheads,
            headdim=headdim,
            dstate=dstate,
            ngroups=ngroups,
            buffer_len=16,
            max_spec_len=max_spec_len,
            wp=wp,
            has_z=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_spec_step_teeth():
    # A correct run passes; corrupting one cached history value must break the
    # output match (guards against a vacuous oracle).
    _run_single_step(
        state_dtype=torch.float32,
        act_dtype=torch.float32,
        nheads=8,
        headdim=64,
        dstate=64,
        ngroups=4,
        buffer_len=16,
        max_spec_len=4,
        wp=5,
        has_z=True,
    )
    with pytest.raises(AssertionError):
        _run_single_step(
            state_dtype=torch.float32,
            act_dtype=torch.float32,
            nheads=8,
            headdim=64,
            dstate=64,
            ngroups=4,
            buffer_len=16,
            max_spec_len=4,
            wp=5,
            has_z=True,
            perturb=True,
        )


def _run_rollback(
    *,
    state_dtype,
    act_dtype,
    nheads,
    headdim,
    dstate,
    ngroups,
    buffer_len,  # history block B
    max_spec_len,
    num_steps=40,
    has_z=True,
    seed=0,
):
    """Drive the verify+commit loop from an empty buffer; accept a random k each
    step. The baseline ``selective_state_update`` decode of the accepted stream is
    the ground truth for outputs; the committed checkpoint at each flush must match
    the baseline state at the matching folded-token count (n_folded bookkeeping)."""
    set_random_seed(seed)
    H, P, N, G = nheads, headdim, dstate, ngroups
    d_inner = H * P
    L, buf = _lbt(buffer_len, max_spec_len)
    both_fp32 = state_dtype == torch.float32 and act_dtype == torch.float32
    rtol, atol = _tolerances(both_fp32)

    A = _tied_A(H, P, N)
    dt_bias = torch.rand(H, device=DEV) - 4.0
    D = torch.randn(H, P, device=DEV)

    num_blocks = 2
    S0 = torch.randn(num_blocks, H, P, N, device=DEV, dtype=state_dtype) * 0.1
    state_spec = S0.clone()
    state_base = S0.clone()  # full pool; row in slot 1

    cache_conv_dim = d_inner + G * N  # x|B only (no C)
    post_conv_cache = torch.zeros(
        num_blocks, buf, cache_conv_dim, device=DEV, dtype=act_dtype
    )
    dt_cache = torch.zeros(num_blocks, H, buf, device=DEV, dtype=torch.float32)
    write_pos = torch.zeros(num_blocks, dtype=torch.int32, device=DEV)
    post_origin = torch.zeros(num_blocks, dtype=torch.int32, device=DEV)
    is_flush = torch.zeros(num_blocks, dtype=torch.int8, device=DEV)
    sbi = torch.tensor([1], device=DEV, dtype=torch.int32)
    reset_replayssm_spec_cursors(
        write_pos,
        post_origin,
        is_flush,
        torch.ones(1, dtype=torch.int8, device=DEV),
        sbi,
        L,
        max_spec_len,
    )

    n_folded = 0
    snapshots = {0: state_base[1].clone()}
    total_accepted = 0
    g = torch.Generator(device="cpu").manual_seed(seed + 1)

    for _ in range(num_steps):
        spec_len = max_spec_len
        x = torch.randn(spec_len, H, P, device=DEV, dtype=act_dtype)
        dt = torch.randn(spec_len, H, device=DEV, dtype=act_dtype)
        Bw = torch.randn(spec_len, G, N, device=DEV, dtype=act_dtype)
        Cw = torch.randn(spec_len, G, N, device=DEV, dtype=act_dtype)
        zw = torch.randn(spec_len, H, P, device=DEV, dtype=act_dtype) if has_z else None

        conv_out = _pack_window_conv_out(x, Bw, Cw, d_inner, G, N, act_dtype)
        dt_spec = dt.float()
        qsl = torch.tensor([0, spec_len], device=DEV, dtype=torch.int32)
        out_spec = torch.empty(spec_len, H, P, device=DEV, dtype=act_dtype)

        wp_before = int(write_pos[1].item())
        flush_before = int(is_flush[1].item())
        selective_state_update_replayssm_spec(
            state_spec,
            post_conv_cache,
            dt_cache,
            conv_out,
            dt_spec,
            A,
            write_pos=write_pos,
            post_conv_state_pos=post_origin,
            is_flush=is_flush,
            query_start_loc=qsl,
            state_batch_indices=sbi,
            max_cache_len=L,
            max_spec_len=max_spec_len,
            d_inner=d_inner,
            ngroups=G,
            dstate=N,
            D=D,
            z=zw,
            dt_bias=dt_bias,
            dt_softplus=True,
            out=out_spec,
        )
        # a flush verify folds the committed history [0, wp_before) into S_0.
        if flush_before and wp_before > 0:
            n_folded += wp_before

        k = int(torch.randint(1, spec_len + 1, (1,), generator=g).item())

        base_out = []
        for s in range(k):
            ot = torch.empty(1, H, P, device=DEV, dtype=act_dtype)
            selective_state_update(
                state_base,
                x[s : s + 1],
                dt[s : s + 1, :, None].expand(1, H, P),
                A,
                Bw[s : s + 1],
                Cw[s : s + 1],
                D=D,
                z=zw[s : s + 1] if has_z else None,
                dt_bias=dt_bias[:, None].expand(H, P),
                dt_softplus=True,
                state_batch_indices=sbi,
                out=ot,
            )
            base_out.append(ot.clone())
            total_accepted += 1
            snapshots[total_accepted] = state_base[1].clone()
        base_out = torch.cat(base_out, dim=0)

        # (a) accepted-position outputs track the baseline decode.
        torch.testing.assert_close(out_spec[:k], base_out, rtol=rtol, atol=atol)

        commit_replayssm_spec(
            write_pos,
            post_origin,
            is_flush,
            torch.tensor([k], device=DEV, dtype=torch.int32),
            sbi,
            L,
            max_spec_len,
        )

        # (b) committed checkpoint == baseline state at the folded-token count.
        if n_folded in snapshots:
            torch.testing.assert_close(
                state_spec[1], snapshots[n_folded], rtol=rtol, atol=atol
            )

    # The buffer must have flushed at least once over 40 steps (else the state
    # check above never ran) -- otherwise the test is vacuous.
    assert n_folded > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize("base_block", _BASE_BLOCKS)
@pytest.mark.parametrize("max_spec_len", _MAX_SPEC_LENS)
def test_spec_rollback_tracks_baseline(precision, base_block, max_spec_len):
    state_dtype, act_dtype = precision
    _run_rollback(
        state_dtype=state_dtype,
        act_dtype=act_dtype,
        nheads=8,
        headdim=64,
        dstate=64,
        ngroups=4,
        buffer_len=base_block,
        max_spec_len=max_spec_len,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize("with_padding", [False, True])
def test_spec_continuous_batching(precision, with_padding):
    """Sparse state slots via state_batch_indices with NULL padding rows and a
    DIFFERENT buffer fill level per row, in a single verify call. Each active
    row is checked against its own standard-decode window oracle; padding-row
    outputs and unused state slots are left untouched."""
    state_dtype, act_dtype = precision
    set_random_seed(0)
    H, P, N, G = 4, 64, 16, 2
    d_inner = H * P
    conv_dim = d_inner + 2 * G * N
    base_block = 16
    max_spec_len = 4
    spec_len = max_spec_len
    L, buf = _lbt(base_block, max_spec_len)
    both_fp32 = state_dtype == torch.float32 and act_dtype == torch.float32
    rtol, atol = _tolerances(both_fp32)

    batch = 3
    padding = 2 if with_padding else 0
    padded_batch = batch + padding
    total_slots = 12
    C = base_block - max_spec_len
    wps = [0, C // 2, C]  # staggered non-flush fills incl. the tight edge

    A = _tied_A(H, P, N)
    dt_bias = torch.rand(H, device=DEV) - 4.0
    D = torch.randn(H, P, device=DEV)

    state_indices = (torch.randperm(total_slots - 1, device=DEV)[:batch] + 1).to(
        torch.int32
    )
    sbi = torch.cat(
        [
            state_indices,
            torch.full((padding,), NULL_BLOCK_ID, dtype=torch.int32, device=DEV),
        ]
    )
    unused = torch.ones(total_slots, dtype=torch.bool, device=DEV)
    unused[state_indices] = False

    S0 = torch.randn(total_slots, H, P, N, device=DEV, dtype=state_dtype) * 0.1
    state_spec = S0.clone()
    post_conv_cache = torch.zeros(
        total_slots, buf, d_inner + G * N, device=DEV, dtype=act_dtype
    )
    dt_cache = torch.zeros(total_slots, H, buf, device=DEV, dtype=torch.float32)
    write_pos = torch.zeros(total_slots, dtype=torch.int32, device=DEV)
    post_origin = torch.zeros(total_slots, dtype=torch.int32, device=DEV)
    is_flush = torch.zeros(total_slots, dtype=torch.int8, device=DEV)

    # per-row raw inputs (history + window); build the per-row oracle.
    oracles = []
    conv_out = torch.zeros(
        padded_batch * spec_len, conv_dim, device=DEV, dtype=act_dtype
    )
    dt_spec = torch.zeros(padded_batch * spec_len, H, device=DEV, dtype=torch.float32)
    z_pack = torch.zeros(padded_batch * spec_len, H, P, device=DEV, dtype=act_dtype)
    for r in range(batch):
        wp = wps[r]
        slot = int(state_indices[r].item())
        T_tot = wp + spec_len
        x = torch.randn(T_tot, H, P, device=DEV, dtype=act_dtype)
        dt = torch.randn(T_tot, H, device=DEV, dtype=act_dtype)
        Bv = torch.randn(T_tot, G, N, device=DEV, dtype=act_dtype)
        Cv = torch.randn(T_tot, G, N, device=DEV, dtype=act_dtype)
        zv = torch.randn(T_tot, H, P, device=DEV, dtype=act_dtype)
        oracles.append(
            _standard_window_oracle(
                S0_slot=S0[slot],
                x_all=x,
                dt_all=dt,
                B_all=Bv,
                C_all=Cv,
                z_all=zv,
                A=A,
                D=D,
                dt_bias=dt_bias,
                dt_softplus=True,
                wp=wp,
                spec_len=spec_len,
                buffer_len=buf,
                act_dtype=act_dtype,
            )
        )
        _scatter_packed_history(
            post_conv_cache,
            dt_cache,
            slot,
            x[:wp],
            Bv[:wp],
            dt[:wp],
            d_inner,
            G,
            N,
        )
        write_pos[slot] = wp
        seg = slice(r * spec_len, (r + 1) * spec_len)
        conv_out[seg] = _pack_window_conv_out(
            x[wp:], Bv[wp:], Cv[wp:], d_inner, G, N, act_dtype
        )
        dt_spec[seg] = dt[wp:].float()
        z_pack[seg] = zv[wp:]

    qsl = torch.arange(
        0, (padded_batch + 1) * spec_len, spec_len, device=DEV, dtype=torch.int32
    )
    out_spec = torch.full(
        (padded_batch * spec_len, H, P), 42.0, device=DEV, dtype=act_dtype
    )
    selective_state_update_replayssm_spec(
        state_spec,
        post_conv_cache,
        dt_cache,
        conv_out,
        dt_spec,
        A,
        write_pos=write_pos,
        post_conv_state_pos=post_origin,
        is_flush=is_flush,
        query_start_loc=qsl,
        state_batch_indices=sbi,
        max_cache_len=L,
        max_spec_len=max_spec_len,
        d_inner=d_inner,
        ngroups=G,
        dstate=N,
        D=D,
        z=z_pack,
        dt_bias=dt_bias,
        dt_softplus=True,
        out=out_spec,
    )

    for r in range(batch):
        seg = slice(r * spec_len, (r + 1) * spec_len)
        torch.testing.assert_close(out_spec[seg], oracles[r], rtol=rtol, atol=atol)
    if with_padding:
        pad = slice(batch * spec_len, padded_batch * spec_len)
        assert torch.equal(out_spec[pad], torch.full_like(out_spec[pad], 42.0))
    # non-flush verify leaves every state slot untouched.
    assert torch.equal(state_spec[unused], S0[unused])
    torch.testing.assert_close(state_spec, S0, rtol=0, atol=0)
