# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative-decode correctness for the GDN (Gated DeltaNet) ReplaySSM kernel.

A spec/verify step takes the checkpoint state + the committed ring buffer (the
per-step delta-rule ``d``/``k``/``g`` caches) + a window of ``max_spec_len =
1 + num_speculative_tokens`` draft tokens, and computes the recurrence OUTPUT at
each window position (causal). It does NOT write the state per draft -- only a
flush step folds the *committed* history into the checkpoint, so unaccepted
drafts can be rolled back.

The history window is ``L = B + max_spec_len`` (block ``B`` = ``buffer_len``): the
physical circular buffer is ``next_pow2(L)`` and ``max_cache_len`` passed to the
kernel is the logical ``L`` (the history block ``BC = next_pow2(B)`` is derived
inside the kernel).

GDN's standard decode kernel (``fused_recurrent_gated_delta_rule_replayssm``) and
the spec kernel SHARE the d/k/g cache layout (the spec one is circular via
``cache_base``; the standard one is linear). With ``cache_base=0`` they coincide,
so the oracle is built directly:
  * build the committed history by running the standard kernel for ``wp`` tokens
    (no flush, so the checkpoint stays fixed),
  * spec: one verify call over the window reading those very caches,
  * oracle: the standard kernel stepped one token at a time over the SAME window
    from the post-history checkpoint+caches.
For the multi-step rollback the ground truth is the baseline
``fused_recurrent_gated_delta_rule_packed_decode`` over the accepted token stream.

Precision: the spec kernel uses a chunked UT-transform whereas the standard /
packed kernels use the sequential recurrence, so even at fp32 they differ by
~1e-3 (an arithmetic-ordering difference, not a bug). bf16 adds the usual rounding
(rel ~1e-2), bounded per step and non-accumulating across the decode.
"""

import pytest
import torch

from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_recurrent_gated_delta_rule_replayssm,
)
from vllm.model_executor.layers.fla.ops.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_spec,
    gdn_replayssm_spec_decode,
    reset_gdn_replayssm_spec_cursors,
)
from vllm.utils.torch_utils import set_random_seed

DEV = "cuda"
PAD_SLOT_ID = -1


def _lbt(base_block: int, max_spec_len: int) -> tuple[int, int]:
    """Logical flush threshold L = B + max_spec_len and physical pow2 buffer."""
    L = base_block + max_spec_len
    buf = 1 << (L - 1).bit_length()
    return L, buf


def _output_tol(both_fp32: bool) -> tuple[float, float]:
    # Spec (chunked) vs standard/baseline (sequential): fp32 differs ~1.7e-3
    # by arithmetic ordering; bf16 adds rounding (~1e-2). atol is a small floor.
    if both_fp32:
        return 5e-3, 2e-3
    return 4e-2, 1e-2


def _state_tol(both_fp32: bool) -> tuple[float, float]:
    if both_fp32:
        return 5e-3, 3e-3
    return 4e-2, 2e-2


def _build_history(
    *,
    mqkv,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    state,
    d_cache,
    k_cache,
    g_cache,
    slot: int,
    wp: int,
):
    """Run the standard decode for the first ``wp`` tokens into ``slot`` (no
    flush, since wp <= B - max_spec_len < buf - 1), populating the shared caches
    and leaving the checkpoint unchanged."""
    sbi = torch.tensor([slot], device=DEV, dtype=torch.int32)
    for t in range(wp):
        ot = torch.empty(1, 1, *state.shape[1:3], device=DEV, dtype=mqkv.dtype)
        fused_recurrent_gated_delta_rule_replayssm(
            mixed_qkv=mqkv[t : t + 1],
            a=a[t : t + 1],
            b=b[t : t + 1],
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=state,
            d_cache=d_cache,
            k_cache=k_cache,
            g_cache=g_cache,
            out=ot,
            ssm_state_indices=sbi,
            write_pos=torch.tensor([t], device=DEV, dtype=torch.int32),
            use_qk_l2norm_in_kernel=True,
        )


def _standard_window_oracle(
    *,
    mqkv,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    state,
    d_cache,
    k_cache,
    g_cache,
    slot: int,
    wp: int,
    spec_len: int,
    HV: int,
    V: int,
) -> torch.Tensor:
    """Continue the standard decode over the window on CLONES of the post-history
    state/caches; return the window outputs (spec_len, HV, V)."""
    st = state.clone()
    d_o, k_o, g_o = d_cache.clone(), k_cache.clone(), g_cache.clone()
    sbi = torch.tensor([slot], device=DEV, dtype=torch.int32)
    win = []
    for s in range(spec_len):
        ot = torch.empty(1, 1, HV, V, device=DEV, dtype=mqkv.dtype)
        fused_recurrent_gated_delta_rule_replayssm(
            mixed_qkv=mqkv[wp + s : wp + s + 1],
            a=a[wp + s : wp + s + 1],
            b=b[wp + s : wp + s + 1],
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=st,
            d_cache=d_o,
            k_cache=k_o,
            g_cache=g_o,
            out=ot,
            ssm_state_indices=sbi,
            write_pos=torch.tensor([wp + s], device=DEV, dtype=torch.int32),
            use_qk_l2norm_in_kernel=True,
        )
        win.append(ot.reshape(1, HV, V).clone())
    return torch.cat(win, dim=0)


def _run_single_step(
    *,
    state_dtype,
    act_dtype,
    HQ,
    HV,
    K,
    V,
    buffer_len,  # history block B
    max_spec_len,
    wp,
    seed=0,
    perturb=False,
):
    set_random_seed(seed)
    H = HQ
    spec_len = max_spec_len
    scale = K**-0.5
    qkv_dim = 2 * H * K + HV * V
    L, buf = _lbt(buffer_len, max_spec_len)
    both_fp32 = state_dtype == torch.float32 and act_dtype == torch.float32
    rtol, atol = _output_tol(both_fp32)

    num_slots = 2
    slot = 1
    A_log = torch.randn(HV, device=DEV, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=DEV, dtype=torch.float32)

    T_tot = wp + spec_len
    mqkv = torch.randn(T_tot, qkv_dim, device=DEV, dtype=act_dtype)
    a = torch.randn(T_tot, HV, device=DEV, dtype=act_dtype)
    b = torch.randn(T_tot, HV, device=DEV, dtype=act_dtype)
    S0 = torch.randn(num_slots, HV, V, K, device=DEV, dtype=state_dtype) * 0.1

    d_cache = torch.zeros(num_slots, HV, buf, V, device=DEV, dtype=act_dtype)
    k_cache = torch.zeros(num_slots, H, buf, K, device=DEV, dtype=act_dtype)
    g_cache = torch.zeros(num_slots, HV, buf, device=DEV, dtype=torch.float32)

    state = S0.clone()
    _build_history(
        mqkv=mqkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        state=state,
        d_cache=d_cache,
        k_cache=k_cache,
        g_cache=g_cache,
        slot=slot,
        wp=wp,
    )
    # history build must not have touched the checkpoint (no flush).
    torch.testing.assert_close(state, S0, rtol=0, atol=0)

    oracle = _standard_window_oracle(
        mqkv=mqkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        state=state,
        d_cache=d_cache,
        k_cache=k_cache,
        g_cache=g_cache,
        slot=slot,
        wp=wp,
        spec_len=spec_len,
        HV=HV,
        V=V,
    )

    if perturb:
        # teeth: corrupt one cached history key -> outputs must diverge.
        k_cache[slot, 0, max(0, wp - 1), 0] += 5.0

    state_spec = state.clone()
    write_pos = torch.zeros(num_slots, dtype=torch.int32, device=DEV)
    write_pos[slot] = wp
    cache_base = torch.zeros(num_slots, dtype=torch.int32, device=DEV)
    is_flush = torch.zeros(num_slots, dtype=torch.int8, device=DEV)
    qsl = torch.tensor([0, spec_len], device=DEV, dtype=torch.int32)
    sbi = torch.tensor([slot], device=DEV, dtype=torch.int32)
    out_spec = torch.empty(spec_len, HV, V, device=DEV, dtype=act_dtype)
    gdn_replayssm_spec_decode(
        mixed_qkv=mqkv[wp:],
        a=a[wp:],
        b=b[wp:],
        A_log=A_log,
        dt_bias=dt_bias,
        checkpoint_state=state_spec,
        d_cache=d_cache,
        k_cache=k_cache,
        g_cache=g_cache,
        out=out_spec,
        query_start_loc=qsl,
        ssm_state_indices=sbi,
        write_pos=write_pos,
        cache_base=cache_base,
        is_flush=is_flush,
        max_cache_len=L,
        max_spec_len=max_spec_len,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(out_spec, oracle, rtol=rtol, atol=atol)
    # non-flush verify must not touch the checkpoint.
    torch.testing.assert_close(state_spec, S0, rtol=0, atol=0)


def _wp_set(base_block: int, max_spec_len: int) -> list[int]:
    # Verify-path fills up to the max non-flush fill C = B - max_spec_len.
    C = base_block - max_spec_len
    return sorted({0, max(0, C // 2), max(0, C)})


_PRECISIONS = [
    pytest.param((torch.float32, torch.float32), id="s32_a32"),
    pytest.param((torch.float32, torch.bfloat16), id="s32_a16"),
    pytest.param((torch.bfloat16, torch.bfloat16), id="s16_a16"),
]
# (num_q_heads, num_v_heads, head_k_dim, head_v_dim). The full sweep runs on the
# small shape; the production Qwen3.5 GDN shapes are exercised by
# test_spec_step_real_geometry.
_SMALL = pytest.param((2, 4, 64, 64), id="small")
_REAL_GEOMETRIES = [
    pytest.param((16, 32, 128, 128), id="qwen4b"),
    pytest.param((16, 64, 128, 128), id="qwen122b"),
]
_BASE_BLOCKS = [16, 32]  # history block B (replayssm_buffer_len)
_MAX_SPEC_LENS = [2, 4, 6, 8]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize("geometry", [_SMALL])
@pytest.mark.parametrize("base_block", _BASE_BLOCKS)
@pytest.mark.parametrize("max_spec_len", _MAX_SPEC_LENS)
def test_spec_step_matches_standard_decode(
    precision, geometry, base_block, max_spec_len
):
    state_dtype, act_dtype = precision
    HQ, HV, K, V = geometry
    for wp in _wp_set(base_block, max_spec_len):
        _run_single_step(
            state_dtype=state_dtype,
            act_dtype=act_dtype,
            HQ=HQ,
            HV=HV,
            K=K,
            V=V,
            buffer_len=base_block,
            max_spec_len=max_spec_len,
            wp=wp,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize("geometry", _REAL_GEOMETRIES)
@pytest.mark.parametrize("max_spec_len", [2, 8])
def test_spec_step_real_geometry(precision, geometry, max_spec_len):
    # Production Qwen3.5 GDN shapes (4B v_heads=32 / 122B-A10B v_heads=64) at the
    # deployable block B=16, both ends of the max_spec_len range.
    state_dtype, act_dtype = precision
    HQ, HV, K, V = geometry
    for wp in _wp_set(16, max_spec_len):
        _run_single_step(
            state_dtype=state_dtype,
            act_dtype=act_dtype,
            HQ=HQ,
            HV=HV,
            K=K,
            V=V,
            buffer_len=16,
            max_spec_len=max_spec_len,
            wp=wp,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_spec_step_teeth():
    _run_single_step(
        state_dtype=torch.float32,
        act_dtype=torch.float32,
        HQ=2,
        HV=4,
        K=64,
        V=64,
        buffer_len=16,
        max_spec_len=4,
        wp=5,
    )
    with pytest.raises(AssertionError):
        _run_single_step(
            state_dtype=torch.float32,
            act_dtype=torch.float32,
            HQ=2,
            HV=4,
            K=64,
            V=64,
            buffer_len=16,
            max_spec_len=4,
            wp=5,
            perturb=True,
        )


def _run_rollback(
    *,
    state_dtype,
    act_dtype,
    HQ,
    HV,
    K,
    V,
    buffer_len,  # history block B
    max_spec_len,
    num_steps=40,
    seed=0,
):
    """Drive the verify+commit loop from an empty buffer; accept a random k each
    step. Baseline packed decode of the accepted stream is the output ground
    truth; the committed checkpoint at each flush must match the baseline state at
    the matching folded-token count (n_folded bookkeeping)."""
    set_random_seed(seed)
    H = HQ
    scale = K**-0.5
    qkv_dim = 2 * H * K + HV * V
    L, buf = _lbt(buffer_len, max_spec_len)
    both_fp32 = state_dtype == torch.float32 and act_dtype == torch.float32
    o_rtol, o_atol = _output_tol(both_fp32)
    s_rtol, s_atol = _state_tol(both_fp32)

    num_slots = 2
    slot = 1
    A_log = torch.randn(HV, device=DEV, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=DEV, dtype=torch.float32)
    S0 = torch.randn(num_slots, HV, V, K, device=DEV, dtype=state_dtype) * 0.1
    state_spec = S0.clone()
    state_base = S0.clone()  # full pool; row in slot 1

    d_cache = torch.zeros(num_slots, HV, buf, V, device=DEV, dtype=act_dtype)
    k_cache = torch.zeros(num_slots, H, buf, K, device=DEV, dtype=act_dtype)
    g_cache = torch.zeros(num_slots, HV, buf, device=DEV, dtype=torch.float32)

    write_pos = torch.zeros(num_slots, dtype=torch.int32, device=DEV)
    cache_base = torch.zeros(num_slots, dtype=torch.int32, device=DEV)
    is_flush = torch.zeros(num_slots, dtype=torch.int8, device=DEV)
    sbi = torch.tensor([slot], device=DEV, dtype=torch.int32)
    reset_gdn_replayssm_spec_cursors(
        write_pos,
        cache_base,
        is_flush,
        torch.ones(1, dtype=torch.int32, device=DEV),
        sbi,
        L,
        max_spec_len,
    )

    n_folded = 0
    snapshots = {0: state_base[slot].clone()}
    total_accepted = 0
    g = torch.Generator(device="cpu").manual_seed(seed + 1)

    for _ in range(num_steps):
        spec_len = max_spec_len
        mqkv = torch.randn(spec_len, qkv_dim, device=DEV, dtype=act_dtype)
        a = torch.randn(spec_len, HV, device=DEV, dtype=act_dtype)
        b = torch.randn(spec_len, HV, device=DEV, dtype=act_dtype)
        qsl = torch.tensor([0, spec_len], device=DEV, dtype=torch.int32)
        out_spec = torch.empty(spec_len, HV, V, device=DEV, dtype=act_dtype)

        wp_before = int(write_pos[slot].item())
        flush_before = int(is_flush[slot].item())
        gdn_replayssm_spec_decode(
            mixed_qkv=mqkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            checkpoint_state=state_spec,
            d_cache=d_cache,
            k_cache=k_cache,
            g_cache=g_cache,
            out=out_spec,
            query_start_loc=qsl,
            ssm_state_indices=sbi,
            write_pos=write_pos,
            cache_base=cache_base,
            is_flush=is_flush,
            max_cache_len=L,
            max_spec_len=max_spec_len,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )
        if flush_before and wp_before > 0:
            n_folded += wp_before

        k = int(torch.randint(1, spec_len + 1, (1,), generator=g).item())
        base_out = []
        for s in range(k):
            ot = torch.empty(1, 1, HV, V, device=DEV, dtype=act_dtype)
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=mqkv[s : s + 1],
                a=a[s : s + 1],
                b=b[s : s + 1],
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                initial_state=state_base,
                out=ot,
                ssm_state_indices=sbi,
                use_qk_l2norm_in_kernel=True,
            )
            base_out.append(ot.reshape(1, HV, V).clone())
            total_accepted += 1
            snapshots[total_accepted] = state_base[slot].clone()
        base_out = torch.cat(base_out, dim=0)

        # (a) accepted-position outputs track the baseline decode.
        torch.testing.assert_close(out_spec[:k], base_out, rtol=o_rtol, atol=o_atol)

        commit_gdn_replayssm_spec(
            write_pos,
            cache_base,
            is_flush,
            torch.tensor([k], device=DEV, dtype=torch.int32),
            sbi,
            L,
            max_spec_len,
        )

        # (b) committed checkpoint == baseline state at the folded-token count.
        if n_folded in snapshots:
            torch.testing.assert_close(
                state_spec[slot], snapshots[n_folded], rtol=s_rtol, atol=s_atol
            )

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
        HQ=2,
        HV=4,
        K=64,
        V=64,
        buffer_len=base_block,
        max_spec_len=max_spec_len,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize("with_padding", [False, True])
def test_spec_continuous_batching(precision, with_padding):
    """Sparse state slots via ssm_state_indices with PAD(-1) padding rows and a
    DIFFERENT buffer fill per row, in one verify call. Each active row is checked
    against its own standard-decode window oracle; padding rows are zeroed and
    unused state slots are left untouched."""
    state_dtype, act_dtype = precision
    set_random_seed(0)
    HQ, HV, K, V = 2, 4, 64, 64
    H = HQ
    scale = K**-0.5
    qkv_dim = 2 * H * K + HV * V
    base_block = 16
    max_spec_len = 4
    spec_len = max_spec_len
    L, buf = _lbt(base_block, max_spec_len)
    both_fp32 = state_dtype == torch.float32 and act_dtype == torch.float32
    rtol, atol = _output_tol(both_fp32)

    batch = 3
    padding = 2 if with_padding else 0
    padded_batch = batch + padding
    total_slots = 12
    C = base_block - max_spec_len
    wps = [0, C // 2, C]

    A_log = torch.randn(HV, device=DEV, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=DEV, dtype=torch.float32)
    state_indices = (torch.randperm(total_slots - 1, device=DEV)[:batch] + 1).to(
        torch.int32
    )
    sbi = torch.cat(
        [
            state_indices,
            torch.full((padding,), PAD_SLOT_ID, dtype=torch.int32, device=DEV),
        ]
    )
    unused = torch.ones(total_slots, dtype=torch.bool, device=DEV)
    unused[state_indices] = False

    S0 = torch.randn(total_slots, HV, V, K, device=DEV, dtype=state_dtype) * 0.1
    state_spec = S0.clone()
    d_cache = torch.zeros(total_slots, HV, buf, V, device=DEV, dtype=act_dtype)
    k_cache = torch.zeros(total_slots, H, buf, K, device=DEV, dtype=act_dtype)
    g_cache = torch.zeros(total_slots, HV, buf, device=DEV, dtype=torch.float32)

    write_pos = torch.zeros(total_slots, dtype=torch.int32, device=DEV)
    cache_base = torch.zeros(total_slots, dtype=torch.int32, device=DEV)
    is_flush = torch.zeros(total_slots, dtype=torch.int8, device=DEV)

    oracles = []
    mqkv_pack = torch.zeros(
        padded_batch * spec_len, qkv_dim, device=DEV, dtype=act_dtype
    )
    a_pack = torch.zeros(padded_batch * spec_len, HV, device=DEV, dtype=act_dtype)
    b_pack = torch.zeros(padded_batch * spec_len, HV, device=DEV, dtype=act_dtype)
    for r in range(batch):
        wp = wps[r]
        slot = int(state_indices[r].item())
        T_tot = wp + spec_len
        mqkv = torch.randn(T_tot, qkv_dim, device=DEV, dtype=act_dtype)
        a = torch.randn(T_tot, HV, device=DEV, dtype=act_dtype)
        b = torch.randn(T_tot, HV, device=DEV, dtype=act_dtype)
        _build_history(
            mqkv=mqkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            state=state_spec,
            d_cache=d_cache,
            k_cache=k_cache,
            g_cache=g_cache,
            slot=slot,
            wp=wp,
        )
        oracles.append(
            _standard_window_oracle(
                mqkv=mqkv,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                state=state_spec,
                d_cache=d_cache,
                k_cache=k_cache,
                g_cache=g_cache,
                slot=slot,
                wp=wp,
                spec_len=spec_len,
                HV=HV,
                V=V,
            )
        )
        write_pos[slot] = wp
        seg = slice(r * spec_len, (r + 1) * spec_len)
        mqkv_pack[seg] = mqkv[wp:]
        a_pack[seg] = a[wp:]
        b_pack[seg] = b[wp:]

    state_before = state_spec.clone()
    qsl = torch.arange(
        0, (padded_batch + 1) * spec_len, spec_len, device=DEV, dtype=torch.int32
    )
    out_spec = torch.full(
        (padded_batch * spec_len, HV, V), 42.0, device=DEV, dtype=act_dtype
    )
    gdn_replayssm_spec_decode(
        mixed_qkv=mqkv_pack,
        a=a_pack,
        b=b_pack,
        A_log=A_log,
        dt_bias=dt_bias,
        checkpoint_state=state_spec,
        d_cache=d_cache,
        k_cache=k_cache,
        g_cache=g_cache,
        out=out_spec,
        query_start_loc=qsl,
        ssm_state_indices=sbi,
        write_pos=write_pos,
        cache_base=cache_base,
        is_flush=is_flush,
        max_cache_len=L,
        max_spec_len=max_spec_len,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )

    for r in range(batch):
        seg = slice(r * spec_len, (r + 1) * spec_len)
        torch.testing.assert_close(out_spec[seg], oracles[r], rtol=rtol, atol=atol)
    if with_padding:
        pad = slice(batch * spec_len, padded_batch * spec_len)
        assert torch.equal(out_spec[pad], torch.zeros_like(out_spec[pad]))
    # non-flush verify leaves every state slot untouched.
    torch.testing.assert_close(state_spec, state_before, rtol=0, atol=0)
    assert torch.equal(state_spec[unused], S0[unused])
