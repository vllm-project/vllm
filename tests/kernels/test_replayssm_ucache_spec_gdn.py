# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for the flashinfer_ucache GDN spec-decode backend.

Covers what is NEW in the vLLM integration (the kernel's own numerics are
anchored by the kernel repo's fp32-reference suite, tests/gdn/):

1. Shared Triton cursor semantics (commit_gdn_replayssm_spec /
   reset_gdn_replayssm_spec_cursors driving the ucache kernel): flush rows
   slide cache_base past the folded window mod RING_SLOTS, first-decode
   reset clears BOTH cursors, null-block rows are never touched.
2. Intra-step shared cursors: N "layers" reading one gathered
   (hist_len, cache_base) pair within a step all see the same window (the
   kernel treats cursors as read-only).
3. Protocol equivalence: the vLLM bookkeeping (block-keyed cursors +
   commit_gdn_replayssm_spec + gather, restart_hist_on_flush=False) must
   produce bit-identical outputs, checkpoints, and window origins to the
   kernel-repo standalone protocol (request-keyed cursors, wrapper commit
   via restart_hist_on_flush=True) over many steps crossing several
   flushes and ring wrap-arounds.
4. Strided packed-qkv slices (production layout) match dense inputs.
5. Null-page rows only scribble the reserved page 0.
6. Block-strided (vLLM paged) pools match dense pools, incl. past 2^31
   elements (64-bit addressing).
7. Padded rows (negative sentinel) retire their CTAs at kernel entry.

Run inside the vLLM container with:
  VLLM_GDN_UCACHE_MODULE=<abs path to gdn_decode_bf16_wy_ucache_flush.py>
"""

import os

import pytest
import torch

# This suite allocates bf16 checkpoint/ring pools; pin the kernel module's
# env-selected dtypes to match BEFORE it loads (the adapter defaults both to
# fp16 for serving). Must run before the first load_ucache_kernel_module().
os.environ["GDN_UCACHE_STATE_DTYPE"] = "bf16"
os.environ["GDN_UCACHE_RING_DTYPE"] = "bf16"

from vllm.model_executor.layers.fla.ops.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_spec,
    reset_gdn_replayssm_spec_cursors,
)
from vllm.model_executor.layers.fla.ops.gdn_ucache_spec import (
    UCACHE_RING_SLOTS,
    UCACHE_W_RING,
    load_ucache_kernel_module,
    ucache_flush_min,
)

DEV = "cuda"
HK, HV, K, V = 16, 64, 128, 128  # qwen122b geometry
RING = UCACHE_RING_SLOTS  # 32


def _kmod():
    if not torch.cuda.is_available():
        return None
    try:
        return load_ucache_kernel_module(strided_qkv=True)
    except Exception:
        return None


pytestmark = pytest.mark.skipif(
    _kmod() is None,
    reason="CUDA + ucache kernel module required "
    "(set VLLM_GDN_UCACHE_MODULE=<path to gdn_decode_bf16_wy_ucache_flush.py>)",
)


def _gating_params():
    torch.manual_seed(7)
    A_log = torch.randn(HV, device=DEV, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(HV, device=DEV, dtype=torch.float32) * 0.1
    return A_log, dt_bias


def _rand_inputs(B, T, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    mk = lambda *s: torch.randn(*s, generator=g, device=DEV, dtype=torch.bfloat16)
    q = mk(B, T, HK, K)
    k = mk(B, T, HK, K)
    v = mk(B, T, HV, V) * 0.5
    a = mk(B, T, HV) * 0.5
    b = mk(B, T, HV)
    return q, k, v, a, b


def _pools(num_blocks, seed=11):
    g = torch.Generator(device=DEV).manual_seed(seed)
    ckpt = (
        torch.randn(
            num_blocks, HV, V, K, generator=g, device=DEV, dtype=torch.float32
        )
        * 0.05
    ).to(torch.bfloat16)
    k_cache = torch.zeros(num_blocks, HK, RING, K, device=DEV, dtype=torch.bfloat16)
    u_cache = torch.zeros(num_blocks, HV, RING, V, device=DEV, dtype=torch.bfloat16)
    g_cache = torch.zeros(num_blocks, HV, RING, device=DEV, dtype=torch.float32)
    return ckpt, k_cache, u_cache, g_cache


def _cursors(n_blocks):
    """Block-keyed cursor triple, exactly as the metadata builder allocates."""
    wp = torch.zeros(n_blocks, dtype=torch.int32, device=DEV)
    cb = torch.zeros(n_blocks, dtype=torch.int32, device=DEV)
    fl = torch.zeros(n_blocks, dtype=torch.int8, device=DEV)
    return wp, cb, fl


def _commit(wp, cb, fl, acc, sbi, T):
    """The builder's per-step commit with its exact parameters
    (L = buffer_len + T = W_RING + T, physical ring = next_pow2(L) = 32)."""
    commit_gdn_replayssm_spec(
        wp, cb, fl, acc, sbi,
        max_cache_len=UCACHE_W_RING + T,
        max_spec_len=T,
        cache_buf_len=RING,
    )


def _gather(t, sbi):
    return t.index_select(0, sbi.to(torch.int64)).contiguous()


def _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt, sbi, kc, uc, gc, hist,
          base, T, restart):
    return mod.gated_delta_rule_mtp_ucache_flush(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=ckpt,
        initial_state_indices=sbi,
        k_cache=kc,
        u_cache=uc,
        g_cache=gc,
        hist_len=hist,
        cache_base=base,
        scale=K**-0.5,
        use_qk_l2norm_in_kernel=True,
        output=None,
        flush_min=ucache_flush_min(T),
        restart_hist_on_flush=restart,
    )


def test_shared_cursor_commit_semantics():
    """The shared Triton commit drives the ucache backend: flushed rows
    (is_flush armed == the kernel just folded, wp >= flush_min) slide
    cache_base past the folded window mod 32 and restart wp at the accepted
    count; verify rows just grow wp; first-decode reset clears BOTH cursors;
    the null block is never touched."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    T = 4
    n_blocks = 32
    wp, cb, fl = _cursors(n_blocks)
    # rows -> blocks 3, 5, 7, 9 (+ a null row on block 0)
    sbi = torch.tensor([3, 5, 7, 9, 0], dtype=torch.int32, device=DEV)
    wp[3], fl[3] = 6, 0            # verify row: 6 + acc
    wp[5], fl[5] = 13, 1           # flushed at threshold: base += 13, wp = acc
    wp[7], cb[7], fl[7] = 16, 20, 1  # flushed at max, wrapping base: (20+16)&31
    wp[9], fl[9] = 12, 0           # verify row landing ON the arm point
    wp[0] = 99                     # null block must never be touched
    acc = torch.tensor([3, 2, 4, 1, 4], dtype=torch.int32, device=DEV)
    _commit(wp, cb, fl, acc, sbi, T)
    torch.cuda.synchronize()
    assert wp[3].item() == 9 and cb[3].item() == 0 and fl[3].item() == 0
    assert wp[5].item() == 2 and cb[5].item() == 13 and fl[5].item() == 0
    assert wp[7].item() == 4 and cb[7].item() == (20 + 16) % RING  # == 4
    assert fl[7].item() == 0
    # wp 12+1=13 == flush_min(4): is_flush arms for the NEXT step
    assert wp[9].item() == 13 and fl[9].item() == 1
    assert wp[0].item() == 99 and cb[0].item() == 0  # null untouched
    assert (wp[1:] <= UCACHE_W_RING).all()
    assert (cb >= 0).all() and (cb < RING).all()

    # first-decode reset (prefill->decode handoff) clears BOTH cursors
    do_reset = torch.tensor([0, 0, 0, 1, 0], dtype=torch.int8, device=DEV)
    reset_gdn_replayssm_spec_cursors(
        wp, cb, fl, do_reset, sbi,
        max_cache_len=UCACHE_W_RING + T, max_spec_len=T,
    )
    torch.cuda.synchronize()
    assert wp[9].item() == 0 and cb[9].item() == 0 and fl[9].item() == 0
    assert wp[5].item() == 2 and cb[5].item() == 13  # others untouched


@pytest.mark.parametrize("T", [4, 8])
def test_intra_step_shared_cursors(T):
    """Two 'layers' share one gathered (hist, base) pair in a step; the
    kernel treats cursors as read-only, so both layers must see the same
    window and produce identical outputs and folds (independent pools)."""
    mod = _kmod()
    A_log, dt_bias = _gating_params()
    B, n_blocks = 3, 8
    fm = ucache_flush_min(T)
    q, k, v, a, b = _rand_inputs(B, T, seed=23)
    sbi = torch.tensor([1, 4, 6], dtype=torch.int32, device=DEV)

    hist_master = torch.tensor([fm, 5, fm + 1], dtype=torch.int32, device=DEV)
    base_master = torch.tensor([0, 28, 5], dtype=torch.int32, device=DEV)
    hist = hist_master.clone()
    base = base_master.clone()
    outs, ckpts = [], []
    for _layer in range(2):
        ckpt, kc, uc, gc = _pools(n_blocks, seed=31)
        # Prime the ring (appends T entries from P=0), then run the step
        # under test with the SHARED master cursors.
        hist0 = torch.zeros(B, dtype=torch.int32, device=DEV)
        base0 = torch.zeros(B, dtype=torch.int32, device=DEV)
        _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt, sbi, kc, uc, gc,
              hist0, base0, T, restart=False)
        out = _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt, sbi, kc, uc,
                    gc, hist, base, T, restart=False)
        torch.cuda.synchronize()
        outs.append(out.clone())
        ckpts.append(ckpt.clone())
    # cursors untouched by both calls (restart=False -> kernel read-only)
    assert torch.equal(hist, hist_master)
    assert torch.equal(base, base_master)
    # both layers saw identical windows -> identical outputs and folds
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(ckpts[0], ckpts[1])


@pytest.mark.parametrize("T", [4])
@pytest.mark.parametrize("nreq", [1, 3, 8])
def test_protocol_equivalence_multi_step(T, nreq):
    """vLLM bookkeeping (block-keyed cursors + shared Triton commit +
    gather, restart=False) vs kernel-repo standalone bookkeeping
    (request-keyed cursors, wrapper commit via restart=True) over 24 steps
    crossing several flush cycles and ring wraps: outputs, checkpoints, and
    window origins must match bit-for-bit."""
    mod = _kmod()
    A_log, dt_bias = _gating_params()
    n_blocks = 16
    # permuted, non-trivial block assignment (block 0 reserved)
    perm = torch.randperm(n_blocks - 1, generator=torch.Generator().manual_seed(13))
    sbi = (perm[:nreq] + 1).to(torch.int32).to(DEV)

    ckpt_a, kc_a, uc_a, gc_a = _pools(n_blocks, seed=41)
    ckpt_b, kc_b, uc_b, gc_b = _pools(n_blocks, seed=41)

    # Protocol A (vLLM): block-keyed cursors + shared Triton commit.
    wp, cb, fl = _cursors(n_blocks)
    # Protocol B (kernel repo): request-keyed cursors, wrapper commit.
    hist_req = torch.zeros(nreq, dtype=torch.int32, device=DEV)
    base_req = torch.zeros(nreq, dtype=torch.int32, device=DEV)

    gen = torch.Generator().manual_seed(97)
    prev_acc = torch.zeros(nreq, dtype=torch.int32, device=DEV)
    for step in range(24):
        q, k, v, a, b = _rand_inputs(nreq, T, seed=1000 + step)
        # A: commit the previous step's acceptance, gather, run restart=False
        _commit(wp, cb, fl, prev_acc, sbi, T)
        hd = _gather(wp, sbi)
        bd = _gather(cb, sbi)
        out_a = _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt_a, sbi,
                      kc_a, uc_a, gc_a, hd, bd, T, restart=False)
        # B: kernel-repo protocol; the wrapper commits flushed rows itself
        # DURING the call (A's builder slides at the NEXT step's commit), so
        # the phase-aligned comparison point is B's PRE-call cursors.
        hist_b_pre = hist_req.clone()
        base_b_pre = base_req.clone()
        out_b = _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt_b, sbi,
                      kc_b, uc_b, gc_b, hist_req, base_req, T, restart=True)
        torch.cuda.synchronize()
        assert torch.equal(out_a, out_b), f"outputs diverged at step {step}"
        assert torch.equal(ckpt_a, ckpt_b), f"checkpoints diverged at step {step}"
        assert torch.equal(hd, hist_b_pre), f"windows diverged at step {step}"
        assert torch.equal(bd, base_b_pre), f"window origins diverged at step {step}"
        assert (hd <= UCACHE_W_RING).all()
        assert (bd >= 0).all() and (bd < RING).all()
        acc = torch.randint(1, T + 1, (nreq,), generator=gen).to(
            torch.int32
        ).to(DEV)
        prev_acc = acc
        hist_req += acc  # protocol B commit (wrapper already slid the base)


def test_strided_packed_qkv_matches_dense():
    """Production layout: q/k/v as last-dim slices of one packed row must
    match dense contiguous copies of the same data."""
    mod = _kmod()
    A_log, dt_bias = _gating_params()
    B, T, n_blocks = 4, 4, 8
    q, k, v, a, b = _rand_inputs(B, T, seed=53)
    packed = torch.cat(
        [q.flatten(2), k.flatten(2), v.flatten(2)], dim=-1
    ).contiguous()  # [B, T, 2*HK*K + HV*V], token stride shared by slices
    qs = packed[..., : HK * K].unflatten(-1, (HK, K))
    ks = packed[..., HK * K : 2 * HK * K].unflatten(-1, (HK, K))
    vs = packed[..., 2 * HK * K :].unflatten(-1, (HV, V))
    sbi = torch.arange(1, B + 1, dtype=torch.int32, device=DEV)

    outs = []
    for (qq, kk, vv) in [(q, k, v), (qs, ks, vs)]:
        ckpt, kc, uc, gc = _pools(n_blocks, seed=61)
        hist = torch.zeros(B, dtype=torch.int32, device=DEV)
        base = torch.zeros(B, dtype=torch.int32, device=DEV)
        out = _call(mod, A_log, dt_bias, qq, kk, vv, a, b, ckpt, sbi,
                    kc, uc, gc, hist, base, T, restart=False)
        torch.cuda.synchronize()
        outs.append(out.clone())
    assert torch.equal(outs[0], outs[1])


def test_null_page_rows_only_touch_page_zero():
    mod = _kmod()
    A_log, dt_bias = _gating_params()
    B, T, n_blocks = 3, 4, 8
    q, k, v, a, b = _rand_inputs(B, T, seed=71)
    # row 1 -> null page 0; rows 0/2 -> real pages
    sbi = torch.tensor([2, 0, 5], dtype=torch.int32, device=DEV)
    ckpt, kc, uc, gc = _pools(n_blocks, seed=79)
    snap = ckpt.clone()
    hist = torch.tensor([13, 0, 13], dtype=torch.int32, device=DEV)
    base = torch.zeros(B, dtype=torch.int32, device=DEV)
    # prime rows 0/2 rings so their flush folds something
    hist0 = torch.zeros(B, dtype=torch.int32, device=DEV)
    base0 = torch.zeros(B, dtype=torch.int32, device=DEV)
    _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt, sbi, kc, uc, gc, hist0,
          base0, T, restart=False)
    snap_after_prime = ckpt.clone()
    _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt, sbi, kc, uc, gc, hist,
          base, T, restart=False)
    torch.cuda.synchronize()
    # pages not referenced by any row are byte-stable
    untouched = [i for i in range(n_blocks) if i not in (0, 2, 5)]
    for i in untouched:
        assert torch.equal(ckpt[i], snap[i]), f"page {i} was modified"
    # flushed real rows actually changed their pages
    assert not torch.equal(ckpt[2], snap_after_prime[2])
    assert not torch.equal(ckpt[5], snap_after_prime[5])


def _vllm_style_strided_pools(num_blocks, seed=11):
    """Carve (ckpt, k, u, g) as block-strided views of one page-major backing,
    exactly like vLLM's _reshape_kv_cache_tensors (inner dims dense, dim-0
    stride = whole page)."""
    shapes = [(HV, V, K), (HK, RING, K), (HV, RING, V), (HV, RING)]
    dtypes = [torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.float32]
    page_bytes = sum(
        int(torch.empty(s, device="meta").numel()) * t.itemsize
        for s, t in zip(shapes, dtypes)
    )
    raw = torch.zeros(num_blocks * page_bytes, dtype=torch.int8, device=DEV)
    out, off = [], 0
    for s, t in zip(shapes, dtypes):
        n = int(torch.empty(s, device="meta").numel())
        view = torch.as_strided(
            raw.view(t),
            size=(num_blocks, *s),
            stride=(page_bytes // t.itemsize,
                    *torch.empty(s, device="meta").stride()),
            storage_offset=off // t.itemsize,
        )
        out.append(view)
        off += n * t.itemsize
    ckpt, kc, uc, gc = out
    g = torch.Generator(device=DEV).manual_seed(seed)
    ckpt.copy_((torch.randn(ckpt.shape, generator=g, device=DEV,
                            dtype=torch.float32) * 0.05).to(torch.bfloat16))
    return ckpt, kc, uc, gc


def test_block_strided_pools_match_dense():
    """vLLM paged layout: block-strided pool views must produce bit-identical
    outputs, ring contents, and folds to dense contiguous pools."""
    mod = _kmod()
    A_log, dt_bias = _gating_params()
    B, T, n_blocks = 4, 4, 6
    sbi = torch.tensor([1, 3, 4, 5], dtype=torch.int32, device=DEV)

    # dense reference pools with identical initial checkpoint values
    ckpt_d, kc_d, uc_d, gc_d = _pools(n_blocks, seed=91)
    ckpt_s, kc_s, uc_s, gc_s = _vllm_style_strided_pools(n_blocks, seed=91)
    ckpt_s.copy_(ckpt_d)
    assert not ckpt_s.is_contiguous() and not kc_s.is_contiguous()

    wp_d, cb_d, fl_d = _cursors(n_blocks)
    wp_s, cb_s, fl_s = _cursors(n_blocks)
    gen = torch.Generator().manual_seed(3)
    prev_acc = torch.zeros(B, dtype=torch.int32, device=DEV)
    for step in range(12):  # crosses >= 2 flush cycles
        qd, kd, vd, a, b = _rand_inputs(B, T, seed=500 + step)
        # packed strided q/k/v (production layout) so the wrapper's
        # static-descriptor mode engages — required for strided pools
        packed = torch.cat(
            [qd.flatten(2), kd.flatten(2), vd.flatten(2)], dim=-1
        ).contiguous()
        q = packed[..., : HK * K].unflatten(-1, (HK, K))
        k = packed[..., HK * K : 2 * HK * K].unflatten(-1, (HK, K))
        v = packed[..., 2 * HK * K :].unflatten(-1, (HV, V))
        _commit(wp_d, cb_d, fl_d, prev_acc, sbi, T)
        _commit(wp_s, cb_s, fl_s, prev_acc, sbi, T)
        hd, bd = _gather(wp_d, sbi), _gather(cb_d, sbi)
        hs, bs = _gather(wp_s, sbi), _gather(cb_s, sbi)
        out_d = _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt_d, sbi,
                      kc_d, uc_d, gc_d, hd, bd, T, restart=False)
        out_s = _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt_s, sbi,
                      kc_s, uc_s, gc_s, hs, bs, T, restart=False)
        torch.cuda.synchronize()
        assert torch.equal(out_d, out_s), f"outputs diverged at step {step}"
        assert torch.equal(ckpt_d[1:], ckpt_s[1:]), f"ckpt diverged at step {step}"
        assert torch.equal(kc_d[1:], kc_s[1:]) and torch.equal(uc_d[1:], uc_s[1:])
        prev_acc = torch.randint(1, T + 1, (B,), generator=gen).to(
            torch.int32
        ).to(DEV)


def test_block_strided_pools_past_2gb():
    """64-bit pool addressing: rows on high blocks whose element offsets
    exceed 2^31 must match dense-pool results (page ~4.7MB -> block ~950
    sits past the old 32-bit ceiling)."""
    if torch.cuda.get_device_properties(0).total_memory < 30 * 2**30:
        pytest.skip("needs ~10GB free GPU memory")
    mod = _kmod()
    A_log, dt_bias = _gating_params()
    B, T, n_blocks = 3, 4, 2000
    sbi = torch.tensor([1951, 1975, 1999], dtype=torch.int32, device=DEV)

    ckpt_d, kc_d, uc_d, gc_d = _pools(n_blocks, seed=101)
    ckpt_s, kc_s, uc_s, gc_s = _vllm_style_strided_pools(n_blocks, seed=101)
    ckpt_s.copy_(ckpt_d)
    # sanity: the high blocks really are past 2^31 elements in the backing
    assert 1999 * ckpt_s.stride(0) > 2**31, (
        f"test page too small to cross 2^31: stride0={ckpt_s.stride(0)}")

    wp_d, cb_d, fl_d = _cursors(n_blocks)
    wp_s, cb_s, fl_s = _cursors(n_blocks)
    prev_acc = torch.zeros(B, dtype=torch.int32, device=DEV)
    gen = torch.Generator().manual_seed(5)
    for step in range(8):  # crosses a flush cycle
        qd, kd, vd, a, b = _rand_inputs(B, T, seed=800 + step)
        packed = torch.cat(
            [qd.flatten(2), kd.flatten(2), vd.flatten(2)], dim=-1
        ).contiguous()
        q = packed[..., : HK * K].unflatten(-1, (HK, K))
        k = packed[..., HK * K : 2 * HK * K].unflatten(-1, (HK, K))
        v = packed[..., 2 * HK * K :].unflatten(-1, (HV, V))
        _commit(wp_d, cb_d, fl_d, prev_acc, sbi, T)
        _commit(wp_s, cb_s, fl_s, prev_acc, sbi, T)
        hd, bd = _gather(wp_d, sbi), _gather(cb_d, sbi)
        hs, bs = _gather(wp_s, sbi), _gather(cb_s, sbi)
        out_s = _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt_s, sbi,
                      kc_s, uc_s, gc_s, hs, bs, T, restart=False)
        out_d = _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt_d, sbi,
                      kc_d, uc_d, gc_d, hd, bd, T, restart=False)
        torch.cuda.synchronize()
        assert torch.equal(out_s, out_d), f"outputs diverged at step {step}"
        assert torch.equal(ckpt_s[1900:], ckpt_d[1900:]), (
            f"high-block ckpt diverged at {step}")
        prev_acc = torch.randint(1, T + 1, (B,), generator=gen).to(
            torch.int32
        ).to(DEV)


def test_pad_skip_negative_rows_exit_early():
    """Padded rows (sentinel idx < 0) retire their CTAs at kernel entry.

    (a) real-row outputs and pages are bit-identical whether pad rows carry
        -1 (pad-skip) or the legacy null-page 0 (P=0 verify) -- pad rows can
        never influence real rows;
    (b) with -1, pad rows write NOTHING: page 0 (ckpt and all three rings)
        stays byte-stable, unlike legacy 0-padding which scribbles it.
    """
    mod = _kmod()
    A_log, dt_bias = _gating_params()
    B, T, n_blocks = 6, 4, 8
    q, k, v, a, b = _rand_inputs(B, T, seed=101)
    # rows 0..2 real -> pages 2/5/7 (row 0 at flush threshold), rows 3..5 pad
    sbi_neg = torch.tensor([2, 5, 7, -1, -1, -1], dtype=torch.int32, device=DEV)
    sbi_nul = torch.tensor([2, 5, 7, 0, 0, 0], dtype=torch.int32, device=DEV)
    hist = torch.tensor([13, 7, 0, 0, 0, 0], dtype=torch.int32, device=DEV)
    base = torch.zeros(B, dtype=torch.int32, device=DEV)

    ckpt1, kc1, uc1, gc1 = _pools(n_blocks, seed=103)
    ckpt2, kc2, uc2, gc2 = _pools(n_blocks, seed=103)
    page0_snap = (ckpt1[0].clone(), kc1[0].clone(), uc1[0].clone(),
                  gc1[0].clone())

    out1 = _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt1, sbi_neg,
                 kc1, uc1, gc1, hist.clone(), base.clone(), T, restart=False)
    out2 = _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt2, sbi_nul,
                 kc2, uc2, gc2, hist.clone(), base.clone(), T, restart=False)
    torch.cuda.synchronize()

    # (a) real rows bit-identical across the two pad conventions. The wrapper
    # returns [B, T, HV, V]; slice BATCH rows (pad rows are uninitialized
    # torch.empty garbage in the pad-skip arm by design — never compare them).
    assert out1.shape[0] == B
    assert torch.equal(out1[:3], out2[:3])
    for p in (2, 5, 7):
        assert torch.equal(ckpt1[p], ckpt2[p]), f"real page {p} ckpt differs"
        assert torch.equal(kc1[p], kc2[p]), f"real page {p} k-ring differs"
        assert torch.equal(uc1[p], uc2[p]), f"real page {p} u-ring differs"
        assert torch.equal(gc1[p], gc2[p]), f"real page {p} g-ring differs"

    # (b) pad-skip rows leave page 0 byte-stable
    assert torch.equal(ckpt1[0], page0_snap[0]), "pad rows wrote ckpt page 0"
    assert torch.equal(kc1[0], page0_snap[1]), "pad rows wrote k-ring page 0"
    assert torch.equal(uc1[0], page0_snap[2]), "pad rows wrote u-ring page 0"
    assert torch.equal(gc1[0], page0_snap[3]), "pad rows wrote g-ring page 0"
    # ...and legacy 0-padding does scribble the null page (sanity that the
    # comparison above is meaningful)
    scribbled = (not torch.equal(kc2[0], page0_snap[1])
                 or not torch.equal(uc2[0], page0_snap[2])
                 or not torch.equal(gc2[0], page0_snap[3]))
    assert scribbled, "expected legacy 0-padding to write the null page"
