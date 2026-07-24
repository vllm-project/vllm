# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for the flashinfer_ucache GDN spec-decode backend.

Covers what is NEW in the vLLM integration (the kernel's own numerics are
anchored by the kernel repo's fp32-reference gates: _ucache_check.py,
_ucache_flush_check.py, _ucache_g3_ring.py):

1. commit_gdn_ucache_hist semantics (flush restart folded into commit,
   first-decode reset, null-block masking).
2. Intra-step shared hist_len: with restart_hist_on_flush=False, N "layers"
   sharing one hist tensor within a step all see the same P (the bug the
   flag exists to prevent).
3. Protocol equivalence: the vLLM bookkeeping (builder-owned commit,
   restart_hist_on_flush=False) must produce bit-identical outputs and
   checkpoints to the kernel-repo protocol (hist += accepted each step,
   wrapper masked_fill_ restart) over many steps crossing several flushes.
4. Strided packed-qkv slices (production layout) match dense inputs.
5. Null-page rows only scribble the reserved page 0.

Run inside the vLLM container with:
  VLLM_GDN_UCACHE_MODULE=<abs path to gdn_decode_bf16_wy_ucache_flush.py>
"""

import os

import pytest
import torch

from vllm.model_executor.layers.fla.ops.gdn_ucache_spec import (
    UCACHE_W_RING,
    commit_gdn_ucache_hist,
    load_ucache_kernel_module,
    ucache_flush_min,
)

DEV = "cuda"
HK, HV, K, V = 16, 64, 128, 128  # qwen122b geometry


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
    k_cache = torch.zeros(
        num_blocks, HK, UCACHE_W_RING, K, device=DEV, dtype=torch.bfloat16
    )
    u_cache = torch.zeros(
        num_blocks, HV, UCACHE_W_RING, V, device=DEV, dtype=torch.bfloat16
    )
    g_cache = torch.zeros(
        num_blocks, HV, UCACHE_W_RING, device=DEV, dtype=torch.float32
    )
    return ckpt, k_cache, u_cache, g_cache


def _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt, sbi, kc, uc, gc, hist,
          T, restart):
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
        scale=K**-0.5,
        use_qk_l2norm_in_kernel=True,
        output=None,
        flush_min=ucache_flush_min(T),
        restart_hist_on_flush=restart,
    )


def test_commit_kernel_semantics():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    n_blocks = 32
    flush_min = ucache_flush_min(4)  # 13
    hist = torch.zeros(n_blocks, dtype=torch.int32, device=DEV)
    # rows -> blocks 3, 5, 7, 9 (block 0 = NULL among indices)
    sbi = torch.tensor([3, 5, 7, 0, 9], dtype=torch.int32, device=DEV)
    hist[3] = 6            # below threshold: 6 + acc
    hist[5] = 13           # at threshold: flushed last step -> acc
    hist[7] = 16           # max legal: flushed -> acc
    hist[9] = 2            # first-decode row: reset to 0 regardless
    hist[0] = 99           # null block must never be touched
    acc = torch.tensor([3, 2, 4, 1, 4], dtype=torch.int32, device=DEV)
    first_decode = torch.tensor([0, 0, 0, 0, 1], dtype=torch.int8, device=DEV)
    commit_gdn_ucache_hist(hist, acc, sbi, first_decode, flush_min=flush_min)
    torch.cuda.synchronize()
    assert hist[3].item() == 9      # 6 + 3
    assert hist[5].item() == 2      # flush restart + 2
    assert hist[7].item() == 4      # flush restart + 4
    assert hist[9].item() == 0      # first-decode reset
    assert hist[0].item() == 99     # null untouched
    assert (hist[1:] <= UCACHE_W_RING).all()  # non-null blocks only


@pytest.mark.parametrize("T", [4, 8])
def test_intra_step_shared_hist_and_flag(T):
    """Two 'layers' share one hist tensor in a step; with restart=False the
    second layer must see the same P and produce the same fold as the first
    (independent pools). With restart=True the second layer would see P=0."""
    mod = _kmod()
    A_log, dt_bias = _gating_params()
    B, n_blocks = 3, 8
    flush_min = ucache_flush_min(T)
    q, k, v, a, b = _rand_inputs(B, T, seed=23)
    sbi = torch.tensor([1, 4, 6], dtype=torch.int32, device=DEV)

    hist_master = torch.tensor(
        [flush_min, 5, flush_min + 1], dtype=torch.int32, device=DEV
    )
    # Two independent "layers" with identical pools and inputs.
    outs, ckpts, hists = [], [], []
    hist = hist_master.clone()
    # Pre-fill rings identically for both layers so P>0 rows have history.
    for layer in range(2):
        ckpt, kc, uc, gc = _pools(n_blocks, seed=31)
        # Prime the ring: run one step from P=0 (appends T entries), then
        # set hist to the master values for the step under test.
        hist0 = torch.zeros(B, dtype=torch.int32, device=DEV)
        _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt, sbi, kc, uc, gc,
              hist0, T, restart=False)
        hist_layer = hist if layer == 0 else hist  # SHARED tensor
        out = _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt, sbi, kc, uc,
                    gc, hist_layer, T, restart=False)
        torch.cuda.synchronize()
        outs.append(out.clone())
        ckpts.append(ckpt.clone())
        hists.append(hist_layer.clone())
    # hist untouched by both calls
    assert torch.equal(hists[0], hist_master)
    assert torch.equal(hists[1], hist_master)
    # both layers saw identical P -> identical outputs and identical folds
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(ckpts[0], ckpts[1])


@pytest.mark.parametrize("T", [4])
@pytest.mark.parametrize("nreq", [1, 3, 8])
def test_protocol_equivalence_multi_step(T, nreq):
    """vLLM bookkeeping (commit kernel + restart=False) vs kernel-repo
    bookkeeping (hist += accepted + wrapper restart) over 24 steps crossing
    several flush cycles: outputs and checkpoints must match bit-for-bit."""
    mod = _kmod()
    A_log, dt_bias = _gating_params()
    n_blocks = 16
    flush_min = ucache_flush_min(T)
    # permuted, non-trivial block assignment (block 0 reserved)
    perm = torch.randperm(n_blocks - 1)[:nreq] + 1
    sbi = perm.to(torch.int32).to(DEV)

    ckpt_a, kc_a, uc_a, gc_a = _pools(n_blocks, seed=41)
    ckpt_b, kc_b, uc_b, gc_b = _pools(n_blocks, seed=41)

    # Protocol A (vLLM): block-keyed master + commit kernel.
    hist_blocks = torch.zeros(n_blocks, dtype=torch.int32, device=DEV)
    # Protocol B (kernel repo): request-keyed hist, wrapper restart.
    hist_req = torch.zeros(nreq, dtype=torch.int32, device=DEV)

    gen = torch.Generator().manual_seed(97)
    prev_acc = torch.zeros(nreq, dtype=torch.int32, device=DEV)
    first = torch.zeros(nreq, dtype=torch.int8, device=DEV)
    for step in range(24):
        q, k, v, a, b = _rand_inputs(nreq, T, seed=1000 + step)
        # A: commit (prev step's acceptance), gather, call with restart=False
        commit_gdn_ucache_hist(
            hist_blocks, prev_acc, sbi, first if step == 0 else None,
            flush_min=flush_min,
        )
        gathered = hist_blocks.index_select(0, sbi.to(torch.int64)).contiguous()
        out_a = _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt_a, sbi,
                      kc_a, uc_a, gc_a, gathered, T, restart=False)
        # B: kernel-repo protocol on request-keyed hist
        out_b = _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt_b, sbi,
                      kc_b, uc_b, gc_b, hist_req, T, restart=True)
        torch.cuda.synchronize()
        assert torch.equal(out_a, out_b), f"outputs diverged at step {step}"
        assert torch.equal(ckpt_a, ckpt_b), f"checkpoints diverged at step {step}"
        assert (gathered <= UCACHE_W_RING).all()
        acc = torch.randint(1, T + 1, (nreq,), generator=gen).to(
            torch.int32
        ).to(DEV)
        prev_acc = acc
        hist_req += acc  # protocol B commit (wrapper already restarted)


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
        out = _call(mod, A_log, dt_bias, qq, kk, vv, a, b, ckpt, sbi,
                    kc, uc, gc, hist, T, restart=False)
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
    # prime rows 0/2 rings so their flush folds something
    hist0 = torch.zeros(B, dtype=torch.int32, device=DEV)
    _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt, sbi, kc, uc, gc, hist0,
          T, restart=False)
    snap_after_prime = ckpt.clone()
    _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt, sbi, kc, uc, gc, hist,
          T, restart=False)
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
    shapes = [(HV, V, K), (HK, UCACHE_W_RING, K), (HV, UCACHE_W_RING, V),
              (HV, UCACHE_W_RING)]
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

    hist_d = torch.zeros(B, dtype=torch.int32, device=DEV)
    hist_s = torch.zeros(B, dtype=torch.int32, device=DEV)
    gen = torch.Generator().manual_seed(3)
    prev_acc_d = torch.zeros(B, dtype=torch.int32, device=DEV)
    hist_blocks_d = torch.zeros(n_blocks, dtype=torch.int32, device=DEV)
    hist_blocks_s = torch.zeros(n_blocks, dtype=torch.int32, device=DEV)
    fm = ucache_flush_min(T)
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
        commit_gdn_ucache_hist(hist_blocks_d, prev_acc_d, sbi, None, flush_min=fm)
        commit_gdn_ucache_hist(hist_blocks_s, prev_acc_d, sbi, None, flush_min=fm)
        hd = hist_blocks_d.index_select(0, sbi.to(torch.int64)).contiguous()
        hs = hist_blocks_s.index_select(0, sbi.to(torch.int64)).contiguous()
        out_d = _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt_d, sbi,
                      kc_d, uc_d, gc_d, hd, T, restart=False)
        out_s = _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt_s, sbi,
                      kc_s, uc_s, gc_s, hs, T, restart=False)
        torch.cuda.synchronize()
        assert torch.equal(out_d, out_s), f"outputs diverged at step {step}"
        assert torch.equal(ckpt_d[1:], ckpt_s[1:]), f"ckpt diverged at step {step}"
        assert torch.equal(kc_d[1:], kc_s[1:]) and torch.equal(uc_d[1:], uc_s[1:])
        prev_acc_d = torch.randint(1, T + 1, (B,), generator=gen).to(
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

    hist_blocks = torch.zeros(n_blocks, dtype=torch.int32, device=DEV)
    prev_acc = torch.zeros(B, dtype=torch.int32, device=DEV)
    fm = ucache_flush_min(T)
    gen = torch.Generator().manual_seed(5)
    hist_blocks_d = hist_blocks.clone()
    for step in range(8):  # crosses a flush cycle
        qd, kd, vd, a, b = _rand_inputs(B, T, seed=800 + step)
        packed = torch.cat(
            [qd.flatten(2), kd.flatten(2), vd.flatten(2)], dim=-1
        ).contiguous()
        q = packed[..., : HK * K].unflatten(-1, (HK, K))
        k = packed[..., HK * K : 2 * HK * K].unflatten(-1, (HK, K))
        v = packed[..., 2 * HK * K :].unflatten(-1, (HV, V))
        commit_gdn_ucache_hist(hist_blocks, prev_acc, sbi, None, flush_min=fm)
        commit_gdn_ucache_hist(hist_blocks_d, prev_acc, sbi, None, flush_min=fm)
        hs = hist_blocks.index_select(0, sbi.to(torch.int64)).contiguous()
        hd = hist_blocks_d.index_select(0, sbi.to(torch.int64)).contiguous()
        out_s = _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt_s, sbi,
                      kc_s, uc_s, gc_s, hs, T, restart=False)
        out_d = _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt_d, sbi,
                      kc_d, uc_d, gc_d, hd, T, restart=False)
        torch.cuda.synchronize()
        assert torch.equal(out_s, out_d), f"outputs diverged at step {step}"
        assert torch.equal(ckpt_s[1900:], ckpt_d[1900:]), f"high-block ckpt diverged at {step}"
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

    ckpt1, kc1, uc1, gc1 = _pools(n_blocks, seed=103)
    ckpt2, kc2, uc2, gc2 = _pools(n_blocks, seed=103)
    page0_snap = (ckpt1[0].clone(), kc1[0].clone(), uc1[0].clone(),
                  gc1[0].clone())

    out1 = _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt1, sbi_neg,
                 kc1, uc1, gc1, hist.clone(), T, restart=False)
    out2 = _call(mod, A_log, dt_bias, q, k, v, a, b, ckpt2, sbi_nul,
                 kc2, uc2, gc2, hist.clone(), T, restart=False)
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
