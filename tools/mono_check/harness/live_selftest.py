# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager-only self-tests of the live MonoKernel dispatch (debug harness; moved out of
``MonoLive``).

Attached on top of an installed ``MonoLive`` by ``attach(lv)`` (``live_worker_ext`` does
it after install) when ``LiveConfig.extra`` asks for them:

* ``indexer_selftest_steps: N`` (indexer_only mode): on the first N mono steps, after
  each indexer refresh, the indexer-only path vs vLLM's full attention (top-k rows,
  index-K cache). Records in ``lv.stats["indexer_selftest"]``.
* ``fused_parity_steps: N`` (fused indexer): on the first N mono steps, after each
  fused-layer launch, the in-kernel indexer's CSR vs vLLM's own indexer refresh +
  convert. Records in ``lv.stats["fused_parity"]``.

Both wrap instance methods (``_refresh_indexer`` / ``mono_forward``) so the production
class carries no test code; the call points and order are the ones the in-class versions
had."""

from __future__ import annotations

import torch

from vllm.models.deepseek_v32.amd.mono.common import convert_topk


def attach(lv) -> bool:
    """Wrap ``lv`` with the self-tests its ``cfg.extra`` requests; True when anything
    was attached."""
    extra = lv.cfg.extra or {}
    n_idx = int(extra.get("indexer_selftest_steps", 0) or 0)
    n_par = int(extra.get("fused_parity_steps", 0) or 0)
    if n_idx > 0:
        orig_refresh = lv._refresh_indexer

        def _refresh_indexer(layer, positions, normed):
            orig_refresh(layer, positions, normed)
            if (
                lv.cfg.indexer_mode == "indexer_only"
                and lv.stats["steps_mono"] <= n_idx
            ):
                _indexer_selftest(lv, layer, positions, normed, lv._st["T"])

        lv._refresh_indexer = _refresh_indexer
    if n_par > 0:
        orig_mono_forward = lv.mono_forward

        def mono_forward(layer, positions, hidden_states, residual):
            L = layer.layer_idx
            parity = L in lv.fused_layers and lv.stats["steps_mono"] <= n_par
            # a fused layer is never the first mono layer, so its input x is
            # ``residual``
            normed = layer.input_layernorm(residual) if parity else None
            out = orig_mono_forward(layer, positions, hidden_states, residual)
            if parity:
                _fused_parity(lv, layer, positions, normed, lv._st["T"])
            return out

        lv.mono_forward = mono_forward
    return n_idx > 0 or n_par > 0


def _fused_parity(lv, layer, positions, normed, T):
    """Eager debug (LiveConfig.extra.fused_parity_steps): the in-kernel
    indexer's CSR (just written by the launch) vs vLLM's own indexer refresh + convert
    on the same step, run into scratch buffers; the index cache is snapshotted before
    and restored after (the run continues on the kernel's own state). Two vLLM runs give
    the noise floor (vLLM rewrites the current row nondeterministically). Per row: set
    overlap |K & V| / |V|."""
    from vllm.models.deepseek_v32.amd.mono.indexer_only import refresh_indexer

    attn = layer.self_attn
    md = lv._st["md"]
    ic = attn.indexer.k_cache.kv_cache
    ic = ic[0] if isinstance(ic, (list, tuple)) else ic
    torch.accelerator.synchronize()
    ip = lv.b_indptr[: T + 1].clone()
    k_ind = lv.b_indices[: int(ip[T])].clone()
    snap = ic.clone()
    csrs = []
    for _ in range(2):
        ic.copy_(snap)
        refresh_indexer(attn, positions, normed, trim=lv.cfg.indexer_trim)
        ind = torch.zeros_like(lv.b_indices)
        convert_topk(md, attn.topk_indices_buffer[:T], ip, ind)
        csrs.append(ind[: int(ip[T])].clone())
    ic.copy_(snap)
    seq = (ip[1:] - ip[:-1]).tolist()
    pos = lv.b_pos[:T].tolist()

    def overlap(a, b):
        res = []
        for r in range(T):
            A = set(a[int(ip[r]) : int(ip[r + 1])].tolist())
            B = set(b[int(ip[r]) : int(ip[r + 1])].tolist())
            res.append(len(A & B) / max(1, len(B)))
        return res

    kv, vv = overlap(k_ind, csrs[0]), overlap(csrs[1], csrs[0])
    rec = dict(
        step=lv.stats["steps_mono"],
        layer=layer.layer_idx,
        ctx=[p + 1 for p in pos],
        n=seq,
        k_vs_v=kv,
        v_vs_v=vv,
    )
    lv.stats.setdefault("fused_parity", []).append(rec)


def _indexer_selftest(lv, layer, positions, normed, T):
    """indexer_only vs full vLLM attention: top-k rows and the whole index-K cache must
    match bitwise."""
    attn = layer.self_attn
    tk = attn.topk_indices_buffer[:T].clone()
    ic = attn.indexer.k_cache.kv_cache
    ic_snap = ic.clone()
    attn(positions=positions, hidden_states=normed)  # full path; rewrites the same rows
    full_tk = attn.topk_indices_buffer[:T]
    rec = dict(
        step=lv.stats["steps_mono"],
        layer=layer.layer_idx,
        topk_equal=bool(torch.equal(tk, full_tk)),
        # order-insensitive: vLLM's top-k order is atomics-dependent at ctx > 2048
        topk_set_equal=bool(
            torch.equal(tk.sort(dim=1).values, full_tk.sort(dim=1).values)
        ),
        index_cache_equal=bool(torch.equal(ic_snap, ic)),
    )
    if not rec["topk_set_equal"]:  # size of the disagreement, per row (max over rows)
        rec["topk_set_diff_max"] = max(
            len(set(tk[r].tolist()) ^ set(full_tk[r].tolist())) for r in range(T)
        )
    if not rec["index_cache_equal"]:
        d = ic_snap.view(torch.uint8) != ic.view(torch.uint8)
        nz = d.nonzero()
        blocks = sorted(set(nz[:, 0].tolist()))
        slots = lv.b_slot[:T].tolist()
        rec.update(
            ndiff=int(d.sum()),
            shape=list(ic.shape),
            dtype=str(ic.dtype),
            diff_blocks=blocks[:16],
            slot_blocks=[s // 16 for s in slots],
            slot_offsets=[s % 16 for s in slots],
            diff_idx_sample=nz[:8].tolist(),
        )
    # Control: is the full path deterministic against itself on the same input?
    snap2, tk2 = ic.clone(), attn.topk_indices_buffer[:T].clone()
    attn(positions=positions, hidden_states=normed)
    rec["full_vs_full_index_cache_equal"] = bool(torch.equal(snap2, ic))
    rec["full_vs_full_topk_equal"] = bool(
        torch.equal(tk2, attn.topk_indices_buffer[:T])
    )
    rec["full_vs_full_topk_set_equal"] = bool(
        torch.equal(
            tk2.sort(dim=1).values, attn.topk_indices_buffer[:T].sort(dim=1).values
        )
    )
    del snap2
    lv.stats.setdefault("indexer_selftest", []).append(rec)
    del ic_snap
