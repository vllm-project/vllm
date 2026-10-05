# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for MonoKernel speculative decoding (MTP verify steps).

1. mono/spec.py step_reason: the go / no-go rule on synthetic attention metadata --
   plain decode, MTP verify steps (1 + k rows per request, also mixed 1 / 1 + k after a
   prefill), prefills, mixed batches, too many rows.
2. Width map: B requests x (1 + k) rows -> kernel width; spec_widths(k) == the
   FULL-graph capture sizes to use.
3. The kernel's per-row attention model on a verify step (kernel/glm/kernel.py
   split_keys / gather_old_kv / patch_new_kv): every row is its own CSR row; a key whose
   slot is any row's new slot of THIS launch comes from the kvnew mailbox, every other
   key from the paged cache (which still holds stale bytes at this launch's slots while
   the cache stage writes them). vs a dense causal golden of the whole request: equal,
   and reading those slots from the cache instead (no mailbox patch) is wrong -- so
   draft row j sees rows 0 .. j - 1 of the same launch.
"""

from types import SimpleNamespace

import torch

from vllm.models.deepseek_v32.amd.mono import spec as SP


def md(
    num_decode_tokens,
    num_actual_tokens,
    num_prefills=0,
    max_query_len=1,
    topk_tokens=2048,
):
    return SimpleNamespace(
        num_decode_tokens=num_decode_tokens,
        num_actual_tokens=num_actual_tokens,
        num_prefills=num_prefills,
        max_query_len=max_query_len,
        topk_tokens=topk_tokens,
    )


def test_step_reason():
    R = SP.step_reason
    sizes = (1, 2, 4, 5, 6, 8, 10, 12)
    fits = lambda T: SP.width_for(T, sizes) is not None  # noqa: E731
    # plain decode (no spec): only q_len 1
    assert R(md(4, 4), 4, True, 1, fits(4)) == ""
    assert R(md(8, 8, max_query_len=2), 8, True, 1, fits(8)) == "query_len"
    # MTP k = 3: verify steps 4 rows / request; a request just out of prefill has 1 row
    # (mixed 1 / 4 is fine)
    assert R(md(8, 8, max_query_len=4), 8, True, 4, fits(8)) == ""
    assert R(md(5, 5, max_query_len=4), 5, True, 4, fits(5)) == ""
    assert R(md(12, 12, max_query_len=4), 12, True, 4, fits(12)) == ""
    assert R(md(16, 16, max_query_len=4), 16, True, 4, fits(16)) == "too_many_rows"
    # longer than the verify length
    assert R(md(8, 8, max_query_len=5), 8, True, 4, fits(8)) == "query_len"
    # prefill / mixed / padding / residual / top-k
    assert (
        R(md(4, 40, num_prefills=1, max_query_len=36), 40, True, 4, fits(40))
        == "prefill_or_mixed"
    )
    assert R(md(4, 8, max_query_len=4), 8, True, 4, fits(8)) == "prefill_or_mixed"
    assert R(md(8, 8, max_query_len=4), 12, True, 4, fits(12)) == "padded_T"
    assert R(md(8, 8, max_query_len=4), 8, False, 4, fits(8)) == "no_residual"
    assert R(md(8, 8, max_query_len=4, topk_tokens=1024), 8, True, 4, fits(8)) == "topk"


def test_width_map():
    assert SP.spec_widths(1) == (2, 4, 6, 8, 10, 12)
    assert SP.spec_widths(3) == (4, 8, 12)
    assert SP.spec_widths(4) == (5, 10)
    tab = {}
    for k in (1, 2, 3):
        sizes = SP.spec_widths(k)
        for B in range(1, 9):
            tab[(k, B)] = SP.width_for(B * (1 + k), sizes)
    assert (
        tab[(3, 1)] == 4
        and tab[(3, 2)] == 8
        and tab[(3, 3)] == 12
        and tab[(3, 4)] is None
    )
    assert tab[(1, 4)] == 8 and tab[(1, 6)] == 12 and tab[(1, 7)] is None


def test_verify_step_attention():
    """Kernel-model attention of one request's verify step (1 + k rows) vs a dense
    causal golden."""
    g = torch.Generator().manual_seed(0)
    # ctx0 accepted tokens before this step; rows j = 0..k at positions ctx0 .. ctx0 + k
    D, ctx0, k = 16, 37, 3
    nslots = 256
    perm = torch.randperm(nslots, generator=g)
    # positions -> physical slots (one request)
    slot_of = {p: int(perm[p]) for p in range(ctx0 + k + 1)}
    kv_true = {p: torch.randn(D, generator=g) for p in range(ctx0 + k + 1)}
    cache = torch.randn(nslots, D, generator=g)  # stale contents everywhere ...
    for p in range(ctx0):
        cache[slot_of[p]] = kv_true[p]  # ... except the keys written by earlier steps
    q = torch.randn(k + 1, D, generator=g)
    new_slots = [slot_of[ctx0 + j] for j in range(k + 1)]  # rows' slot_mapping
    kvnew = [kv_true[ctx0 + j] for j in range(k + 1)]  # the cache task's mailbox values

    def attend(qv, keys):
        s = torch.stack(keys) @ qv
        w = torch.softmax(s, 0)
        return (w[:, None] * torch.stack(keys)).sum(0)

    for patch in (True, False):
        err = 0.0
        for j in range(k + 1):
            # causal length of row j (sparse-MLA: min(ctx_start + j + 1, 2048); identity
            # here)
            L = ctx0 + j + 1
            # row j's CSR (physical slots), as vLLM's convert writes it
            csr = [slot_of[p] for p in range(L)]
            keys = []
            for sl in csr:
                src = cache[sl]
                # patch_new_kv: any row's new slot of this launch -> that row's mailbox
                # values
                if patch:
                    for r, ns in enumerate(new_slots):
                        if sl == ns:
                            src = kvnew[r]
                keys.append(src)
            gold = attend(q[j], [kv_true[p] for p in range(L)])
            err = max(err, float((attend(q[j], keys) - gold).abs().max()))
        if patch:
            assert err < 1e-6, err
        else:
            # without the mailbox patch draft rows would read stale cache rows
            assert err > 1e-3, err
