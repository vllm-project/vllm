# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of MonoKernel speculative decoding (MTP verify steps): the step go / no-go
rule (mono/spec.py) on synthetic metadata, the width map, and the kernel's per-row
attention model of a verify step (kernel/glm/kernel.py split_keys / gather_old_kv /
patch_new_kv) against a dense causal golden."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm.models.deepseek_v32.amd.mono import spec as SP


@pytest.mark.parametrize(
    "n_dec, T_md, prefills, qlen, topk, T, residual, max_qlen, want",
    [
        (4, 4, 0, 1, 2048, 4, True, 1, ""),  # plain decode: q_len 1 only
        (8, 8, 0, 2, 2048, 8, True, 1, "query_len"),
        # MTP k = 3: 4 rows / request; a request just out of prefill has 1 row
        (8, 8, 0, 4, 2048, 8, True, 4, ""),
        (5, 5, 0, 4, 2048, 5, True, 4, ""),
        (12, 12, 0, 4, 2048, 12, True, 4, ""),
        (16, 16, 0, 4, 2048, 16, True, 4, "too_many_rows"),
        (8, 8, 0, 5, 2048, 8, True, 4, "query_len"),  # longer than the verify length
        (4, 40, 1, 36, 2048, 40, True, 4, "prefill_or_mixed"),
        (4, 8, 0, 4, 2048, 8, True, 4, "prefill_or_mixed"),
        (8, 8, 0, 4, 2048, 12, True, 4, "padded_T"),
        (8, 8, 0, 4, 2048, 8, False, 4, "no_residual"),
        (8, 8, 0, 4, 1024, 8, True, 4, "topk"),
    ],
)
def test_step_reason(n_dec, T_md, prefills, qlen, topk, T, residual, max_qlen, want):
    md = NS(
        num_decode_tokens=n_dec,
        num_actual_tokens=T_md,
        num_prefills=prefills,
        max_query_len=qlen,
        topk_tokens=topk,
    )
    fits = SP.width_for(T, SP.KERNEL_WIDTHS) is not None
    assert SP.step_reason(md, T, residual, max_qlen, fits) == want


def test_width_map():
    assert SP.spec_widths(1) == (2, 4, 6, 8, 10, 12)
    assert SP.spec_widths(3) == (4, 8, 12)
    assert SP.spec_widths(4) == (5, 10)
    # B requests x (1 + k) rows -> kernel width
    got = {
        (k, B): SP.width_for(B * (1 + k), SP.spec_widths(k))
        for k, B in ((3, 1), (3, 2), (3, 3), (3, 4), (1, 4), (1, 6), (1, 7))
    }
    assert list(got.values()) == [4, 8, 12, None, 8, 12, None]


def test_verify_step_attention():
    """One request's verify step (1 + k rows): every row is its own CSR row; a key whose
    slot is any row's new slot of THIS launch comes from the kvnew mailbox, every other
    key from the paged cache (still stale at this launch's slots). Equal to the dense
    causal golden; reading those slots from the cache instead is wrong, so draft row j
    sees rows 0 .. j - 1 of the same launch."""
    g = torch.Generator().manual_seed(0)
    D, ctx0, k, nslots = 16, 37, 3, 256
    perm = torch.randperm(nslots, generator=g)
    slot_of = {p: int(perm[p]) for p in range(ctx0 + k + 1)}
    kv_true = {p: torch.randn(D, generator=g) for p in range(ctx0 + k + 1)}
    cache = torch.randn(nslots, D, generator=g)  # stale contents everywhere ...
    for p in range(ctx0):
        cache[slot_of[p]] = kv_true[p]  # ... except the keys written by earlier steps
    q = torch.randn(k + 1, D, generator=g)
    # rows' slot_mapping -> the cache task's mailbox values
    mailbox = {slot_of[ctx0 + j]: kv_true[ctx0 + j] for j in range(k + 1)}

    def attend(qv, keys):
        keys = torch.stack(keys)
        return (torch.softmax(keys @ qv, 0)[:, None] * keys).sum(0)

    for patch in (True, False):
        err = 0.0
        for j in range(k + 1):
            # row j's causal CSR (physical slots), as vLLM's convert writes it
            csr = [slot_of[p] for p in range(ctx0 + j + 1)]
            keys = [mailbox[s] if patch and s in mailbox else cache[s] for s in csr]
            gold = attend(q[j], [kv_true[p] for p in range(ctx0 + j + 1)])
            err = max(err, float((attend(q[j], keys) - gold).abs().max()))
        assert err < 1e-6 if patch else err > 1e-3, (patch, err)
