# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LongCat n-gram embedding id computation vs a pure-Python reference.

Guards the hash-id semantics of ``ngram_compute_n_gram_ids`` plus the
EOS-position fixup (``compute_eos_position_ngram_ids``): an EOS *current*
token hashes with its full look-back, while later positions' look-back stops
at the EOS boundary (LongCat reference behavior).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.models.longcat_flash_ngram import (
    compute_eos_position_ngram_ids,
)
from vllm.platforms import current_platform

VOCAB = 163840
N, K = 5, 4  # oe_neighbor_num, oe_split_num (LongCat-2.0)
M = int(100.567 * VOCAB)
EOS = 2
NUM_EMB = K * (N - 1)

SIZES = [M + c * 2 + 1 for c in range(NUM_EMB)]
OFFSETS = [0]
for s in SIZES:
    OFFSETS.append(OFFSETS[-1] + s)


def _ref_ids(tokens: list[int]) -> list[list[int]]:
    """Reference n-gram ids on raw tokens (HF LongCat semantics)."""
    out = []
    for pos in range(len(tokens)):
        row = []
        for i in range(N - 1):
            n_real = i + 2
            for j in range(K):
                cfg = i * K + j
                mod = SIZES[cfg]
                h = 0
                for delta in range(n_real):
                    p = pos - delta
                    if p < 0:
                        break
                    t = tokens[p]
                    if delta > 0 and t == EOS:
                        break  # look-back stops at an EOS boundary
                    h += (t * pow(VOCAB, delta, mod)) % mod
                row.append(h % mod + OFFSETS[cfg])
        out.append(row)
    return out


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA kernel")
def test_ngram_ids_match_reference_with_eos():
    device = "cuda"
    torch.manual_seed(0)
    tokens = torch.randint(3, VOCAB, (14,), dtype=torch.int32).tolist()
    tokens[5] = EOS
    tokens[6] = EOS  # double EOS: second one's look-back stops at the first
    tokens[13] = EOS  # trailing EOS (chat-template turn boundary)

    ctx_len = N - 1
    toks_neg = [-t if t == EOS else t for t in tokens]
    width = ctx_len + len(tokens)
    table = torch.full((1, width), -1, dtype=torch.int32, device=device)
    table[0, ctx_len:] = torch.tensor(toks_neg, dtype=torch.int32, device=device)

    ne_weights = torch.zeros(N - 1, K, N, dtype=torch.int32)
    ne_mods = torch.zeros(N - 1, K, dtype=torch.int32)
    for i in range(N - 1):
        for j in range(K):
            mod = SIZES[i * K + j]
            ne_mods[i, j] = mod
            for delta in range(N):
                ne_weights[i, j, delta] = pow(VOCAB, delta, mod)
    ngram = SimpleNamespace(
        n=N,
        k=K,
        num_embedders=NUM_EMB,
        ne_weights=ne_weights.to(device),
        ne_mods=ne_mods.to(device),
        exclusive_sizes=torch.tensor(OFFSETS, dtype=torch.int32, device=device),
    )

    T = len(tokens)
    qsl = torch.tensor([0, T], dtype=torch.int32, device=device)
    row_indices = torch.zeros(1, dtype=torch.int64, device=device)
    column_starts = torch.full((1,), ctx_len, dtype=torch.int32, device=device)
    got = torch.empty(T, NUM_EMB, dtype=torch.int32, device=device)
    ops.ngram_compute_n_gram_ids(
        N,
        K,
        ngram.ne_weights,
        ngram.ne_mods,
        ngram.exclusive_sizes,
        qsl,
        table,
        row_indices,
        column_starts,
        got,
    )

    cur = torch.tensor(tokens, dtype=torch.int32, device=device)
    tok_req = torch.zeros(T, dtype=torch.int64, device=device)
    col = ctx_len + torch.arange(T, device=device)
    eos_tok = (cur == EOS).nonzero(as_tuple=True)[0]
    got[eos_tok] = compute_eos_position_ngram_ids(
        ngram, EOS, table, tok_req, col, eos_tok
    )

    want = torch.tensor(_ref_ids(tokens), dtype=torch.int32)
    torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
