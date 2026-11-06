# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The split-row top-k pipeline against an exact sort-based top-k/top-p cut."""

import pytest
import torch

import vllm.v1.sample.ops.topk_topp_triton as mod
from vllm.platforms import current_platform
from vllm.v1.sample.ops.topk_topp_triton import (
    _TOPK_SPLIT_MAX_K,
    _TOPK_SPLIT_MIN_VOCAB,
    _topk_split_supported,
    apply_top_k_top_p_triton,
)

if not current_platform.is_cuda_alike():
    pytest.skip("needs a CUDA-like GPU", allow_module_level=True)

rocm_only = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="the split pipeline runs on ROCm only"
)

VOCAB = 151936
DEVICE = "cuda"


def _logits(rows: int, seed: int, vocab: int = VOCAB) -> torch.Tensor:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(rows, vocab, generator=g, device=DEVICE) * 4
    return x.bfloat16().float()


def _exact(logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor | None):
    """Top-k with ties to the lowest index, then top-p over the survivors."""
    out = torch.full_like(logits, float("-inf"))
    vals, idx = torch.sort(logits.double(), dim=-1, descending=True, stable=True)
    for r in range(logits.shape[0]):
        v, i = vals[r, : int(k[r])], idx[r, : int(k[r])]
        keep = torch.ones_like(v, dtype=torch.bool)
        if p is not None and float(p[r]) < 1.0 and torch.isfinite(v[0]):
            prob = torch.softmax(v, 0)
            keep = torch.cumsum(prob, 0) - prob < float(p[r])
        out[r, i[keep]] = logits[r, i[keep]]
    return out


def _split(logits, k, p):
    out = logits.clone()
    mod._apply_topk_split(out, k, p, int(k.max()), float("-inf"))
    return out


def _ks(vals) -> torch.Tensor:
    return torch.tensor(vals, dtype=torch.int32, device=DEVICE)


def _ps(vals) -> torch.Tensor:
    return torch.tensor(vals, dtype=torch.float32, device=DEVICE)


def _ties_at_kth(rows: int, k: int) -> torch.Tensor:
    x = _logits(rows, 1)
    kth = torch.topk(x, k).values[:, -1:]
    x[:, ::997] = kth
    return x


def _ties_at_topp_cut(rows: int, k: int, vocab: int = VOCAB) -> torch.Tensor:
    x = torch.full((rows, vocab), -30.0, device=DEVICE)
    x[:, 5 : 5 + 3 * k : 3] = 0.5
    x[:, [9000, 70000, 140000]] = torch.tensor([2.0, 1.5, 1.0], device=DEVICE)
    return x


def _few_finite() -> torch.Tensor:
    x = torch.full((2, VOCAB), float("-inf"), device=DEVICE)
    x[:, [0, 7, 300]] = torch.tensor([1.0, 3.0, 2.0], device=DEVICE)
    return x


CASES = {
    "mixed": (lambda: _logits(4, 0), [1, 16, 17, 64], [0.9, 0.95, 0.5, 1.0]),
    "no_p": (lambda: _logits(2, 2), [20, 20], None),
    "p_edges": (lambda: _logits(2, 3), [24, 33], [1e-6, 0.999999]),
    "tie_k": (lambda: _ties_at_kth(2, 20), [20, 20], [0.95, 1.0]),
    "tie_p": (lambda: _ties_at_topp_cut(2, 20), [20, 20], [0.5, 0.9]),
    "uniform": (lambda: torch.zeros(2, VOCAB, device=DEVICE), [20, 64], [0.93, 1.0]),
    "few_finite": (_few_finite, [20, 20], None),
}


@rocm_only
@pytest.mark.parametrize("case", CASES)
def test_matches_exact_cut(case):
    make, k, p = CASES[case]
    logits, k = make(), _ks(k)
    p = None if p is None else _ps(p)
    assert torch.equal(_split(logits, k, p), _exact(logits, k, p))


@rocm_only
def test_nan_row_leaves_other_rows_exact():
    logits = _logits(3, 5)
    logits[1] = float("nan")
    k, p = _ks([20, 20, 20]), _ps([0.95, 0.95, 0.95])
    out = _split(logits, k, p)
    healthy = [0, 2]
    assert torch.equal(out[healthy], _exact(logits[healthy], k[healthy], p[healthy]))


@rocm_only
def test_new_k_in_a_tile_compiles_nothing():
    def compiled():
        kernels = (mod._topk_split_chunk_kernel, mod._topk_split_combine_kernel)
        return sum(len(c[0]) for kern in kernels for c in kern.device_caches.values())

    p = _ps([0.95, 0.95])
    _split(_logits(2, 6), _ks([17, 17]), p)
    before = compiled()
    for k_val in (18, 20, 31, 32):
        _split(_logits(2, 6), _ks([k_val, k_val]), p)
    assert compiled() == before


@pytest.mark.parametrize(
    "rows,max_top_k,vocab,taken",
    [
        (1, 20, VOCAB, True),
        (64, 24, VOCAB, True),
        (65, 24, VOCAB, False),
        (4, _TOPK_SPLIT_MAX_K, VOCAB, True),
        (5, _TOPK_SPLIT_MAX_K, VOCAB, False),
        (2, _TOPK_SPLIT_MAX_K + 1, VOCAB, False),
        (2, None, VOCAB, False),
        (2, 20, _TOPK_SPLIT_MIN_VOCAB - 1, False),
    ],
)
def test_dispatch(rows, max_top_k, vocab, taken):
    """Taken shapes give the exact cut; declined ones match the monolithic."""
    logits = _ties_at_topp_cut(rows, 20, vocab)
    k = torch.full((rows,), 20, dtype=torch.int32, device=DEVICE)
    p = torch.full((rows,), 0.5, device=DEVICE)
    mono = apply_top_k_top_p_triton(logits.clone(), k, p, has_topp_only_rows=False)
    exact = _exact(logits, k, p)
    assert not torch.equal(mono, exact), "input must tell the two paths apart"
    out = apply_top_k_top_p_triton(
        logits.clone(), k, p, has_topp_only_rows=False, max_top_k=max_top_k
    )
    expected = exact if taken and _topk_split_supported() else mono
    assert torch.equal(out, expected)
