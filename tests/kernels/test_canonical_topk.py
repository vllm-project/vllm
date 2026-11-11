# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""canonicalize_topk_ turns any valid top-k (any tie choice, any order) into the
same result: candidates above the k-th score plus the lowest-index ties, ascending."""

import pytest
import torch

from vllm.model_executor.layers.canonical_topk import _ordered_key, canonicalize_topk_

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEVICE = torch.device("cuda")


def _reference(logits: torch.Tensor, row_ends: torch.Tensor, k: int) -> torch.Tensor:
    """Canonical top-k on CPU: rank by (key desc, index asc), keep k, sort ascending."""
    out = torch.full((logits.shape[0], k), -1, dtype=torch.int32)
    keys = _ordered_key(logits.cpu())
    for r in range(logits.shape[0]):
        n = int(row_ends[r])
        cols = torch.arange(n)
        order = sorted(cols.tolist(), key=lambda c: (-int(keys[r, c]), c))
        keep = sorted(order[: min(k, n)])
        out[r, : len(keep)] = torch.tensor(keep, dtype=torch.int32)
    return out


def _scrambled_topk(
    logits: torch.Tensor, row_ends: torch.Tensor, k: int, gen: torch.Generator
) -> torch.Tensor:
    """A valid top-k that resolves ties at the k-th score at random, in random
    order, with empty slots scattered (as an atomic radix top-k may produce)."""
    out = torch.full((logits.shape[0], k), -1, dtype=torch.int32)
    keys = _ordered_key(logits.cpu())
    for r in range(logits.shape[0]):
        n = int(row_ends[r])
        if n == 0:
            continue
        rk = keys[r, :n]
        m = min(k, n)
        kth = torch.sort(rk, descending=True).values[m - 1]
        above = torch.nonzero(rk > kth).flatten()
        tied = torch.nonzero(rk == kth).flatten()
        tied = tied[torch.randperm(len(tied), generator=gen)][: m - len(above)]
        chosen = torch.cat([above, tied])
        chosen = chosen[torch.randperm(len(chosen), generator=gen)]
        slots = torch.randperm(k, generator=gen)[: len(chosen)]
        out[r, slots] = chosen.to(torch.int32)
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("levels", [4, 64, None])
@pytest.mark.parametrize("cols, k", [(3000, 128), (5000, 2048), (700, 1024)])
def test_canonical_topk_matches_reference(dtype, levels, cols, k):
    gen = torch.Generator().manual_seed(cols + k + (levels or 0))
    rows = 6
    logits = torch.randn(rows, cols, generator=gen)
    if levels is not None:
        # Few distinct scores: many ties at the k-th one.
        logits = torch.round(logits * levels / 4) / levels
    logits = logits.to(dtype)
    # Mixed row lengths, including a short row (fewer than k candidates) and an
    # empty one.
    row_ends = torch.tensor(
        [cols, cols - 1, cols // 2, min(k, cols) // 3, 0, cols], dtype=torch.int32
    )
    expected = _reference(logits, row_ends, k)
    for _ in range(3):
        indices = _scrambled_topk(logits, row_ends, k, gen).to(DEVICE)
        canonicalize_topk_(logits.to(DEVICE), indices, row_ends.to(DEVICE))
        torch.testing.assert_close(indices.cpu(), expected, rtol=0, atol=0)


def test_signed_zero_and_nan_follow_kernel_order():
    # The kernels rank -0.0 below +0.0 and NaN above +inf (convert_to_uint32_v2).
    logits = torch.tensor(
        [[0.0, -0.0, float("nan"), 1.0, -0.0, 0.0, float("inf")]], dtype=torch.float32
    )
    row_ends = torch.tensor([7], dtype=torch.int32)
    # Top-4 = NaN, +inf, 1.0, then one +0.0 (the lowest index, 0).
    out = torch.tensor([[5, 2, 6, 3]], dtype=torch.int32, device=DEVICE)
    expected = torch.tensor([[0, 2, 3, 6]], dtype=torch.int32)
    canonicalize_topk_(logits.to(DEVICE), out, row_ends.to(DEVICE))
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["persistent", "cooperative", "per_row", "torch"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sparse_indexer_topk_is_canonical(
    monkeypatch: pytest.MonkeyPatch, workspace_init, backend, dtype
):
    """With VLLM_DETERMINISTIC_SPARSE_TOPK every decode top-k backend returns the
    canonical result, whatever its own tie choice and order."""
    import vllm.envs as envs
    from vllm.model_executor.layers.indexer_topk import get_indexer_topk

    monkeypatch.setattr(envs, "VLLM_DETERMINISTIC_SPARSE_TOPK", True, raising=False)
    get_indexer_topk.cache_clear()
    gen = torch.Generator().manual_seed(7)
    rows, cols, k = 4, 8192, 2048
    # Few distinct scores, so many candidates tie with the k-th one.
    logits = (torch.round(torch.randn(rows, cols, generator=gen) * 4) / 4).to(dtype)
    seq_lens = torch.tensor([8192, 5000, 3001, 2100], dtype=torch.int32)
    expected = _reference(logits, seq_lens, k)
    indices = torch.full((rows, k), -1, dtype=torch.int32, device=DEVICE)
    selector = get_indexer_topk(backend)
    try:
        selector(logits.to(DEVICE), seq_lens.to(DEVICE), 1, indices, k, cols)
    except RuntimeError as e:  # backend not available on this device
        pytest.skip(str(e))
    finally:
        get_indexer_topk.cache_clear()
    torch.testing.assert_close(indices.cpu(), expected, rtol=0, atol=0)
