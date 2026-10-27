# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The split-row top-k pipeline must agree with the monolithic kernel exactly.

Both paths live in `topk_topp_triton`; `max_top_k` selects between them, so
each case here runs the same logits through both and compares bitwise.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.sample.ops.topk_topp_triton import (
    _TOPK_SPLIT_MAX_K,
    _TOPK_SPLIT_MIN_VOCAB,
    apply_top_k_top_p_triton,
)

if not current_platform.is_rocm():
    pytest.skip("the split top-k pipeline is gated to ROCm", allow_module_level=True)

VOCAB = 32768  # over _TOPK_SPLIT_MIN_VOCAB, small enough to keep tests quick
DEVICE = "cuda"


def _logits(rows: int, seed: int, scale: float = 4.0) -> torch.Tensor:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return (
        torch.randn(rows, VOCAB, generator=g, device=DEVICE, dtype=torch.float32)
        * scale
    )


def _both(logits, k, p):
    """(reference, split) for the same logits."""
    ref = apply_top_k_top_p_triton(logits.clone(), k, p, max_top_k=None)
    split = apply_top_k_top_p_triton(
        logits.clone(), k, p, max_top_k=int(k.max()) if k is not None else None
    )
    return ref, split


def _assert_identical(ref, split):
    # -inf == -inf compares equal, so torch.equal is the whole check.
    assert torch.equal(ref, split), (
        f"{int((ref != split).sum())} of {ref.numel()} positions differ"
    )


@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_matches_reference_at_decode_widths(rows):
    """The widths this exists for: one program per row starves the card."""
    k = torch.full((rows,), 20, dtype=torch.int32, device=DEVICE)
    p = torch.full((rows,), 0.95, dtype=torch.float32, device=DEVICE)
    _assert_identical(*_both(_logits(rows, 0), k, p))


@pytest.mark.parametrize("seed", range(8))
def test_matches_reference_across_draws(seed):
    k = torch.full((4,), 20, dtype=torch.int32, device=DEVICE)
    p = torch.full((4,), 0.95, dtype=torch.float32, device=DEVICE)
    _assert_identical(*_both(_logits(4, seed), k, p))


@pytest.mark.parametrize("k_val", [1, 2, 7, 20, 31, _TOPK_SPLIT_MAX_K])
def test_matches_reference_across_k(k_val):
    k = torch.full((2,), k_val, dtype=torch.int32, device=DEVICE)
    p = torch.full((2,), 0.9, dtype=torch.float32, device=DEVICE)
    _assert_identical(*_both(_logits(2, 1), k, p))


@pytest.mark.parametrize("p_val", [0.1, 0.5, 0.9, 0.99, 1.0])
def test_matches_reference_across_p(p_val):
    """p = 1.0 included: a saturating prefix-mass cut truncates such a row."""
    k = torch.full((2,), 20, dtype=torch.int32, device=DEVICE)
    p = torch.full((2,), p_val, dtype=torch.float32, device=DEVICE)
    _assert_identical(*_both(_logits(2, 2), k, p))


def test_matches_reference_with_mixed_per_row_params():
    k = torch.tensor([1, 20, 7, 64], dtype=torch.int32, device=DEVICE)
    p = torch.tensor([1.0, 0.95, 0.5, 0.8], dtype=torch.float32, device=DEVICE)
    _assert_identical(*_both(_logits(4, 3), k, p))


def test_matches_reference_without_top_p():
    k = torch.full((4,), 20, dtype=torch.int32, device=DEVICE)
    _assert_identical(*_both(_logits(4, 4), k, None))


def test_matches_reference_on_tied_logits():
    """Duplicates must be emitted once each, not retired together."""
    logits = _logits(2, 5)
    logits[:, :64] = 3.5  # far more ties than any row's k
    k = torch.full((2,), 20, dtype=torch.int32, device=DEVICE)
    p = torch.full((2,), 0.95, dtype=torch.float32, device=DEVICE)
    _assert_identical(*_both(logits, k, p))


def test_matches_reference_with_masked_rows():
    """Grammar masks leave rows mostly -inf."""
    logits = _logits(2, 6)
    logits[0, 100:] = float("-inf")
    logits[1, :-100] = float("-inf")
    k = torch.full((2,), 20, dtype=torch.int32, device=DEVICE)
    p = torch.full((2,), 0.95, dtype=torch.float32, device=DEVICE)
    _assert_identical(*_both(logits, k, p))


def test_a_zero_variance_row_diverges_from_the_reference():
    """The one input where the two paths disagree, recorded rather than hidden.

    On a row with no variance at all the reference masks nothing, so more than
    k tokens survive top_k; the split path keeps k and then applies top-p to
    them. Partial ties agree (see above) -- only full uniformity differs, which
    is where the reference's normal-approximation pivot search has no spread to
    work from. Real decode distributions are not uniform, but the difference is
    a behaviour change and belongs in the PR's limitations.
    """
    logits = torch.zeros(2, VOCAB, device=DEVICE, dtype=torch.float32)
    k = torch.full((2,), 20, dtype=torch.int32, device=DEVICE)
    p = torch.full((2,), 0.95, dtype=torch.float32, device=DEVICE)
    ref, split = _both(logits, k, p)
    assert int((ref > float("-inf")).sum()) == 2 * VOCAB, "reference masks nothing"
    assert int((split > float("-inf")).sum()) == 2 * 19, "split keeps top-p of k"


def test_the_split_path_is_actually_taken():
    """Guard against the suite passing by never leaving the reference path.

    Without this, a gate that silently declines every case would still show
    green on every agreement test above.
    """
    from unittest import mock

    import vllm.v1.sample.ops.topk_topp_triton as mod

    k = torch.full((2,), 20, dtype=torch.int32, device=DEVICE)
    p = torch.full((2,), 0.95, dtype=torch.float32, device=DEVICE)
    with mock.patch.object(
        mod, "_apply_topk_split", wraps=mod._apply_topk_split
    ) as spy:
        apply_top_k_top_p_triton(_logits(2, 8), k, p, max_top_k=20)
        assert spy.call_count == 1
        apply_top_k_top_p_triton(_logits(2, 8), k, p, max_top_k=None)
        assert spy.call_count == 1, "max_top_k=None must stay on the reference"


@pytest.mark.parametrize(
    "max_top_k,reason",
    [
        (_TOPK_SPLIT_MAX_K + 1, "k past the candidate cap"),
        (0, "k below one"),
    ],
)
def test_declines_outside_its_bounds(max_top_k, reason):
    """Unsupported bounds fall through rather than answering approximately."""
    rows = 2
    k = torch.full((rows,), 20, dtype=torch.int32, device=DEVICE)
    p = torch.full((rows,), 0.95, dtype=torch.float32, device=DEVICE)
    logits = _logits(rows, 9)
    ref = apply_top_k_top_p_triton(logits.clone(), k, p, max_top_k=None)
    out = apply_top_k_top_p_triton(logits.clone(), k, p, max_top_k=max_top_k)
    _assert_identical(ref, out)


def test_declines_a_vocabulary_too_small_to_benefit():
    rows, small = 2, _TOPK_SPLIT_MIN_VOCAB // 2
    g = torch.Generator(device=DEVICE).manual_seed(10)
    logits = torch.randn(rows, small, generator=g, device=DEVICE) * 4
    k = torch.full((rows,), 20, dtype=torch.int32, device=DEVICE)
    p = torch.full((rows,), 0.95, dtype=torch.float32, device=DEVICE)
    ref = apply_top_k_top_p_triton(logits.clone(), k, p, max_top_k=None)
    out = apply_top_k_top_p_triton(logits.clone(), k, p, max_top_k=20)
    _assert_identical(ref, out)
