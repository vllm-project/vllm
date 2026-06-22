# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static (no-GPU) unit tests for the FlashInfer NVFP4 VO-split factor and
the Gemma 3/4 mm-prefix bidirectional image-span masking.

These exercise pure tensor/index logic on CPU — no CUDA, no kernel launch:

* ``_vo_split_factor`` — how many ``(qk=512, vo=256)`` passes a head needs.
* ``_build_mm_prefix_custom_mask`` — the per-request boolean attention mask,
  ``(causal AND sliding-window) OR (q in span AND kv in span)``.
* ``_mm_prefix_prefill_spans`` — which prefill requests have an image span
  intersecting the query window (and the byte-identical None fast path).
"""

from types import SimpleNamespace

import pytest
import torch

try:
    from vllm.v1.attention.backends.flashinfer import (
        FlashInferMetadataBuilder,
        _vo_split_factor,
    )

    HAS_FI = True
except Exception:
    HAS_FI = False

pytestmark = pytest.mark.skipif(
    not HAS_FI, reason="FlashInfer attention backend not importable"
)


# --------------------------------------------------------------------------- #
# _vo_split_factor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "head_size,is_nvfp4,expected",
    [
        (256, True, 1),   # uniform Gemma 3 head — single pass
        (128, True, 1),
        (256, False, 1),
        (512, True, 2),   # Gemma 4 global head — two (qk=512, vo=256) passes
        (512, False, 2),  # the split is dtype-independent
    ],
)
def test_vo_split_factor(head_size, is_nvfp4, expected, monkeypatch):
    # the >256 NVFP4 path requires the (default-on) VO-split knob; pin it on.
    monkeypatch.setenv("VLLM_NVFP4_KV_VOSPLIT", "1")
    assert _vo_split_factor(head_size, is_nvfp4) == expected


def test_vo_split_factor_nvfp4_needs_knob(monkeypatch):
    # NVFP4 512-wide head with the split disabled must error, not silently
    # plan an unsupported HEAD_DIM_VO=512.
    monkeypatch.setenv("VLLM_NVFP4_KV_VOSPLIT", "0")
    with pytest.raises(ValueError, match="two-pass VO split"):
        _vo_split_factor(512, True)


# --------------------------------------------------------------------------- #
# _build_mm_prefix_custom_mask
# --------------------------------------------------------------------------- #
def _mask(window_left, qo_lens, kv_lens, span_lists):
    fake = SimpleNamespace(device=torch.device("cpu"), window_left=window_left)
    flat = FlashInferMetadataBuilder._build_mm_prefix_custom_mask(
        fake, qo_lens, kv_lens, span_lists
    )
    # single-request helper: reshape back to (qo_len, kv_len)
    return flat.reshape(qo_lens[0], kv_lens[0])


def test_mm_mask_pure_causal_no_span():
    # No window, no spans -> strict lower-triangular causal mask.
    m = _mask(-1, [4], [4], [[]])
    expected = torch.tensor(
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]],
        dtype=torch.bool,
    )
    assert torch.equal(m, expected)


def test_mm_mask_span_is_bidirectional():
    # Span [1,2]: queries 1,2 may attend *forward* to keys 1,2 inside the
    # span (the whole point — image tokens attend bidirectionally), while
    # everything outside the span stays causal.
    m = _mask(-1, [4], [4], [[(1, 2)]])
    expected = torch.tensor(
        [[1, 0, 0, 0],
         [1, 1, 1, 0],   # q=1 now sees k=2 (forward, in-span)
         [1, 1, 1, 0],
         [1, 1, 1, 1]],
        dtype=torch.bool,
    )
    assert torch.equal(m, expected)
    # q=2 still cannot see k=3 (k=3 is outside the span and non-causal)
    assert not m[2, 3]


def test_mm_mask_sliding_window_ands_with_causal():
    # window_left=1: a query sees only itself and one key back, ANDed with
    # causal. No spans.
    m = _mask(1, [4], [4], [[]])
    expected = torch.tensor(
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [0, 1, 1, 0],   # k=0 is outside the window for q=2
         [0, 0, 1, 1]],
        dtype=torch.bool,
    )
    assert torch.equal(m, expected)


def test_mm_mask_span_overrides_window():
    # The mask carries the window itself (the mm wrapper is planned
    # window_left=-1), and an in-span pair must survive even when the window
    # would have excluded it: q=3,k=1 are >window apart but both in span.
    m = _mask(1, [4], [4], [[(1, 3)]])
    assert m[3, 1]                       # span overrides the window
    assert m[1, 3]                       # and is bidirectional
    assert not m[3, 0]                   # k=0 outside span and window -> off


def test_mm_mask_with_context_offset():
    # qo_len < kv_len: the query rows are end-aligned to the KV sequence, so
    # absolute query positions start at kv_len - qo_len (= context_len).
    m = _mask(-1, [2], [4], [[]])  # 2 new queries over a 4-long KV
    # q_abs = [2, 3], k_abs = [0,1,2,3]; causal k <= q
    expected = torch.tensor(
        [[1, 1, 1, 0],   # q=2 sees k 0..2
         [1, 1, 1, 1]],  # q=3 sees k 0..3
        dtype=torch.bool,
    )
    assert torch.equal(m, expected)


# --------------------------------------------------------------------------- #
# _mm_prefix_prefill_spans
# --------------------------------------------------------------------------- #
def _spans(mm_ranges, qo_indptr, seq_lens, num_decodes, num_prefills):
    cam = SimpleNamespace(
        mm_req_doc_ranges=mm_ranges,
        query_start_loc_cpu=torch.tensor(qo_indptr, dtype=torch.int32),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
    )
    return FlashInferMetadataBuilder._mm_prefix_prefill_spans(
        None, cam, num_decodes, num_prefills
    )


def test_spans_none_when_no_mm_ranges():
    # No image spans anywhere -> None (the scalar-causal fast path).
    assert _spans(None, [0, 8], [8], 0, 1) is None
    assert _spans({}, [0, 8], [8], 0, 1) is None


def test_spans_intersecting_window_kept():
    # One prefill request (kv_len=10, qo_len=10 -> context_len=0); span (2,5)
    # intersects the query window and is returned.
    out = _spans({0: [(2, 5)]}, [0, 10], [10], 0, 1)
    assert out == [[(2, 5)]]


def test_spans_fully_in_context_dropped():
    # kv_len=10, qo_len=2 -> context_len=8; span (2,5) ends before the window
    # (e=5 < context_len=8), so it needs no masking -> filtered -> None.
    assert _spans({0: [(2, 5)]}, [0, 2], [10], 0, 1) is None


def test_spans_skip_decode_requests():
    # 1 decode + 1 prefill; the doc range belongs to the prefill (index 1).
    # query_start_loc covers [decode, prefill] -> prefill qo_len = 10.
    out = _spans({1: [(3, 6)]}, [0, 1, 11], [5, 11], 1, 1)
    assert out == [[(3, 6)]]
