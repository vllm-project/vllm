# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for placeholder-id clamping in multimodal text embedding.

Speculative decoding fills padding/placeholder draft-token slots with the
sentinel ``PADDING_SLOT_ID = -1``. These ids can reach a multimodal model's
text embedding via ``SupportsMultiModal._embed_text_input_ids``; a negative id
is not a valid embedding index and triggers a CUDA
"vectorized gather kernel index out of bounds" assertion.

These tests verify the ids handed to the embedding lookup are always in range.

Run with: pytest tests/multimodal/test_text_embed_clamp_unit.py -v
"""

import torch

from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID


class _Stub:
    """Minimal stand-in carrying only the attribute the method reads."""

    def __init__(self, has_oov_mm_tokens: bool):
        self._has_oov_mm_tokens = has_oov_mm_tokens


def _capture_embed():
    """Return (embed_fn, seen) where seen[-1] is the last ids tensor passed."""
    seen: list[torch.Tensor] = []

    def embed_fn(ids: torch.Tensor) -> torch.Tensor:
        seen.append(ids.clone())
        # Mimic an embedding table lookup that would assert on a bad index.
        assert int(ids.min()) >= 0, "negative id reached the embedding lookup"
        return ids.float().unsqueeze(-1)

    return embed_fn, seen


def test_default_branch_clamps_negative_placeholder_ids():
    # Gemma4-style: image token is in-vocab so _has_oov_mm_tokens is False,
    # meaning is_multimodal-based masking does not run.
    input_ids = torch.tensor([5, PADDING_SLOT_ID, 7, PADDING_SLOT_ID])
    embed_fn, seen = _capture_embed()

    SupportsMultiModal._embed_text_input_ids(
        _Stub(has_oov_mm_tokens=False),
        input_ids,
        embed_fn,
        is_multimodal=None,
    )

    passed = seen[-1]
    assert int(passed.min()) >= 0
    # Placeholder slots are clamped to 0; real ids are untouched.
    assert passed.tolist() == [5, 0, 7, 0]
    # The caller's tensor must not be mutated.
    assert input_ids.tolist() == [5, PADDING_SLOT_ID, 7, PADDING_SLOT_ID]


def test_oov_branch_clamps_after_masking():
    # _has_oov_mm_tokens=True: multimodal positions are masked to 0, and any
    # remaining negative placeholder ids must still be clamped.
    input_ids = torch.tensor([5, 999, PADDING_SLOT_ID, 7])
    is_multimodal = torch.tensor([False, True, False, False])
    embed_fn, seen = _capture_embed()

    SupportsMultiModal._embed_text_input_ids(
        _Stub(has_oov_mm_tokens=True),
        input_ids,
        embed_fn,
        is_multimodal=is_multimodal,
    )

    passed = seen[-1]
    assert int(passed.min()) >= 0
    # idx1 masked by is_multimodal, idx2 clamped from the -1 placeholder.
    assert passed.tolist() == [5, 0, 0, 7]


def test_all_in_vocab_ids_unchanged():
    input_ids = torch.tensor([0, 1, 2, 3])
    embed_fn, seen = _capture_embed()

    SupportsMultiModal._embed_text_input_ids(
        _Stub(has_oov_mm_tokens=False),
        input_ids,
        embed_fn,
        is_multimodal=None,
    )

    assert seen[-1].tolist() == [0, 1, 2, 3]
