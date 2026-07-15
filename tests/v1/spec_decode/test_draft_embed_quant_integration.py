# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration of the low-bit draft vocab embedding (A1c) into the spec config.

The numerical core (QuantizedVocabEmbedding) is covered by
test_quantized_draft_embedding.py. Here we test the SpeculativeConfig knob that
selects the draft embedding bit-width. The actual swap happens at draft model
*construction* (qwen3_5_mtp.py builds a QuantizedVocabEmbedding when the knob is
set) so the fp16 table never materializes on the GPU — that wiring is validated
end-to-end on real hardware (RUN-A and follow-ups), since it needs a built model.

CPU-only: config plumbing, no download / distributed init.
"""

import pytest

from vllm.config.speculative import SpeculativeConfig


def test_verify_draft_embed_quant_bits_defaults_none():
    """Unset -> None (no quantization, full-precision draft embed)."""
    assert SpeculativeConfig._verify_draft_embed_quant_bits(None) is None


def test_verify_draft_embed_quant_bits_allows_8_and_4():
    assert SpeculativeConfig._verify_draft_embed_quant_bits(8) == 8
    assert SpeculativeConfig._verify_draft_embed_quant_bits(4) == 4


def test_verify_draft_embed_quant_bits_rejects_other_values():
    """Only 4 or 8 (or None) are supported bit-widths."""
    with pytest.raises(ValueError):
        SpeculativeConfig._verify_draft_embed_quant_bits(3)
    with pytest.raises(ValueError):
        SpeculativeConfig._verify_draft_embed_quant_bits(16)
