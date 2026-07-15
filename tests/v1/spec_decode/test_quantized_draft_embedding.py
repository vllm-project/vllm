# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the low-bit draft vocab embedding (A1c memory fix).

The MTP draft loads its own full-precision copy of a huge vocab embedding on the
last PP rank (Qwen3.5: vocab 248320 x hidden 5120 = 2.37 GiB fp16), which is what
pushes 27B+draft over 2x16 GiB. Because the draft is only a speculative proposer
(rejection sampling guarantees the final output equals target greedy regardless of
draft quality), a *lossy* low-bit draft embedding is correctness-safe: it only
lowers acceptance, never changes the output.

Crucially (RUN-A finding), the int storage must be allocated *at construction* and
the fp16 checkpoint weight quantized *as it loads* — so the full fp16 [V, H] tensor
never materializes on the GPU. Hence the primary constructor takes a shape, and
`load_full_weight` quantizes an fp16 weight into the pre-allocated int storage.
`from_weight` is a convenience (tests / post-load) that does both at once.

Pure-torch (CPU), no GPU / PP / model needed.
"""

import torch

from vllm.model_executor.layers.quantized_draft_embedding import (
    QuantizedVocabEmbedding,
)


def test_int8_lookup_matches_fp16_within_quant_error():
    """int8 per-row symmetric: lookup ~= fp16 lookup within one quant step."""
    torch.manual_seed(0)
    vocab, hidden = 1000, 64
    weight = torch.randn(vocab, hidden, dtype=torch.float16)

    qemb = QuantizedVocabEmbedding.from_weight(weight, bits=8)

    ids = torch.tensor([0, 5, 42, 999, 500], dtype=torch.long)
    ref = torch.nn.functional.embedding(ids, weight)
    out = qemb(ids)

    assert out.shape == ref.shape
    assert out.dtype == torch.float16
    max_err = (out.float() - ref.float()).abs().max().item()
    assert max_err < 0.03, f"int8 lookup error too large: {max_err}"


def test_int8_stores_one_byte_per_weight():
    """The win: int8 weight storage is half of fp16, plus a tiny per-row scale."""
    torch.manual_seed(0)
    vocab, hidden = 1000, 64
    weight = torch.randn(vocab, hidden, dtype=torch.float16)

    qemb = QuantizedVocabEmbedding.from_weight(weight, bits=8)

    assert qemb.qweight.dtype == torch.int8
    assert qemb.qweight.numel() == vocab * hidden
    assert qemb.scale.numel() == vocab


def test_int4_lookup_matches_fp16_within_quant_error():
    """int4 is coarser than int8 but still tracks the fp16 lookup."""
    torch.manual_seed(0)
    vocab, hidden = 1000, 64
    weight = torch.randn(vocab, hidden, dtype=torch.float16)

    qemb = QuantizedVocabEmbedding.from_weight(weight, bits=4)

    ids = torch.tensor([0, 5, 42, 999, 500], dtype=torch.long)
    ref = torch.nn.functional.embedding(ids, weight)
    out = qemb(ids)

    assert out.shape == ref.shape
    assert out.dtype == torch.float16
    max_err = (out.float() - ref.float()).abs().max().item()
    assert max_err < 0.5, f"int4 lookup error too large: {max_err}"
    assert max_err > 0.03, f"int4 unexpectedly precise: {max_err}"


def test_int4_packs_two_weights_per_byte():
    """The bigger win: int4 stores half a byte per weight (2 nibbles/byte)."""
    torch.manual_seed(0)
    vocab, hidden = 1000, 64
    weight = torch.randn(vocab, hidden, dtype=torch.float16)

    qemb = QuantizedVocabEmbedding.from_weight(weight, bits=4)

    assert qemb.qweight.dtype == torch.uint8
    assert qemb.qweight.numel() == vocab * hidden // 2
    assert qemb.scale.numel() == vocab


# --- Load-time construction (RUN-A: never materialize fp16 [V, H] on GPU) ------


def test_construct_allocates_int_storage_without_full_fp16():
    """Constructing from a shape pre-allocates int storage (no fp16 [V, H]).

    The `weight` entry-point for the checkpoint loader must NOT be a full
    fp16 [V, H] parameter (that is the OOM peak); it is an empty placeholder.
    """
    vocab, hidden = 200, 32
    qemb = QuantizedVocabEmbedding(vocab, hidden, bits=4)

    assert qemb.qweight.dtype == torch.uint8
    assert qemb.qweight.numel() == vocab * hidden // 2
    assert qemb.scale.numel() == vocab
    # The loader placeholder must not hold a full fp16 table.
    assert qemb.weight.numel() < vocab * hidden


def test_load_full_weight_quantizes_into_preallocated_storage():
    """load_full_weight fills the pre-allocated int storage from an fp16 table."""
    torch.manual_seed(0)
    vocab, hidden = 200, 32
    qemb = QuantizedVocabEmbedding(vocab, hidden, bits=4)

    weight = torch.randn(vocab, hidden, dtype=torch.float16)
    qemb.load_full_weight(weight)

    ids = torch.tensor([0, 7, 199], dtype=torch.long)
    ref = torch.nn.functional.embedding(ids, weight)
    out = qemb(ids)
    assert (out.float() - ref.float()).abs().max().item() < 0.5


def test_weight_loader_attr_routes_to_quantization():
    """The `weight` placeholder carries a weight_loader that quantizes the
    incoming fp16 checkpoint tensor (the AutoWeightsLoader entry point)."""
    torch.manual_seed(0)
    vocab, hidden = 200, 32
    qemb = QuantizedVocabEmbedding(vocab, hidden, bits=8)

    loader = qemb.weight.weight_loader
    weight = torch.randn(vocab, hidden, dtype=torch.float16)
    loader(qemb.weight, weight)

    ids = torch.tensor([1, 50, 150], dtype=torch.long)
    ref = torch.nn.functional.embedding(ids, weight)
    out = qemb(ids)
    assert (out.float() - ref.float()).abs().max().item() < 0.03
