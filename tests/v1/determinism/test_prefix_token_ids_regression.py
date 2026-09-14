# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the prefill-prefix construction used by
``test_decode_logprobs_match_prefill_logprobs`` in ``test_batch_invariance.py``.

The decode/prefill consistency test verifies, for each decoded token at
position ``i``, that running prefill on ``prompt + decode_tokens[:i]``
reproduces the same logprob for ``decode_tokens[i]`` that decode produced.

A previous implementation reconstructed the prefill prefix from text by
running ``llm.generate`` to detokenize the partial output and then passing
``prompt + partial_text`` back as a string prompt. vLLM internally re-encoded
that string through the model's tokenizer, which silently changes the token
sequence for many BPE tokenizers because ``tokenizer.encode(tokenizer.decode(ids))``
is not lossless. That broke the underlying invariant of the test (the
prefill prefix was no longer the same token sequence as the decode prefix)
and produced sporadic false mismatches.

These tests pin down two properties so the issue does not silently regress:

1. There exist real-world inputs/tokenizers where ``encode(decode(ids)) != ids``.
   Future "simplification" PRs that reintroduce the text-based round trip must
   contend with this evidence.
2. Building the prefix directly with ``prompt_token_ids + decode_tokens[:i]``
   trivially preserves the token sequence and is the implementation that
   ``test_decode_logprobs_match_prefill_logprobs`` relies on.

The tests are intentionally tokenizer-only so they run quickly on CPU in CI
and do not require a GPU, model weights, or batch-invariant kernels.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.cpu_test


@pytest.fixture(scope="module")
def qwen_coder_tokenizer():
    """A tokenizer that exhibits BPE merges across the decode/encode boundary.

    Skips if ``transformers`` (or the cached tokenizer files) are unavailable
    in the current environment.
    """
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        )
    except Exception as exc:  # pragma: no cover - depends on local cache/network
        pytest.skip(f"Tokenizer not available offline: {exc}")


def _encode(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False)


def test_decode_then_encode_can_change_token_ids(qwen_coder_tokenizer):
    """The decode -> encode round trip is not lossless for some BPE tokenizers.

    With Qwen2.5-Coder, the adjacent tokens ``'.'`` and ``'#'`` get merged into
    a single ``'.#'`` token after detokenization. Reconstructing a prefill
    prefix via text therefore produces a *different* token sequence than the
    decode path used.
    """
    tok = qwen_coder_tokenizer

    prompt = "Yesterday I went to the store and bought a new toy."
    continuation = "#1."

    prompt_ids = _encode(tok, prompt)
    continuation_ids = _encode(tok, continuation)
    original_ids = prompt_ids + continuation_ids

    text = tok.decode(original_ids)
    re_ids = _encode(tok, text)

    # The round trip must change the token sequence on this input; otherwise
    # the regression scenario this test is guarding against would not exist.
    assert original_ids != re_ids, (
        "Expected `encode(decode(ids)) != ids` for this BPE tokenizer/input. "
        f"original={original_ids}, round_trip={re_ids}"
    )


def test_token_id_prefix_is_preserved_without_round_trip(qwen_coder_tokenizer):
    """Building a prefix directly from token ids preserves the sequence.

    This mirrors the construction used by
    ``test_decode_logprobs_match_prefill_logprobs``:
    ``prefix_token_ids = prompt_token_ids + decode_tokens[:i]``.
    """
    tok = qwen_coder_tokenizer

    prompt = "Yesterday I went to the store and bought a new toy."
    continuation = "#1."

    prompt_ids = _encode(tok, prompt)
    decode_tokens = _encode(tok, continuation)

    for i in range(len(decode_tokens) + 1):
        prefix_token_ids = prompt_ids + decode_tokens[:i]

        # The first ``len(prompt_ids)`` ids are exactly the original prompt.
        assert prefix_token_ids[: len(prompt_ids)] == prompt_ids
        # The remaining ids are the first ``i`` decode tokens, unchanged.
        assert prefix_token_ids[len(prompt_ids) :] == decode_tokens[:i]
