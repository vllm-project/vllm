# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Word ``probability`` against the real Whisper tokenizer and logprob types.

``test_transcription_granularity_gating`` pins the arithmetic with a fake
tokenizer whose every word is one token. What it cannot reach is how the real
Whisper BPE splits text: words spread over several tokens, punctuation that is
its own token but belongs to the preceding word, and the real ids of timestamp
tokens (``>= <|0.00|>``) and eot. Those decide which probabilities land in which
word, so they are tested here with the real tokenizer (no model weights, CPU)
and with both containers the engine can hand back (``list[dict]`` and
``FlatLogprobs``).
"""

import math
from types import SimpleNamespace

import pytest

from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing
from vllm.logprobs import Logprob, create_sample_logprobs
from vllm.tokenizers import get_tokenizer

MODEL_NAME = "openai/whisper-large-v3-turbo"
TEXT = " Unbelievably, it's Nemotron!"
# How the real tokenizer splits TEXT. Asserted first, so tokenizer drift fails
# with this message instead of a confusing probability mismatch below.
PIECES = [" Un", "believably", ",", " it", "'s", " Nem", "ot", "ron", "!"]
# _group_words starts a word at each leading-space piece; punctuation pieces
# have none, so they stay with the word before them.
WORDS = {"Unbelievably,": [0, 1, 2], "it's": [3, 4], "Nemotron!": [5, 6, 7, 8]}
# The tokens each word's probability averages: the trailing "," and "!" stay in
# the text but are left out of the score. "'s" carries letters, so it counts.
SCORED = {"Unbelievably,": [0, 1], "it's": [3, 4], "Nemotron!": [5, 6, 7]}
# One distinct probability per text token, so a token counted in the wrong
# word changes that word's mean.
TEXT_PROBS = [0.91, 0.42, 0.77, 0.99, 0.63, 0.55, 0.88, 0.30, 0.12]
# A timestamp token and eot follow the text, as in a real Whisper decode. Their
# probabilities would visibly move any mean they leaked into.
NON_TEXT_PROB = 1e-4
PROMPT_POSITIONS = 4  # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL_NAME)


@pytest.fixture(scope="module")
def decode(tokenizer):
    ids = tokenizer.encode(TEXT, add_special_tokens=False)
    assert [tokenizer.decode([i]) for i in ids] == PIECES, (
        "the Whisper tokenizer no longer splits the fixture text as expected; "
        "update PIECES/WORDS before trusting the probability assertions"
    )
    timestamp = tokenizer.convert_tokens_to_ids("<|2.00|>")
    eot = tokenizer.eos_token_id
    assert timestamp >= tokenizer.convert_tokens_to_ids("<|0.00|>")
    assert eot in set(tokenizer.all_special_ids)
    token_ids = ids + [timestamp, eot]
    # One onset per decoder position: the prompt, then each generated token.
    onsets = [0.0] * PROMPT_POSITIONS + [0.2 * (k + 1) for k in range(len(token_ids))]
    return token_ids, onsets


def _logprobs(token_ids, probs, flat):
    container = create_sample_logprobs(flat)
    for tid, p in zip(token_ids, probs):
        container.append({tid: Logprob(logprob=math.log(p), rank=1)})
    return container


def _group(tokenizer, token_ids, onsets, log_probs=None, offset=0.0):
    return SpeechToTextBaseServing._group_words(
        SimpleNamespace(tokenizer=tokenizer), token_ids, onsets, offset, log_probs
    )


@pytest.mark.parametrize("flat", [False, True], ids=["list", "FlatLogprobs"])
def test_word_probability_follows_real_bpe_word_boundaries(tokenizer, decode, flat):
    token_ids, onsets = decode
    probs = TEXT_PROBS + [NON_TEXT_PROB, NON_TEXT_PROB]
    words = _group(tokenizer, token_ids, onsets, _logprobs(token_ids, probs, flat))

    assert [w.word for w in words] == list(WORDS)
    expected = [
        sum(TEXT_PROBS[i] for i in members) / len(members)
        for members in SCORED.values()
    ]
    assert [w.probability for w in words] == pytest.approx(expected, abs=1e-9)


def test_logprobs_do_not_move_word_times(tokenizer, decode):
    """Probability is additive: the same words, at the same times, with or
    without logprobs -- including the chunk offset."""
    token_ids, onsets = decode
    probs = TEXT_PROBS + [NON_TEXT_PROB, NON_TEXT_PROB]
    with_probs = _group(
        tokenizer, token_ids, onsets, _logprobs(token_ids, probs, False), 30.0
    )
    without = _group(tokenizer, token_ids, onsets, None, 30.0)
    assert [(w.word, w.start, w.end) for w in with_probs] == [
        (w.word, w.start, w.end) for w in without
    ]
    assert with_probs[0].start >= 30.0
    assert all(w.probability is None for w in without)


def test_a_token_missing_from_its_position_is_skipped_not_zeroed(tokenizer, decode):
    """A position whose dict lacks the sampled id contributes nothing: the
    word averages its remaining tokens, and a word with none gets ``None``."""
    token_ids, onsets = decode
    container = create_sample_logprobs(False)
    for i, (tid, p) in enumerate(
        zip(token_ids, TEXT_PROBS + [NON_TEXT_PROB, NON_TEXT_PROB])
    ):
        if i == 1:  # "believably" -> logprob recorded under a different id
            container.append({tid + 1: Logprob(logprob=math.log(p))})
        elif i in (3, 4):  # every token of "it's" missing
            container.append({})
        else:
            container.append({tid: Logprob(logprob=math.log(p))})
    words = _group(tokenizer, token_ids, onsets, container)

    # "believably" is missing and "," is punctuation: only " Un" is left.
    assert words[0].probability == pytest.approx(TEXT_PROBS[0])
    assert words[1].probability is None
    assert words[2].probability == pytest.approx(sum(TEXT_PROBS[5:8]) / 3)


@pytest.mark.parametrize(
    "text, pieces, probs, expected",
    [
        # A hesitant ',"' does not drag down a confident word, and the leading
        # quote is its own token with a space: also left out.
        (
            ' "Teeth," he',
            [' "', "Te", "eth", ',"', " he"],
            [0.2, 0.999, 0.999, 0.485, 0.9],
            {'"Teeth,"': 0.999, "he": 0.9},
        ),
        # Internal punctuation stays in: "'t" is part of the word.
        (
            " I don't know...",
            [" I", " don", "'t", " know", "..."],
            [0.9, 0.8, 0.4, 0.7, 0.1],
            {"I": 0.9, "don't": 0.6, "know...": 0.7},
        ),
        # A word that is only punctuation is still scored, on all its tokens.
        (
            " (well) -",
            [" (", "well", ")", " -"],
            [0.5, 0.9, 0.3, 0.25],
            {"(well)": 0.9, "-": 0.25},
        ),
    ],
)
def test_edge_punctuation_is_left_out_of_the_word_probability(
    tokenizer, text, pieces, probs, expected
):
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert [tokenizer.decode([i]) for i in ids] == pieces
    # Close the decode as a real one does: a timestamp token, then eot.
    token_ids = ids + [
        tokenizer.convert_tokens_to_ids("<|2.00|>"),
        tokenizer.eos_token_id,
    ]
    onsets = [0.0] * PROMPT_POSITIONS + [0.2 * (k + 1) for k in range(len(token_ids))]
    log_probs = _logprobs(token_ids, probs + [NON_TEXT_PROB, NON_TEXT_PROB], False)
    words = _group(tokenizer, token_ids, onsets, log_probs)
    assert {w.word: w.probability for w in words} == pytest.approx(expected)
