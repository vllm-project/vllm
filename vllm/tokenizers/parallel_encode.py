# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tokenize a long prompt in pieces that are encoded in parallel.

A Hugging Face fast tokenizer encodes a single text on one CPU thread. When it
gets a list of texts, the Rust `tokenizers` library encodes them on all CPU
threads. For the byte-level BPE tokenizers whose pre-tokenizer regex is in
`_SPLIT_REGEXES`, a long text can be cut at places where the pre-tokenizer
always ends one pre-token and starts the next one. The token IDs of the pieces,
put one after the other, are then the token IDs of the whole text.
"""

import itertools
import json
import weakref
from collections.abc import Mapping
from typing import Any, Final

import regex as re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import TokenizersBackend

from vllm.logger import init_logger

from .protocol import TokenizerLike

logger = init_logger(__name__)

# A prompt with fewer characters than this goes through the normal tokenizer
# call. That is about 65k tokens of English text or code, which the normal
# call encodes in about 50 ms, so a shorter prompt saves little.
MIN_PARALLEL_ENCODE_CHARS: Final = 1 << 18

# Each piece ends at the first cut place after this many characters.
PARALLEL_ENCODE_PIECE_CHARS: Final = 1 << 16

# The places where a text may be cut: before a space, or after a carriage
# return or a line feed, where an ASCII letter or digit follows. With every
# regex in `_SPLIT_REGEXES`, a pre-token ends at such a place and the next one
# starts there. A space in front of a letter or digit always starts a
# pre-token, and no alternative of the regexes runs from a line break into a
# letter or digit. The regexes also find the same pre-tokens in a piece as in
# the whole text. They have no lookbehind, and their one lookahead, `(?!\S)`
# after a run of whitespace, sees a space in the whole text where the piece
# ends. A run of whitespace that ends with a line break is matched by
# `\s*[\r\n]+` before that lookahead is tried.
_CUT_PATTERN: Final = re.compile(r"(?= [A-Za-z0-9])|(?<=[\r\n])(?=[A-Za-z0-9])")

# The pre-tokenizer regexes for which the places in `_CUT_PATTERN` are known to
# be pre-token boundaries.
_SPLIT_REGEXES: Final = frozenset(
    {
        # Qwen2, Qwen2.5 and Qwen3.
        (
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
            r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        ),
        # Qwen3.5 and later. Here a word can also contain combining marks.
        (
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
            r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        ),
    }
)

# The keyword arguments of the normal tokenizer call that this module handles.
_SUPPORTED_ENCODE_KWARGS: Final = frozenset(
    {"add_special_tokens", "truncation", "max_length"}
)

# A short text with cut places of both kinds next to special tokens, digits,
# a tab, a combining mark, CJK characters and an emoji. The normal call on this
# text must give the same token IDs as its pieces.
_SAMPLE_TEXT: Final = (
    "<|im_start|>user\nA b\n\nC 1 23\r\nx\ty e\u0301 \u4e2d\u6587 \U0001f600 7"
    "<|im_end|>\n"
)

# The result of `_supports_parallel_encode` for each tokenizer object. With
# weak keys, a new tokenizer at a reused memory address does not get an old
# result.
_SUPPORTED_TOKENIZERS: Final[weakref.WeakKeyDictionary[TokenizersBackend, bool]] = (
    weakref.WeakKeyDictionary()
)


def try_parallel_encode(
    tokenizer: TokenizerLike,
    text: str,
    encode_kwargs: Mapping[str, Any],
) -> list[int] | None:
    """Encode `text` as pieces in parallel when the tokenizer allows it.

    Args:
        tokenizer: The tokenizer of the renderer.
        text: The text prompt.
        encode_kwargs: The keyword arguments of the normal tokenizer call.

    Returns:
        The token IDs of `tokenizer(text, **encode_kwargs)`, or `None` when
        `text` has fewer than `MIN_PARALLEL_ENCODE_CHARS` characters or no
        place to cut, or when the tokenizer or the arguments are not
        supported. The caller then makes the normal call.

    """
    if len(text) < MIN_PARALLEL_ENCODE_CHARS:
        return None
    if not isinstance(tokenizer, TokenizersBackend):
        return None
    if not encode_kwargs.keys() <= _SUPPORTED_ENCODE_KWARGS:
        return None

    max_length = encode_kwargs.get("max_length")
    truncation = encode_kwargs.get("truncation")
    # The normal call also truncates when only `max_length` is given.
    if truncation is None:
        truncation = max_length is not None
    side = tokenizer.truncation_side
    if truncation is False:
        max_length = None
    elif (
        truncation is not True
        or not isinstance(max_length, int)
        or max_length <= 0
        or side not in ("left", "right")
    ):
        # A truncation strategy given by name and truncation to the default
        # length of the model are not handled.
        return None

    if not _supports_parallel_encode(tokenizer):
        return None

    pieces = _cut_into_pieces(text)
    if len(pieces) < 2:
        return None

    token_ids = _encode_pieces(tokenizer, pieces)

    # Like the normal call, keep the first or the last `max_length` token IDs,
    # depending on the truncation side of the tokenizer.
    if max_length is not None and len(token_ids) > max_length:
        if side == "left":
            del token_ids[:-max_length]
        else:
            del token_ids[max_length:]
    return token_ids


def _cut_into_pieces(text: str) -> list[str]:
    """Cut `text` into pieces of at least `PARALLEL_ENCODE_PIECE_CHARS`
    characters, each one ending at a place in `_CUT_PATTERN`. The last piece
    can be shorter."""
    starts = [0]
    while (
        match := _CUT_PATTERN.search(text, starts[-1] + PARALLEL_ENCODE_PIECE_CHARS)
    ) is not None:
        starts.append(match.start())
    return [text[a:b] for a, b in itertools.pairwise([*starts, len(text)])]


def _encode_pieces(tokenizer: TokenizersBackend, pieces: list[str]) -> list[int]:
    """Encode `pieces` in one call and return their token IDs one after the
    other."""
    # A list of texts goes to `encode_batch` of the Rust tokenizer, which
    # encodes the texts on all CPU threads. Special tokens are off because a
    # supported tokenizer adds none, which `_supports_parallel_encode` checks.
    encoding = tokenizer(pieces, add_special_tokens=False, return_attention_mask=False)
    return list(itertools.chain.from_iterable(encoding["input_ids"]))


def _supports_parallel_encode(tokenizer: TokenizersBackend) -> bool:
    """Whether pieces cut at `_CUT_PATTERN` give the token IDs of the whole
    text with `tokenizer`. Each tokenizer object is checked once."""
    supported = _SUPPORTED_TOKENIZERS.get(tokenizer)
    if supported is None:
        supported = _has_supported_config(tokenizer) and _matches_normal_call(tokenizer)
        _SUPPORTED_TOKENIZERS[tokenizer] = supported
        logger.debug(
            "Long prompts are %s in parallel pieces with the tokenizer %s.",
            "tokenized" if supported else "not tokenized",
            tokenizer.name_or_path,
        )
    return supported


def _has_supported_config(tokenizer: TokenizersBackend) -> bool:
    """Whether the configuration of the Rust tokenizer makes every place in
    `_CUT_PATTERN` a place where the token IDs of a text can be split."""
    backend = tokenizer.backend_tokenizer
    # A replacement backend, such as the fastokens shim, is not checked.
    if not isinstance(backend, Tokenizer):
        return False
    try:
        normalizer = _component_config(backend.normalizer)
        pre_tokenizer = _component_config(backend.pre_tokenizer) or {}
        post_processor = _component_config(backend.post_processor)
    except Exception:
        # A component that is written in Python cannot be serialized.
        logger.debug("Cannot read the tokenizer configuration.", exc_info=True)
        return False

    # NFC never joins a space, an ASCII letter or an ASCII digit with the
    # character in front of it, so it does not change the text across a cut
    # place. Other normalizers are not checked.
    if normalizer is not None and normalizer.get("type") != "NFC":
        return False

    steps = pre_tokenizer.get("pretokenizers") or []
    if pre_tokenizer.get("type") != "Sequence" or len(steps) != 2:
        return False
    split, byte_level = steps
    if (
        split.get("type") != "Split"
        or (split.get("pattern") or {}).get("Regex") not in _SPLIT_REGEXES
        or split.get("behavior") != "Isolated"
        or split.get("invert")
    ):
        return False
    # The byte-level step must not add a space at the start of each piece and
    # must not split the text again with its own regex.
    if (
        byte_level.get("type") != "ByteLevel"
        or byte_level.get("add_prefix_space")
        or byte_level.get("use_regex")
    ):
        return False

    # BPE dropout picks merges at random, so the same text can get other IDs.
    if not isinstance(backend.model, BPE) or backend.model.dropout:
        return False

    # A post-processor that adds special tokens would add them to every piece.
    # The byte-level post-processor only changes offsets.
    if post_processor is not None and post_processor.get("type") != "ByteLevel":
        return False

    # Added tokens are found in the raw text before the pre-tokenizer runs.
    # A cut place inside the content of an added token could split one of
    # its occurrences, and an added token that strips whitespace or needs
    # word boundaries looks at the text on the other side of a cut place.
    # The search starts at the second character, because a cut place in front
    # of an added token does not split it.
    for token in backend.get_added_tokens_decoder().values():
        if (
            token.lstrip
            or token.rstrip
            or token.single_word
            or _CUT_PATTERN.search(token.content, 1) is not None
        ):
            return False
    return True


def _component_config(component: Any) -> dict[str, Any] | None:
    """Return the configuration of a normalizer, pre-tokenizer or
    post-processor of a Rust tokenizer, as it is written in `tokenizer.json`.
    Unlike `Tokenizer.to_str`, this does not write the vocabulary, which takes
    about 100 ms for a Qwen tokenizer."""
    if component is None:
        return None
    return json.loads(component.__getstate__())


def _matches_normal_call(tokenizer: TokenizersBackend) -> bool:
    """Whether the normal call on `_SAMPLE_TEXT` gives the token IDs of the
    sample cut at every place in `_CUT_PATTERN`, with and without special
    tokens. This catches a tokenizer class that changes the text or adds
    tokens in Python before the Rust tokenizer runs."""
    cuts = [match.start() for match in _CUT_PATTERN.finditer(_SAMPLE_TEXT)]
    pieces = [
        _SAMPLE_TEXT[a:b]
        for a, b in itertools.pairwise([0, *cuts, len(_SAMPLE_TEXT)])
        if a < b
    ]
    expected = _encode_pieces(tokenizer, pieces)
    return all(
        list(tokenizer(_SAMPLE_TEXT, add_special_tokens=add)["input_ids"]) == expected
        for add in (True, False)
    )
