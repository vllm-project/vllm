# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from functools import lru_cache
from itertools import groupby

from sentencepiece import sentencepiece_pb2
from tokenizers.decoders import ByteLevel
from transformers.convert_slow_tokenizer import bytes_to_unicode

from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.fastokens import get_fastokens_decoder_config


@lru_cache(maxsize=1)
def _byte_level_decoder() -> dict[str, int]:
    return {char: byte for byte, char in bytes_to_unicode().items()}


def decode_token_ids_with_offsets(
    tokenizer: TokenizerLike, token_ids: list[int], *, skip_special_tokens: bool = False
) -> tuple[str, list[int]] | None:
    """Return backend-native prompt decoding and character starts if supported.

    Callers must check that the text matches the echo's decoding policy.
    """
    if skip_special_tokens:
        special_ids = set(tokenizer.all_special_ids)
        visible = [i for i, tid in enumerate(token_ids) if tid not in special_ids]
        decoded = decode_token_ids_with_offsets(
            tokenizer, [token_ids[i] for i in visible]
        )
        if decoded is None:
            return None
        text, starts = decoded
        offsets = [len(text)] * len(token_ids)
        for i, start in zip(visible, starts):
            offsets[i] = start
        for i in range(len(offsets) - 2, -1, -1):
            if token_ids[i] in special_ids:
                offsets[i] = offsets[i + 1]
        return text, offsets

    decode_with_offsets = getattr(tokenizer, "decode_prompt_with_offsets", None)
    if callable(decode_with_offsets):
        return decode_with_offsets(token_ids)

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or backend.decoder is None:
        return None

    byte_fallback_config = None
    if not isinstance(backend.decoder, ByteLevel):
        try:
            config = json.loads(backend.decoder.__getstate__())
        except (AttributeError, TypeError, ValueError):
            config = get_fastokens_decoder_config(backend)
        if config is None:
            return None
        steps = config["decoders"] if config["type"] == "Sequence" else [config]
        if not (
            steps
            and steps[0]["type"] == "ByteLevel"
            and all(step["type"] == "Fuse" for step in steps[1:])
        ):
            if not any(step["type"] == "ByteFallback" for step in steps):
                return None
            byte_fallback_config = config

    tokens = [backend.id_to_token(token_id) for token_id in token_ids]
    if any(token is None for token in tokens):
        return None
    if byte_fallback_config is not None:
        return _decode_byte_fallback_with_offsets(tokens, byte_fallback_config)

    byte_decoder = _byte_level_decoder()
    pieces: list[bytes] = []
    for token in tokens:
        try:
            piece = bytes(byte_decoder[char] for char in token)
        except KeyError:
            # ByteLevel decodes a token literally if any character is outside
            # its alphabet (for example an added Unicode token).
            piece = token.encode("utf-8")
        pieces.append(piece)

    return decode_utf8_with_offsets(pieces)


def decode_utf8_with_offsets(pieces: list[bytes]) -> tuple[str, list[int]]:
    """Decode UTF-8 pieces, retaining starts inside split characters."""
    text = b"".join(pieces).decode("utf-8", errors="replace")
    return text, _utf8_token_offsets(pieces, text)


def decode_sentencepiece_with_offsets(
    sp_model, token_ids: list[int]
) -> tuple[str, list[int]]:
    proto = sentencepiece_pb2.SentencePieceText.FromString(
        sp_model.decode_ids_as_serialized_proto(token_ids)
    )
    byte_to_char = [0]
    for index, char in enumerate(proto.text):
        byte_to_char.extend([index] * (len(char.encode("utf-8")) - 1))
        byte_to_char.append(index + 1)
    return proto.text, [byte_to_char[piece.begin] for piece in proto.pieces]


def _decode_byte_fallback_with_offsets(
    tokens: list[str], config: dict
) -> tuple[str, list[int]] | None:
    steps = config["decoders"] if config["type"] == "Sequence" else [config]
    fallback = next(
        (i for i, step in enumerate(steps) if step["type"] == "ByteFallback"), None
    )
    if fallback is None:
        return None
    for step in steps[:fallback]:
        if step["type"] != "Replace" or "String" not in step["pattern"]:
            return None
        tokens = [
            token.replace(step["pattern"]["String"], step["content"])
            for token in tokens
        ]
    after = steps[fallback + 1 :]
    if after:
        if after[0]["type"] != "Fuse":
            return None
        after = after[1:]
    if any(step["type"] != "Strip" for step in after):
        return None

    parts: list[str] = []
    offsets: list[int] = []
    position = 0
    for is_byte, group in groupby(tokens, _is_byte_fallback_token):
        pieces = list(group)
        if not is_byte:
            for token in pieces:
                offsets.append(position)
                parts.append(token)
                position += len(token)
            continue
        byte_pieces = [bytes([int(token[3:5], 16)]) for token in pieces]
        try:
            text = b"".join(byte_pieces).decode("utf-8")
        except UnicodeDecodeError:
            # ByteFallback replaces every byte in an invalid run separately.
            text = "\ufffd" * len(byte_pieces)
            starts = list(range(len(byte_pieces)))
        else:
            starts = _utf8_token_offsets(byte_pieces, text)
        offsets.extend(position + start for start in starts)
        parts.append(text)
        position += len(text)
    text = "".join(parts)
    for step in after:
        content = step["content"]
        if not content:
            return None
        left, right = 0, len(text)
        for _ in range(step["start"]):
            if text.startswith(content, left, right):
                left += len(content)
        for _ in range(step["stop"]):
            if text.endswith(content, left, right):
                right -= len(content)
        text = text[left:right]
        offsets = [max(0, min(start - left, len(text))) for start in offsets]
    return text, offsets


def _is_byte_fallback_token(token: str) -> bool:
    return (
        len(token) == 6
        and token.startswith("<0x")
        and token.endswith(">")
        and all(char in "0123456789abcdefABCDEF" for char in token[3:5])
    )


def _utf8_token_offsets(pieces: list[bytes], text: str) -> list[int]:
    raw = b"".join(pieces)
    offsets: list[int] = []
    byte_offset = token_start = token_index = 0
    for char_offset, char in enumerate(text):
        if char == "\ufffd" and not raw.startswith(b"\xef\xbf\xbd", byte_offset):
            try:
                raw[byte_offset : byte_offset + 4].decode("utf-8")
            except UnicodeDecodeError as error:
                byte_offset += error.end
        else:
            byte_offset += len(char.encode("utf-8"))
        while token_index < len(pieces) and token_start < byte_offset:
            offsets.append(char_offset)
            token_start += len(pieces[token_index])
            token_index += 1
    offsets.extend([len(text)] * (len(pieces) - token_index))
    return offsets


def _replace_none_with_empty(tokens: list[str | None]):
    for i, token in enumerate(tokens):
        if token is None:
            tokens[i] = ""


def _convert_tokens_to_string_with_added_encoders(
    tokenizer: TokenizerLike,
    output_tokens: list[str],
    skip_special_tokens: bool,
    spaces_between_special_tokens: bool,
) -> str:
    # Adapted from
    # https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/tokenization_utils.py#L921
    # NOTE(woosuk): The following code is slow because it runs a for loop over
    # the output_tokens. In Python, running a for loop over a list can be slow
    # even when the loop body is very simple.
    # Performance improvements: avoid repeated attribute and function lookups;
    # localize frequently used objects;

    sub_texts: list[str] = []
    current_sub_text: list[str] = []
    convert_tokens_to_string = tokenizer.convert_tokens_to_string
    added_vocab_set = set(tokenizer.get_added_vocab())
    all_special_tokens = (
        set(tokenizer.all_special_tokens) if skip_special_tokens else ()
    )

    for token in output_tokens:
        # Use precomputed set for skip-special check
        if token in all_special_tokens:
            continue
        if token in added_vocab_set:
            if current_sub_text:
                sub_texts.append(convert_tokens_to_string(current_sub_text))
                current_sub_text.clear()
            sub_texts.append(token)
        else:
            current_sub_text.append(token)
    if current_sub_text:
        sub_texts.append(convert_tokens_to_string(current_sub_text))
    if spaces_between_special_tokens:
        return " ".join(sub_texts)
    return "".join(sub_texts)


# 5 is an arbitrary value that should work for all
# tokenizers (bigger = more conservative).
INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET = 5


_CACHED_MARKER_KEY = "_vllm_space_marker_cache"
_NOT_CACHED = "__not_computed__"


def get_leading_space_marker(tokenizer: TokenizerLike) -> str | None:
    """Read the space marker from the tokenizer's pre_tokenizer config.

    Only Metaspace pre_tokenizers (used by SentencePiece-based models like
    Llama, Mistral, T5) have a replacement character whose leading instance
    gets stripped by decode(). ByteLevel (GPT-2), BertPreTokenizer (BERT),
    and others do not have this issue.

    Returns the marker character, or None if decode() is safe for single
    tokens.
    """
    cached = getattr(tokenizer, _CACHED_MARKER_KEY, _NOT_CACHED)
    if cached is not _NOT_CACHED:
        return cached  # type: ignore[return-value]

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        result = None
    else:
        result = None
        try:
            config = json.loads(backend.to_str())
        except Exception:
            pass
        else:
            pre = config.get("pre_tokenizer") or {}
            pre_type = pre.get("type")
            if pre_type == "Metaspace":
                result = pre.get("replacement", "▁")
            elif pre_type == "Sequence":
                for sub in pre.get("pretokenizers", []):
                    if sub.get("type") == "Metaspace":
                        result = sub.get("replacement", "▁")
                        break

    setattr(tokenizer, _CACHED_MARKER_KEY, result)
    return result


def _restore_leading_spaces(raw_token: str, token_str: str, marker: str) -> str:
    """Restore leading spaces that decode() stripped from a raw vocab piece."""
    num_markers = 0
    for ch in raw_token:
        if ch != marker:
            break
        num_markers += 1
    if num_markers == 0:
        return token_str
    existing = len(token_str) - len(token_str.lstrip(" "))
    missing = num_markers - existing
    return " " * missing + token_str if missing > 0 else token_str


def convert_prompt_ids_to_tokens(
    tokenizer: TokenizerLike,
    prompt_ids: list[int],
    skip_special_tokens: bool = False,
) -> tuple[list[str], int, int]:
    """Converts the prompt ids to tokens and returns the tokens and offsets
    for incremental detokenization.

    Note that not all tokens are converted to strings. Only the tokens that
    are necessary for incremental detokenization are converted to strings.
    """
    # We do not need to convert the whole prompt to tokens.
    # Offset a little more in case we have special tokens.
    new_tokens = tokenizer.convert_ids_to_tokens(
        prompt_ids[-INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET - 2 :],
        skip_special_tokens=skip_special_tokens,
    )
    read_offset = len(new_tokens)
    prefix_offset = max(read_offset - INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET, 0)
    # This is required to guard against out-of-vocab prompt token ids
    _replace_none_with_empty(new_tokens)  # type: ignore[arg-type]
    return new_tokens, prefix_offset, read_offset


def convert_ids_list_to_tokens(
    tokenizer: TokenizerLike,
    token_ids: list[int],
) -> list[str]:
    """Detokenize the input ids individually.

    Uses decode() for human-readable output, then checks the raw vocab
    piece via convert_ids_to_tokens() to restore any leading spaces that
    decode() stripped (SentencePiece add_dummy_prefix inverse).

    Args:
      tokenizer: tokenizer used by model under test
      token_ids: convert these tokens (Python list form)

    Returns:
      Python list of token string representations

    """
    if not token_ids:
        return []
    marker = get_leading_space_marker(tokenizer)
    if marker is None:
        return [tokenizer.decode([tid]) or "" for tid in token_ids]
    raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return [
        _restore_leading_spaces(raw, tokenizer.decode([tid]) or "", marker)
        for tid, raw in zip(token_ids, raw_tokens)
    ]


# Based on
# https://github.com/huggingface/text-generation-inference/blob/v0.9.4/server/text_generation_server/models/model.py#L62C9-L62C15
# under Apache 2.0 license
def detokenize_incrementally(
    tokenizer: TokenizerLike,
    all_input_ids: list[int],
    prev_tokens: list[str] | None,
    prefix_offset: int,
    read_offset: int,
    skip_special_tokens: bool = False,
    spaces_between_special_tokens: bool = True,
) -> tuple[list[str], str, int, int]:
    """Detokenizes the input ids incrementally and returns the new tokens
    and the new text.

    If `prev_tokens` is None, this function will convert the input ids to
    tokens and return the tokens and the new text. Otherwise, it will return the
    new tokens and the new text.

    This function will also return the new prefix offset and the new read
    offset to be used in the next iteration.

    The offsets are necessary to defeat cleanup algorithms in the decode which
    decide to add a space or not depending on the surrounding ids.

    Args:
        tokenizer: The tokenizer to use.
        all_input_ids: The input ids. The last id is the new token id.
        prev_tokens: The previous tokens. If None, this function will convert
            the input ids to tokens and return the tokens and the new text.
        prefix_offset: The prefix offset.
        read_offset: The read offset.
        skip_special_tokens: Whether to skip special tokens.
        spaces_between_special_tokens: Whether to add spaces between special
            tokens.

    """
    new_token_id = all_input_ids[-1]
    # This is the first iteration for this sequence
    is_first_iter = prev_tokens is None
    if is_first_iter:
        (prev_tokens, prefix_offset, read_offset) = convert_prompt_ids_to_tokens(
            tokenizer, all_input_ids[:-1], skip_special_tokens=skip_special_tokens
        )
    assert prev_tokens is not None

    # If the new token id is out of bounds, return an empty string.
    if 0 <= new_token_id < len(tokenizer):
        # Put new_token_id in a list so skip_special_tokens is respected
        new_tokens = tokenizer.convert_ids_to_tokens(
            [new_token_id], skip_special_tokens=skip_special_tokens
        )
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]
        else:
            # This is required to guard against out-of-vocab prompt token ids
            # (for example when using dummy weights)
            _replace_none_with_empty(new_tokens)  # type: ignore[arg-type]
    else:
        new_tokens = [""]
    output_token_count = len(prev_tokens) + len(new_tokens)
    window_start = (
        0 if is_first_iter else min(prefix_offset, read_offset, len(prev_tokens))
    )
    output_tokens = (
        prev_tokens[window_start:] if window_start else prev_tokens
    ) + new_tokens

    # If this is the first iteration, return all tokens.
    if is_first_iter:
        new_tokens = output_tokens

    # The prefix text is necessary only to defeat cleanup algorithms in
    # the decode which decide to add a space or not depending on the
    # surrounding ids.
    if tokenizer.is_fast or not tokenizer.get_added_vocab():
        prefix_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset - window_start : read_offset - window_start]
        )
        new_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset - window_start :]
        )
    else:
        prefix_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset - window_start : read_offset - window_start],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
        new_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset - window_start :],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

    if len(new_text) <= len(prefix_text) or new_text.endswith("�"):
        # utf-8 char at the end means it's a potential unfinished byte sequence
        # from byte fallback tokenization.
        # If it's in the middle, it's probably a real invalid id generated
        # by the model
        return new_tokens, "", prefix_offset, read_offset

    new_text = new_text[len(prefix_text) :]
    return new_tokens, new_text, read_offset, output_token_count
