# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from typing import TYPE_CHECKING

import regex as re
from cachetools import LRUCache
from diskcache import Cache

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils import LazyLoader

if TYPE_CHECKING:
    import outlines_core as oc
    import transformers.file_utils as file_utils
    import transformers.models.gpt2.tokenization_gpt2 as tokenization_gpt2

    from vllm.transformers_utils.tokenizer import AnyTokenizer
else:
    oc = LazyLoader("oc", globals(), "outlines_core")
    file_utils = LazyLoader("file_utils", globals(), "transformers.file_utils")
    tokenization_gpt2 = LazyLoader(
        "tokenization_gpt2",
        globals(),
        "transformers.models.gpt2.tokenization_gpt2",
    )

logger = init_logger(__name__)

CACHE = None


class OutlinesVocabulary:
    """
    Wrapper class for `outlines_core.Vocabulary`,
    which allows us to store a hash with the vocabulary
    """

    def __init__(self, vocabulary: oc.Vocabulary) -> None:
        # Actual vocabulary object
        self.inner = vocabulary
        # Have to do abs(hash()) because python hashes can
        # be negative, and we are using hash as a cache key.
        hex_str = hashlib.sha256(
            vocabulary.__repr__().encode('utf-8')).hexdigest()
        hash_int = int(hex_str, 16)
        self._hash = hash_int


def get_outlines_cache_path() -> str:
    """Get the context object that contains previously-computed return values"""
    outlines_cache_dir = os.getenv("OUTLINES_CACHE_DIR")
    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    home_dir = os.path.expanduser("~")

    if outlines_cache_dir:
        # OUTLINES_CACHE_DIR takes precedence
        return outlines_cache_dir
    elif xdg_cache_home:
        return os.path.join(xdg_cache_home, ".cache", "outlines")
    # If homedir is "/", we may be inside a container, and thus writing to
    # root would be problematic, so we fallback to using a tempfile.
    # Also validate the path exists, since os.path.expanduser does
    # not garuntee existence.
    elif os.path.isdir(home_dir) and home_dir != "/":
        # Default Unix fallback: ~/.cache/outlines
        return os.path.join(home_dir, ".cache", "outlines")
    else:
        import tempfile

        # home_dir may be / inside a docker container without existing user
        tempdir = tempfile.gettempdir()
        return os.path.join(tempdir, ".cache", "outlines")


def get_outlines_cache():
    """Get the Cache instance to be used for index caching"""

    cache_dir = get_outlines_cache_path()
    if envs.VLLM_V1_USE_OUTLINES_CACHE:
        logger.warning("Enabling outlines cache. This is an unbounded on-disk "
                       "cache. It may consume a lot of disk space and should "
                       "not be used with untrusted clients.")
        cache = Cache(cache_dir, eviction_policy="none", cull_limit=0)
        outlines_version = importlib.metadata.version("outlines_core")

        cached_version = cache.get('__version__', None)
        if cached_version != outlines_version:
            cache.clear()
        cache.set('__version__', outlines_version)
        return cache
    else:
        return LRUCache(maxsize=128)


re_llama_byte_token = re.compile(r"^<0x[0-9A-F]{2}>$")
re_replacement_seq = re.compile(r"^.{0,6}�+.{0,6}$")


def _reduced_vocabulary(
    tokenizer: AnyTokenizer,
    eos_token_id: int,
) -> dict[bytes, list[int]]:
    """Create a map from vocabulary tokens to lists of equivalent token ids.

    Returns:
        A Dict of token string -> equivalent token ids
    """

    unicode_to_bytes = {
        v: k
        for k, v in tokenization_gpt2.bytes_to_unicode().items()
    }

    def convert_token_to_string(token: str) -> str:

        string = tokenizer.convert_tokens_to_string([token])

        # A hack to handle missing spaces to HF's Llama tokenizers
        if (type(token) is str
                and token.startswith(file_utils.SPIECE_UNDERLINE)
                or token == "<0x20>"):
            return " " + string

        return string

    vocabulary: dict[bytes, list[int]] = {}
    empty_token_ids: list[int] = []
    for token, token_idx in tokenizer.get_vocab().items():
        if token in tokenizer.all_special_tokens:  # type: ignore
            continue

        token_str = convert_token_to_string(token)
        if token_str:
            if isinstance(token, (bytes, bytearray)):
                # For BPE tokenizers where tokens are stored as bytes.

                # safe to ignore since token_str is of type (bytearray, bytes)
                # by this point.
                token_bytes = bytes(token_str)  # type: ignore[arg-type]

            elif "\ufffd" in token_str and not re_replacement_seq.match(
                    token_str):
                # Handle tokens with invalid UTF-8 sequences.
                if re_llama_byte_token.match(token):
                    # Llama-like tokenizers use <0xXX> for incomplete sequences.
                    token_bytes = bytes([int(token[3:5], 16)])
                else:
                    # GPT2 tokenizers: map each byte back using unicode_to_bytes
                    byte_vals = [unicode_to_bytes.get(c) for c in token]
                    if None in byte_vals:
                        raise RuntimeError(
                            f"Cannot convert token `{token}`"
                            f" ({token_idx}) to bytes: {token_str}")
                    # safe to ignore, since if None in byte_vals,
                    # an error is thrown.
                    token_bytes = bytes(byte_vals)  # type: ignore[arg-type]
            else:
                token_bytes = token_str.encode('utf-8')

            if token_idx != eos_token_id:
                vocabulary.setdefault(token_bytes, []).append(token_idx)
        else:
            empty_token_ids.append(token_idx)

    return vocabulary


def get_outlines_vocabulary(tokenizer: AnyTokenizer) -> oc.Vocabulary:
    """Get the `Vocabulary` object for a given tokenizer.
    """
    if hasattr(tokenizer, "_outlines_vocabulary"):
        return tokenizer._outlines_vocabulary  # type: ignore

    try:
        if hasattr(
                tokenizer,
                "eos_token_id",
        ) and tokenizer.eos_token_id is not None:
            eos_token_id = tokenizer.eos_token_id
        else:
            raise ValueError(
                f"Error during structured outputs setup for outlines: Tokenizer ({type(tokenizer)}) has no `eos_token_id` property, but `eos_token_id` is required for structured outputs to work properly."  # noqa: E501
            )

        reduced_vocab = _reduced_vocabulary(
            tokenizer,
            eos_token_id  #type: ignore
        )
        vocabulary = OutlinesVocabulary(
            oc.Vocabulary(eos_token_id, reduced_vocab))
        tokenizer._outlines_vocabulary = vocabulary  # type: ignore

        return vocabulary
    except AttributeError as e:
        raise ValueError(f"Cannot get the vocabulary of the tokenizer "
                         f"({type(tokenizer)}). The tokenizer should have a "
                         "get_vocab method.") from e


def grammar_is_likely_lark(grammar_str: str) -> bool:
    """
    Check if grammar appears to use Lark syntax.

    Args:
        grammar_str: Input grammar string

    Returns:
        bool: True if grammar appears to be in Lark format, False otherwise

    Examples:
        >>> grammar_is_likely_lark("rule: 'abc'")
        True
        >>> grammar_is_likely_lark("rule ::= 'abc'")
        False
    """
    if not grammar_str or not isinstance(grammar_str, str):
        return False

    for line in grammar_str.split('\n'):
        # Remove both comment styles
        line = re.sub(r'(#|//).*$', '', line).strip()
        if not line:
            continue

        # Look for EBNF rule definition
        if '::=' in line:
            return False

    return True


def convert_lark_to_ebnf(grammar_str: str) -> str:
    """
    Convert a Lark grammar string to EBNF format.

    EBNF reference:
    https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
    Lark grammar reference:
    https://lark-parser.readthedocs.io/en/latest/grammar.html

    Args:
        grammar_str: Input grammar in Lark format

    Returns:
        str: Converted grammar in EBNF format

    Examples:
        >>> print(convert_lark_to_ebnf("rule: 'hello'"))
        root ::= rule
        rule ::= "hello"
    """
    if not isinstance(grammar_str, str):
        raise ValueError(f"Grammar must be a string, got {type(grammar_str)}")
    if not grammar_str.strip():
        raise ValueError("Grammar string cannot be empty")

    defined_rules = set()
    referenced_rules = set()
    output_lines = []

    def clean_line(line: str) -> str:
        """Remove comments and whitespace from line."""
        return re.sub(r'(#|//).*$', '', line).strip()

    def check_quotes(text: str, rule_name: str, line_num: int) -> None:
        """Validate quote matching in text."""
        if text.count("'") % 2 != 0 or text.count('"') % 2 != 0:
            raise ValueError(
                f"Mismatched quotes in {rule_name} on line {line_num}")

    def extract_references(text: str) -> set[str]:
        """Extract rule references from text."""
        # Remove quoted strings and special characters
        text = re.sub(r'"[^"]*"', '', text)
        text = re.sub(r'[+*?()|\[\]{}]', ' ', text)
        return set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text))

    # First pass: Find root rule and validate rule definitions
    lines = [clean_line(line) for line in grammar_str.split('\n')]
    first_rule = None

    for line_num, line in enumerate(lines, 1):
        if not line or line.startswith('|'):
            continue

        if ':' in line:
            try:
                name = line.split(':', 1)[0].strip().strip('?')
                defined_rules.add(name)
                if first_rule is None:
                    first_rule = name
                if name == 'start':
                    first_rule = 'start'
            except IndexError as e:
                raise ValueError(f"Invalid rule format on line {line_num}. "
                                 "Expected 'rule_name: definition'") from e

    if not defined_rules:
        raise ValueError("No valid rules found in grammar")

    # Add root rule
    output_lines.append(f"root ::= {first_rule}")

    # Second pass: Process rule definitions and alternatives
    current_rule = None
    current_definition = []

    for line_num, line in enumerate(lines, 1):
        if not line:
            continue

        try:
            if ':' in line and not line.startswith('|'):
                # Save previous rule if exists
                if current_rule:
                    output_lines.append(
                        f"{current_rule} ::= {' | '.join(current_definition)}")

                # Process new rule
                name, definition = line.split(':', 1)
                current_rule = name.strip().strip('?')

                check_quotes(definition, f"rule '{current_rule}'", line_num)
                definition = re.sub(r"'([^']*)'", r'"\1"', definition)
                referenced_rules.update(extract_references(definition))
                current_definition = [definition.strip()]

            elif line.startswith('|'):
                if not current_rule:
                    raise ValueError(f"Alternative '|' on line {line_num} "
                                     "without a preceding rule definition")

                alt_def = line[1:].strip()
                check_quotes(alt_def, f"alternative for rule '{current_rule}'",
                             line_num)
                alt_def = re.sub(r"'([^']*)'", r'"\1"', alt_def)
                referenced_rules.update(extract_references(alt_def))
                current_definition.append(alt_def)

        except ValueError as e:
            raise ValueError(f"Error on line {line_num}: {str(e)}") from e

    # Add final rule if exists
    if current_rule:
        output_lines.append(
            f"{current_rule} ::= {' | '.join(current_definition)}")

    # Validate all rules are defined
    undefined_rules = referenced_rules - defined_rules - {'root'}
    if undefined_rules:
        raise ValueError("Referenced rules are not defined: "
                         f"{', '.join(sorted(undefined_rules))}")

    return '\n'.join(output_lines)


def choice_as_grammar(choice: list[str]) -> str:

    def escape_ebnf_string(s: str) -> str:
        """Escape special characters in a EBNF string."""
        # Escape double quotes and backslashes
        return re.sub(r'(["\\])', r'\\\1', s)

    escaped_choices = (escape_ebnf_string(c) for c in choice)
    grammar = ('root ::= ' + ' | '.join(f'"{c}"' for c in escaped_choices))
    return grammar
