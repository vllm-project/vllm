# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING
from functools import lru_cache
import sre_constants
import sre_parse
import re
import json

import torch
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import LazyLoader
from vllm.sampling_params import SamplingParams
from vllm.v1.structured_output.backend_types import (StructuredOutputBackend,
                                                     StructuredOutputGrammar,
                                                     StructuredOutputOptions)


if TYPE_CHECKING:
    import outlines_core as oc
else:
    oc = LazyLoader("oc", globals(), "outlines_core")

logger = init_logger(__name__)

@lru_cache
def _compile_index(regex_string: str, vocabulary: oc.Vocabulary) -> oc.Index:
    return oc.Index(regex_string, vocabulary)

def _reduced_vocabulary(tokenizer: AnyTokenizer, eos_token_id: int) -> Dict[str | bytes, List[int]]:
    """Create a map from decoded vocabulary tokens to lists of equivalent token ids.
    
    Returns:
        A Dict of token string -> equivalent token ids
    """
    unicode_to_bytes = {v: k for k, v in bytes_to_unicode().items()}

    re_llama_byte_token = re.compile(r"^<0x[0-9A-F]{2}>$")
    re_replacement_seq = re.compile(r"^▁* +\.*$")

    def byte_symbol(byte: int) -> str:
        return f"\x00{byte:02X}" if byte >= 0x80 else chr(byte)
    
    def convert_token_to_string(token: str) -> str:
        from transformers.file_utils import SPIECE_UNDERLINE

        string = tokenizer.convert_tokens_to_string([token])

        # A hack to handle missing spaces to HF's Llama tokenizers
        if (type(token) is str and token.startswith(SPIECE_UNDERLINE)
                or token == "<0x20>"):
            return " " + string

        return string

    vocabulary: dict[str | bytes, list[int]] = {}
    empty_token_ids: list[int] = []
    for token, token_idx in tokenizer.get_vocab().items():
        if token in tokenizer.special_tokens:
            continue

        token_str = convert_token_to_string(token)

        if token_str:
            if isinstance(token, bytes):
                # For BPE tokenizers where tokens are stored as bytes.
                token_str = "".join(byte_symbol(b) for b in token)
            elif "\ufffd" in token_str and not re_replacement_seq.match(token):
                # Handle tokens with invalid UTF-8 sequences.
                if re_llama_byte_token.match(token):
                    # Llama-like tokenizers use <0xXX> for incomplete sequences.
                    token_bytes = [int(token[3:5], 16)]
                else:
                    # GPT2-like tokenizers: map each byte back using unicode_to_bytes.
                    token_bytes = [unicode_to_bytes.get(c) for c in token]
                    if None in token_bytes:
                        raise RuntimeError(
                            f"Cannot convert token `{token}` ({token_idx}) to bytes: {token_str}"
                        )
                token_str = "".join(byte_symbol(b) for b in token_bytes)

            if token_idx != eos_token_id:
                vocabulary.setdefault(token_str, []).append(token_idx)
        else:
            empty_token_ids.append(token_idx)

    return vocabulary


class OutlinesBackend(StructuredOutputBackend):

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.vocab_size = vllm_config.model_config.get_vocab_size()

        tokenizer_group = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            parallel_config=vllm_config.parallel_config,
            lora_config=vllm_config.lora_config)  # type: ignore[arg-type]
        tokenizer_group.ping()
        tokenizer = tokenizer_group.get_lora_tokenizer(None)
        
        reduced_vocab = None
        try:
            vocab = tokenizer.get_vocab()
            eos_token_id = None
            if hasattr(
                        tokenizer,
                        "eos_token_id",
                ) and tokenizer.eos_token_id is not None:
                    eos_token_id = tokenizer.eos_token_id
            
            reduced_vocab = _reduced_vocabulary(
                tokenizer,
                eos_token_id # type: ignore
            )
        except AttributeError as e:
            raise ValueError(
                f"Cannot get the vocabulary of the tokenizer "
                f"{type(tokenizer)}. The tokenizer should have a "
                "get_vocab method.") from e
        
        self.vocabulary = oc.Vocabulary(eos_token_id, reduced_vocab) # type: ignore[arg-type]
    
    def compile_grammar(self, request_type: StructuredOutputOptions,
                        grammar_spec: str) -> StructuredOutputGrammar:
        if request_type == StructuredOutputOptions.JSON:
            regex = oc.json_schema.build_regex_from_schema(grammar_spec)
            index = _compile_index(regex, self.vocabulary)
        elif request_type == StructuredOutputOptions.REGEX:
            index = _compile_index(grammar_spec, self.vocabulary)
        # To do: choice. But how is choice expressed here? is it a precompiled regex
        # since grammar_spec is a string in StructuredOutputManager, or is it a list?
        else:
            raise ValueError(
                f"Unsupported request type for Outlines backend  ({request_type!s})")

        return OutlinesGrammar(
            vocab_size=self.vocab_size,
            guide=oc.Guide(index)
        )

    def allocate_token_bitmask(self, max_num_seqs: int):
        return torch.full(
            (max_num_seqs, (self.vocab_size + 31) // 32),
            -1,
            dtype=torch.int32,
            pin_memory=torch.cuda.is_available(),
        )

@dataclass
class OutlinesGrammar(StructuredOutputGrammar):

    vocab_size: int
    guide: oc.Guide = field(hash=False)
    num_processed_tokens: int = field(default_factory=lambda: 0,
                                      repr=False,
                                      hash=False,
                                      init=False)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        for token in tokens:
            try:
                self.guide.advance(token)
                self.num_processed_tokens += 1
            except ValueError:
                return False
        return True
    
    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        mask = bitmask[idx]
        self.guide.write_mask_into(
            mask.data_ptr(),
            mask.numel(),
            mask.element_size()
        )
    
    def is_terminated(self) -> bool:
        return self.guide.is_finished()
    
    def reset(self):
        self.guide.reset()


def validate_structured_output_request_outlines(params: SamplingParams):
    if params.guided_decoding is None:
        return
    
    gd_params = params.guided_decoding

    if gd_params.regex:
        validate_regex_buildable(gd_params.regex)
    elif gd_params.json:
        if isinstance(gd_params.json, str):
            try:
                schema = json.loads(gd_params.json)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON grammar specification.") from e
        else:
            schema = gd_params.json
        pattern = oc.json_schema.build_regex_from_schema(
            schema)
        validate_regex_buildable(pattern)
    elif gd_params.choice:
        choices = [
            re.escape(str(choice)) for choice in gd_params.choice
        ]
        regex = "(" + "|".join(choices) + ")"
        validate_regex_buildable(regex)
    elif gd_params.grammar:
        raise ValueError("Outlines guided decoding backend does not support grammar specifications")


def _prefix_needs_context(parsed) -> bool:
    """Return True if there's a look-around/anchor before any consumer."""

    def _subpattern_consumes(parsed) -> bool:
        """Return True if subpattern can consume at least one character."""
        tokens = parsed.data if hasattr(parsed, 'data') else parsed
        for ttype, tval in tokens:
            if ttype in (sre_parse.LITERAL, sre_parse.IN, sre_parse.ANY):
                return True
            elif ttype == sre_parse.MAX_REPEAT:
                mn, mx, sub = tval
                if mx != 0 and _subpattern_consumes(sub):
                    return True
            elif ttype == sre_parse.BRANCH:
                _, branches = tval
                if any(_subpattern_consumes(br) for br in branches):
                    return True
            elif ttype == sre_parse.SUBPATTERN:
                if _subpattern_consumes(tval[3]):
                    return True
        return False
    
    tokens = parsed.data if hasattr(parsed, 'data') else parsed
    for ttype, tval in tokens:
        # Direct anchors or look-around
        if ttype == sre_parse.AT or ttype in (sre_constants.ASSERT, sre_constants.ASSERT_NOT):
            return True

        # Nested subpattern: check
        if ttype == sre_parse.SUBPATTERN:
            # tval: (group, add_flags, del_flags, subpattern)
            if _prefix_needs_context(tval[3]):
                return True
            if _subpattern_consumes(tval[3]):
                return False

        # if any branch has a prefix anchor => True,
        # else if at least one branch consumes => prefix ends => False
        elif ttype == sre_parse.BRANCH:
            saw_consumer = False
            for br in tval[1]:
                if _prefix_needs_context(br):
                    return True
                if _subpattern_consumes(br):
                    saw_consumer = True
            if saw_consumer:
                return False

        # Immediate consumer tokens
        elif ttype in (sre_parse.LITERAL, sre_parse.IN, sre_parse.ANY):
            return False

        # if subpattern has anchor => True, if it can consume => stop
        elif ttype == sre_parse.MAX_REPEAT:
            if _prefix_needs_context(tval[2]):
                return True
            if _subpattern_consumes(tval[2]):
                return False

    return False

def _check_unsupported(parsed):
    """Check for regex features unsupported by regex-automata"""
    tokens = parsed.data if hasattr(parsed, 'data') else parsed
    for ttype, tval in tokens:

        # backreference
        if ttype in (sre_parse.GROUPREF, sre_parse.GROUPREF_EXISTS):
            raise ValueError("Backereferences are unsupported.")

        # look-around assertion
        elif ttype in (sre_constants.ASSERT, sre_constants.ASSERT_NOT):
            raise ValueError("Look-Around assertion are unsupported.")

        # unicode word boundaries
        elif ttype == sre_parse.AT:
            if tval in (sre_constants.AT_BOUNDARY, sre_constants.AT_NON_BOUNDARY):
                raise ValueError("Unicode word boundaries are unsupported.")

        elif ttype == sre_parse.BRANCH:
            # tval is (None, branches)
            for branch in tval[1]:
                if _check_unsupported(branch):
                    return True

        elif ttype == sre_parse.MAX_REPEAT:
            # tval is (min, max, subpattern)
            if _check_unsupported(tval[2]):
                return True
    return False


def validate_regex_buildable(pattern: str) -> None:
    """
    Validates that the input regex is not using unsupported features
    of the `regex-automata` crate (outlines_core regex engine), and has a
    universal start state.
    definition of universal start state used can be found at:
    https://docs.rs/regex-automata/latest/regex_automata/dfa/trait.Automaton.html#method.universal_start_state
    """
    try:
        parsed = sre_parse.parse(pattern)
        _check_unsupported(parsed)
        if _prefix_needs_context(parsed):
            raise ValueError("anchored universal start state doesn't exist for this regex")
    except Exception as e:
        raise ValueError(f"Error parsing guided regex: {e}")
