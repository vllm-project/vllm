# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from cachetools import LRUCache
import sre_constants
import sre_parse
import re
import json

import torch

from vllm.config import VllmConfig
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.model_executor.guided_decoding.outlines_logits_processors import (get_vocabulary,
                                                                            get_disk_cache)
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

CACHE = None

if envs.VLLM_V1_USE_OUTLINES_CACHE:
    logger.warning("Enabling outlines cache. This is an unbounded on-disk "
                   "cache. It may consume a lot of disk space and should "
                   "not be used with untrusted clients.")
    CACHE = get_disk_cache()
else:
    CACHE = LRUCache(maxsize=128)


def _compile_index(regex_string: str, vocabulary: oc.Vocabulary) -> oc.Index:
    cache_key = f"{vocabulary._hash}_{regex_string}"
    if CACHE is not None and cache_key in CACHE:
        return CACHE[cache_key]

    index = oc.Index(regex_string, vocabulary)
    if CACHE is not None:
        CACHE[cache_key] = index
    
    return index

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
        
        self.vocabulary = get_vocabulary(tokenizer)
    
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
                self.guide.advance(token, return_tokens=False)
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
        validate_regex_is_buildable(gd_params.regex)
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
        validate_regex_is_buildable(pattern)
    elif gd_params.choice:
        choices = [
            re.escape(str(choice)) for choice in gd_params.choice
        ]
        regex = "(" + "|".join(choices) + ")"
        validate_regex_is_buildable(regex)
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


def validate_regex_is_buildable(pattern: str) -> None:
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
