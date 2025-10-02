# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025-present the Outlines developers
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import ast
import importlib
import json
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from regex import escape as regex_escape

from vllm.sampling_params import SamplingParams
from vllm.utils import LazyLoader
from vllm.v1.structured_output.backend_types import (StructuredOutputBackend,
                                                     StructuredOutputGrammar,
                                                     StructuredOutputOptions)
from vllm.v1.structured_output.utils import (OutlinesVocabulary,
                                             get_outlines_cache,
                                             get_outlines_vocabulary)

if TYPE_CHECKING:
    import outlines_core as oc
    import outlines_core.json_schema as json_schema
else:
    oc = LazyLoader("oc", globals(), "outlines_core")
    json_schema = LazyLoader("json_schema", globals(),
                             "outlines_core.json_schema")

# Python 3.11+ sre_parse and sre_constants
# are deprecated, so we must import them from re
if sys.version_info >= (3, 11):
    # Hack to get around pre-commit regex module rule
    # because going through re is the only way to get sre_parse
    # and sre_constants in Python 3.11+
    _re = importlib.import_module("re")
    sre_parse = _re._parser
    sre_constants = _re._constants
else:
    import sre_constants
    import sre_parse


@dataclass
class OutlinesBackend(StructuredOutputBackend):

    def __post_init__(self):
        self.vocabulary = get_outlines_vocabulary(self.tokenizer)
        self.cache = get_outlines_cache()

    def _compile_index(self, regex_string: str,
                       vocabulary: OutlinesVocabulary) -> oc.Index:
        cache_key = f"{vocabulary._hash}_{regex_string}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        index = oc.Index(regex_string, vocabulary.inner)
        self.cache[cache_key] = index

        return index

    def compile_grammar(self, request_type: StructuredOutputOptions,
                        grammar_spec: str) -> StructuredOutputGrammar:
        if request_type == StructuredOutputOptions.JSON:
            regex = json_schema.build_regex_from_schema(grammar_spec)
        elif request_type == StructuredOutputOptions.REGEX:
            regex = grammar_spec
        elif request_type == StructuredOutputOptions.CHOICE:
            choices = ast.literal_eval(grammar_spec)
            choices = [regex_escape(c) for c in choices]
            regex = "(" + "|".join(choices) + ")"
        else:
            raise ValueError(
                f"Invalid request type for Outlines backend ({request_type!s})"
            )
        index = self._compile_index(regex, self.vocabulary)
        max_rollback_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config is not None else 0)
        return OutlinesGrammar(vocab_size=self.vocab_size,
                               guide=oc.Guide(
                                   index, max_rollback=max_rollback_tokens))

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        return torch.full(
            (max_num_seqs, (self.vocab_size + 31) // 32),
            -1,
            dtype=torch.int32,
            pin_memory=torch.cuda.is_available(),
        )

    def destroy(self):
        pass


@dataclass
class OutlinesGrammar(StructuredOutputGrammar):

    vocab_size: int
    guide: oc.Guide = field(hash=False)
    num_processed_tokens: int = field(default_factory=lambda: 0,
                                      repr=False,
                                      hash=False,
                                      init=False)

    # outlines_core signals done on DFA accept; vLLM expects done after EOS.
    # We delay the finished flag by one step so EOS can still be emitted.
    _prev_finished: bool = field(default=False,
                                 init=False,
                                 repr=False,
                                 hash=False)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        if self.guide.accepts_tokens(tokens):
            # Advance cannot fail because we checked Guide.accepts_tokens()
            for t in tokens:
                self.guide.advance(t)
                self.num_processed_tokens += 1
            return True
        return False

    def rollback(self, num_tokens: int) -> None:
        self.guide.rollback_state(num_tokens)
        self.num_processed_tokens -= num_tokens

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        accepted: list[int] = []
        for tok in tokens:
            accepted.append(tok)
            if not self.guide.accepts_tokens(accepted):
                accepted.pop()
                break
        return accepted

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        mask = bitmask[idx]
        self.guide.write_mask_into(mask.data_ptr(), mask.numel(),
                                   mask.element_size())

    def is_terminated(self) -> bool:
        curr = self.guide.is_finished()
        prev = self._prev_finished
        self._prev_finished = curr
        return prev

    def reset(self):
        self.num_processed_tokens = 0
        self._prev_finished = False
        self.guide.reset()


def validate_structured_output_request_outlines(params: SamplingParams):
    if params.structured_outputs is None:
        return

    so_params = params.structured_outputs

    if so_params.regex:
        validate_regex_is_buildable(so_params.regex)
    elif so_params.json:
        if isinstance(so_params.json, str):
            try:
                # make sure schema is valid json
                json.loads(so_params.json)
                schema = so_params.json
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON grammar specification.") from e
        else:
            try:
                schema = json.dumps(so_params.json)
            except Exception as e:
                raise ValueError(
                    f"Error serializing structured outputs jsonschema: {e}"
                ) from e
        pattern = json_schema.build_regex_from_schema(schema)
        validate_regex_is_buildable(pattern)
    elif so_params.choice:
        choices = [regex_escape(str(choice)) for choice in so_params.choice]
        regex = "(" + "|".join(choices) + ")"
        validate_regex_is_buildable(regex)
    elif so_params.grammar:
        raise ValueError("Outlines structured outputs backend "
                         "does not support grammar specifications")


def _prefix_needs_context(parsed) -> bool:
    """Return True if there's a look-around/anchor before any consumer."""

    def subpattern_consumes(parsed) -> bool:
        """Return True if subpattern can consume at least one character."""
        tokens = parsed.data if hasattr(parsed, 'data') else parsed
        for ttype, tval in tokens:
            # literal, character class, or dot always consumes
            if ttype in (sre_parse.LITERAL, sre_parse.IN, sre_parse.ANY):
                return True
            # quantified subpattern: check inner pattern
            elif ttype == sre_parse.MAX_REPEAT:
                _, mx, sub = tval
                if mx != 0 and subpattern_consumes(sub):
                    return True
            # alternation: if any branch consumes, the whole does
            elif ttype == sre_parse.BRANCH:
                _, branches = tval
                if any(subpattern_consumes(br) for br in branches):
                    return True
            # grouped subpattern: recurse into its contents
            elif ttype == sre_parse.SUBPATTERN and subpattern_consumes(
                    tval[3]):
                return True
        # No consumers, return False
        return False

    tokens = parsed.data if hasattr(parsed, 'data') else parsed
    for ttype, tval in tokens:
        # Direct anchors or look-around
        if ttype == sre_parse.AT or ttype in (sre_constants.ASSERT,
                                              sre_constants.ASSERT_NOT):
            return True

        # Nested subpattern: check
        if ttype == sre_parse.SUBPATTERN:
            # tval: (group, add_flags, del_flags, subpattern)
            if _prefix_needs_context(tval[3]):
                return True
            if subpattern_consumes(tval[3]):
                return False

        # if any branch has a prefix anchor => True,
        # else if at least one branch consumes => prefix ends => False
        elif ttype == sre_parse.BRANCH:
            saw_consumer = False
            for br in tval[1]:
                if _prefix_needs_context(br):
                    return True
                if subpattern_consumes(br):
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
            if subpattern_consumes(tval[2]):
                return False

    return False


def _check_unsupported(parsed) -> None:
    """Check for regex features unsupported by regex-automata"""
    tokens = parsed.data if hasattr(parsed, 'data') else parsed
    for ttype, tval in tokens:

        # backreference
        if ttype in (sre_parse.GROUPREF, sre_parse.GROUPREF_EXISTS):
            raise ValueError("Backreferences are unsupported.")

        # look-around assertion
        elif ttype in (sre_constants.ASSERT, sre_constants.ASSERT_NOT):
            raise ValueError("Look-Around assertion are unsupported.")

        # unicode word boundaries
        elif ttype == sre_parse.AT:
            if tval in (sre_constants.AT_BOUNDARY,
                        sre_constants.AT_NON_BOUNDARY):
                raise ValueError("Unicode word boundaries are unsupported.")

        elif ttype == sre_parse.BRANCH:
            # tval is (None, branches)
            for branch in tval[1]:
                _check_unsupported(branch)

        # tval is (min, max, subpattern)
        elif ttype == sre_parse.MAX_REPEAT:
            _check_unsupported(tval[2])


def validate_regex_is_buildable(pattern: str) -> None:
    """
    Validates that the input regex is not using unsupported features
    of the `regex-automata` crate (outlines_core regex engine) and has a
    universal start state.
    definition of universal start state used can be found at:
    https://docs.rs/regex-automata/latest/regex_automata/dfa/trait.Automaton.html#method.universal_start_state
    """
    try:
        parsed = sre_parse.parse(pattern)

    except sre_constants.error as e:
        raise ValueError(f"Error parsing regex: {e}") from e

    try:
        _check_unsupported(parsed)
    except ValueError as e:
        raise ValueError(
            f"Regex uses unsupported feature for structured outputs: {e}. "
            "Only basic matching constructs are supported—lookarounds, "
            "backreferences, and unicode boundaries are not.") from e

    if _prefix_needs_context(parsed):
        raise ValueError(
            "Regex does not have a anchored universal start state"
            "This means that the Regex uses anchors (^) or look-arounds "
            "in a way which requires context before any token is matched."
            "structured outputs needs regexes that can match without needing "
            "that context. Try rewriting the pattern without using these "
            f"constructs. Pattern:\n{pattern}")
