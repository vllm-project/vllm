# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shadow grammar manager for speculative decoding.

Maintains a per-request GrammarMatcher in the worker process that mirrors
the scheduler's authoritative grammar state. During draft token generation,
the shadow matcher applies grammar bitmasks to draft logits so that all
proposed tokens are guaranteed grammar-valid — eliminating post-hoc
truncation in the scheduler.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.utils.import_utils import LazyLoader
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.request import get_structured_output_key
from vllm.v1.structured_output.utils import (
    choice_as_grammar,
    compile_regex_with_timeout,
    convert_lark_to_ebnf,
    grammar_is_likely_lark,
)

if TYPE_CHECKING:
    import torch
    import xgrammar as xgr
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


@dataclass
class ShadowGrammarState:
    """Per-request shadow grammar state in the worker."""

    matcher: xgr.GrammarMatcher
    vocab_size: int
    num_accepted_tokens: int = 0
    num_draft_tokens: int = 0
    is_terminated: bool = False


class ShadowGrammarManager:
    """Manages shadow grammar matchers for structured-output requests
    in the worker process during speculative decoding.

    The shadow matchers mirror the scheduler's authoritative grammar state
    by being advanced with the same accepted tokens. During draft proposal,
    they provide bitmasks that constrain draft token sampling.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        self.vllm_config = vllm_config
        self._active_count: int = 0
        self.num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config is not None
            else 0
        )
        self._states: dict[str, ShadowGrammarState] = {}
        self._bitmask: torch.Tensor | None = None

        tokenizer = cached_tokenizer_from_config(model_config=vllm_config.model_config)
        vocab_size = vllm_config.model_config.get_vocab_size()

        from vllm.utils.mistral import is_mistral_tokenizer

        if is_mistral_tokenizer(tokenizer):
            stop_token_ids = [tokenizer.eos_token_id]
            self._vocab_size = len(tokenizer.vocab)
            tokenizer_info = xgr.TokenizerInfo(
                encoded_vocab=tokenizer.vocab,
                vocab_type=xgr.VocabType.RAW
                if tokenizer.is_tekken
                else xgr.VocabType.BYTE_FALLBACK,
                vocab_size=self._vocab_size,
                stop_token_ids=stop_token_ids,
                add_prefix_space=True,
            )
        else:
            self._vocab_size = vocab_size
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                tokenizer,
                vocab_size=vocab_size,
            )

        import vllm.envs

        self._compiler = xgr.GrammarCompiler(
            tokenizer_info,
            max_threads=4,
            cache_enabled=True,
            cache_limit_bytes=vllm.envs.VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024,
        )

        self._disable_any_whitespace = (
            vllm_config.structured_outputs_config.disable_any_whitespace
            if vllm_config.structured_outputs_config is not None
            else False
        )

    def has_grammar(self, req_id: str) -> bool:
        return req_id in self._states

    def register_request(self, req_id: str, sampling_params: SamplingParams) -> None:
        """Create a shadow matcher for a new structured-output request."""
        if sampling_params.structured_outputs is None:
            return
        if req_id in self._states:
            return

        so_params = sampling_params.structured_outputs
        try:
            request_type, grammar_spec = get_structured_output_key(so_params)
        except ValueError:
            return

        ctx = self._compile_grammar(request_type, grammar_spec)
        if ctx is None:
            return

        matcher = xgr.GrammarMatcher(
            ctx,
            max_rollback_tokens=self.num_speculative_tokens,
        )
        self._states[req_id] = ShadowGrammarState(
            matcher=matcher,
            vocab_size=self._vocab_size,
        )

    def remove_request(self, req_id: str) -> None:
        """Remove shadow matcher when a request finishes."""
        self._states.pop(req_id, None)

    def advance_with_accepted_tokens(self, req_id: str, tokens: list[int]) -> None:
        """Advance the shadow matcher with tokens accepted by the scheduler."""
        state = self._states.get(req_id)
        if state is None or state.is_terminated:
            return
        for token in tokens:
            if not state.matcher.accept_token(token):
                logger.warning(
                    "Shadow grammar for %s failed to accept token %d. "
                    "Removing shadow grammar.",
                    req_id,
                    token,
                )
                self._states.pop(req_id, None)
                return
            state.num_accepted_tokens += 1
        state.is_terminated = state.matcher.is_terminated()

    def fill_bitmask(self, req_id: str, bitmask: torch.Tensor, idx: int) -> bool:
        """Fill bitmask for the next draft token.

        Returns True if bitmask was filled, False if no grammar for this req.
        """
        state = self._states.get(req_id)
        if state is None or state.is_terminated:
            return False
        state.matcher.fill_next_token_bitmask(bitmask, idx)
        return True

    def accept_draft_token(self, req_id: str, token_id: int) -> bool:
        """Accept a draft token (tentative — will be rolled back after
        draft proposal completes).

        Returns True if accepted, False if rejected.
        """
        state = self._states.get(req_id)
        if state is None or state.is_terminated:
            return True
        if state.matcher.accept_token(token_id):
            state.num_draft_tokens += 1
            state.is_terminated = state.matcher.is_terminated()
            return True
        return False

    def rollback_draft_tokens(self, req_id: str) -> None:
        """Roll back all tentative draft tokens after proposal completes."""
        state = self._states.get(req_id)
        if state is None or state.num_draft_tokens == 0:
            return
        state.matcher.rollback(state.num_draft_tokens)
        state.num_draft_tokens = 0
        state.is_terminated = state.matcher.is_terminated()

    def allocate_bitmask(self, batch_size: int) -> torch.Tensor:
        """Allocate (or reuse) a bitmask tensor for draft sampling."""
        if self._bitmask is None or self._bitmask.shape[0] < batch_size:
            self._bitmask = xgr.allocate_token_bitmask(batch_size, self._vocab_size)
        return self._bitmask[:batch_size]

    def _compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> xgr.CompiledGrammar | None:
        """Compile a grammar spec into a CompiledGrammar."""
        try:
            if request_type == StructuredOutputOptions.JSON:
                return self._compiler.compile_json_schema(
                    grammar_spec,
                    any_whitespace=not self._disable_any_whitespace,
                )
            elif request_type == StructuredOutputOptions.JSON_OBJECT:
                return self._compiler.compile_json_schema(
                    '{"type": "object"}',
                    any_whitespace=not self._disable_any_whitespace,
                )
            elif request_type == StructuredOutputOptions.GRAMMAR:
                if grammar_is_likely_lark(grammar_spec):
                    grammar_spec = convert_lark_to_ebnf(grammar_spec)
                return self._compiler.compile_grammar(grammar_spec)
            elif request_type == StructuredOutputOptions.REGEX:
                return compile_regex_with_timeout(
                    self._compiler.compile_regex, grammar_spec
                )
            elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
                s_tag = json.loads(grammar_spec)
                if "structures" in s_tag:
                    tags = [
                        xgr.StructuralTagItem(
                            begin=s["begin"],
                            schema=json.dumps(s["schema"]),
                            end=s["end"],
                        )
                        for s in s_tag["structures"]
                    ]
                    return self._compiler.compile_structural_tag(
                        tags, s_tag["triggers"]
                    )
                else:
                    return self._compiler.compile_structural_tag(grammar_spec)
            elif request_type == StructuredOutputOptions.CHOICE:
                choice_grammar = choice_as_grammar(json.loads(grammar_spec))
                return self._compiler.compile_grammar(choice_grammar)
            else:
                logger.warning(
                    "Unsupported grammar type for shadow matcher: %s",
                    request_type,
                )
                return None
        except Exception:
            logger.warning(
                "Failed to compile shadow grammar for type %s. "
                "Draft tokens will not be grammar-constrained.",
                request_type,
                exc_info=True,
            )
            return None

    def destroy(self) -> None:
        """Cleanup."""
        self._states.clear()
        self._bitmask = None
