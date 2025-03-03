# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import enum
import multiprocessing
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

import numpy as np
import torch
import xgrammar as xgr

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


class GuidedDecodingOptions(enum.Enum):
    JSON = enum.auto()
    JSON_OBJECT = enum.auto()
    REGEX = enum.auto()
    GRAMMAR = enum.auto()
    CHOICE = enum.auto()


GuidedDecodingKey = Tuple[GuidedDecodingOptions, str]
MAX_ROLLBACK_TOKENS = 100


def apply_bitmask(
    logits: torch.Tensor,
    vocab_mask: torch.Tensor,
    indices: List[int],
) -> None:
    xgr.apply_token_bitmask_inplace(logits, vocab_mask, indices=indices)


@dataclass(slots=True, unsafe_hash=True)  # type: ignore[call-overload]
class Grammar:
    # NOTE: This would be a generic-enough class for
    # supporting different backends, in the future.
    # For now, just xgrammar.
    #
    # TODO: support max_rollback_tokens
    # https://xgrammar.mlc.ai/docs/api/python/index.html#xgrammar.GrammarMatcher.find_jump_forward_string
    # for jump-forward decoding

    vocab_size: int
    matcher: xgr.GrammarMatcher = field(hash=False)
    ctx: xgr.CompiledGrammar = field(hash=False)
    max_rollback_tokens: int = field(default=MAX_ROLLBACK_TOKENS)
    num_processed_tokens: int = field(default_factory=lambda: 0,
                                      repr=False,
                                      hash=False,
                                      init=False)

    def accept_tokens(self, request_id: str, tokens: List[int]) -> bool:
        # NOTE: accept_token will determines whether we accept this token
        # and will also update the machine state
        for token in tokens:
            self.num_processed_tokens += 1
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.", request_id, token)
                return False
        return True

    # this should be ran in parallel with model decoding
    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> bool:
        return self.matcher.fill_next_token_bitmask(bitmask, idx)

    def rollback(self, num_tokens: int):
        if self.num_processed_tokens > 0:
            self.num_processed_tokens -= num_tokens
            self.matcher.rollback(num_tokens)

    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()

    def __copy__(self):
        return Grammar(matcher=xgr.GrammarMatcher(
            self.ctx, max_rollback_tokens=self.max_rollback_tokens),
                       vocab_size=self.vocab_size,
                       ctx=self.ctx,
                       max_rollback_tokens=self.max_rollback_tokens)


class GuidedDecodingManager:

    def __init__(self, vllm_config: VllmConfig, max_cache_size: int = 500):
        tokenizer_group = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            parallel_config=vllm_config.parallel_config,
            lora_config=vllm_config.lora_config)  # type: ignore[arg-type]
        tokenizer_group.ping()
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        self.vllm_config = vllm_config

        tokenizer = tokenizer_group.get_lora_tokenizer(None)
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=self.vocab_size)
        self.compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)

        self.max_cache_size = max_cache_size
        self.request_key_to_grammar: OrderedDict[GuidedDecodingKey,
                                                 Grammar] = OrderedDict()

        # The default max_workers if not specified is the number of CPUs * 5,
        # which is way too high since these tasks are CPU-bound, not I/O bound.
        # We also know we would never dominate CPU usage with just grammar
        # compilation, so we set it to half the number of CPUs.
        max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.requests: Set[Request] = set()
        self._requests_lock = threading.Lock()
        self._grammar_bitmask = xgr.allocate_token_bitmask(
            self.vllm_config.scheduler_config.max_num_seqs, self.vocab_size)

    def __getitem__(self, key: GuidedDecodingKey) -> Optional[Grammar]:
        if key in self.request_key_to_grammar:
            # Move accessed item to the end (most recently used)
            value = self.request_key_to_grammar.pop(key)
            self.request_key_to_grammar[key] = value
            return value
        return None

    def remove_requests(self, request_ids: List[str]) -> None:
        with self._requests_lock:
            self.requests = {
                req
                for req in self.requests if req.request_id not in request_ids
            }

    def populate_cache(self, request: Request):
        if not request.use_guided_decoding:
            return False
        grammar = self.request_key_to_grammar.get(request.guided_decoding_key)
        if grammar:
            request.grammar = copy.copy(grammar)
            return False
        request.grammar = self.cache(request)
        return True

    def cache(self, request: Request):
        return self.executor.submit(self._executor_loop, request)

    def _executor_loop(self, request: Request) -> Grammar:
        key = request.guided_decoding_key
        with self._requests_lock:
            self.requests.add(request)
        if key in self.request_key_to_grammar:
            grammar = self.request_key_to_grammar[key]
            return copy.copy(grammar)
        grammar = self.initialize_grammar(key)
        # If cache is full, remove the least recently used item
        if len(self.request_key_to_grammar) >= self.max_cache_size:
            self.request_key_to_grammar.popitem(last=False)
        self.request_key_to_grammar[key] = grammar
        return copy.copy(grammar)

    def initialize_grammar(self, key: GuidedDecodingKey) -> Grammar:
        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request.
        #
        # TODO: we still need to handle xgrammar compilation failures
        request_type, grammar_spec = key

        if request_type == GuidedDecodingOptions.JSON:
            # TODO -- allow any_whitespace to be configurable
            # pending merge of https://github.com/vllm-project/vllm/pull/12744
            ctx = self.compiler.compile_json_schema(grammar_spec,
                                                    any_whitespace=False)
        elif request_type == GuidedDecodingOptions.JSON_OBJECT:
            ctx = self.compiler.compile_builtin_json_grammar()
        elif request_type == GuidedDecodingOptions.GRAMMAR:
            ctx = self.compiler.compile_grammar(grammar_spec)
        else:
            logger.error("Validation should have already occurred. "
                         "Please file an issue.")
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})")

        max_rollback_tokens = MAX_ROLLBACK_TOKENS
        if self.vllm_config.speculative_config:
            max_rollback_tokens = self.vllm_config.speculative_config\
                                      .num_lookahead_slots + 1
        return Grammar(matcher=xgr.GrammarMatcher(
            ctx, max_rollback_tokens=max_rollback_tokens),
                       vocab_size=self.vocab_size,
                       ctx=ctx,
                       max_rollback_tokens=max_rollback_tokens)

    def setup_grammars(self):
        with self._requests_lock:
            for req in self.requests:
                if req.grammar is not None:
                    continue

                # Check if grammar is ready in cache
                grammar = self[req.guided_decoding_key]
                if grammar is not None:
                    req.grammar = copy.copy(grammar)
                    continue

    def grammar_bitmask(self, requests: dict[str, Request],
                        guided_decoding_request_ids: dict[str, int],
                        batch_len: int) -> Optional[np.ndarray]:
        # Prepare the guided decoding bitmask for this batch.
        if not guided_decoding_request_ids:
            return None

        # Fill the bitmask using the index of each request equal to its
        # position in the batch. Resize the bitmask down to the size of
        # the batch.
        bitmask_tensor = self._grammar_bitmask
        for req_id, batch_index in guided_decoding_request_ids.items():
            request = requests[req_id]
            assert request.grammar is not None
            if not request.grammar.matcher.is_terminated():
                request.grammar.fill_bitmask(bitmask_tensor, batch_index)
        if batch_len < self._grammar_bitmask.shape[0]:
            bitmask_tensor = self._grammar_bitmask[:batch_len]

        # After finishing with the xgrammar operations, we convert to
        # np.ndarray, because that is much more efficient for serialization
        # and deserialization when sending this to the GPU workers.
        return bitmask_tensor.numpy()
