# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import enum
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import torch
import xgrammar as xgr

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


class GuidedDecodingOptions(enum.Enum):
    json = enum.auto()
    json_object = enum.auto()
    regex = enum.auto()
    grammar = enum.auto()
    choice = enum.auto()


GuidedDecodingKey = Tuple[GuidedDecodingOptions, str]
MAX_ROLLBACK_TOKENS = 100


def apply_bitmask(logits: torch.Tensor, vocab_mask: torch.Tensor,
                  indices: List[int]) -> None:
    xgr.apply_token_bitmask_inplace(logits, vocab_mask, indices=indices)


@dataclass(slots=True, unsafe_hash=True)
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
    max_rollback_tokens: int = field(default=MAX_ROLLBACK_TOKENS, kw_only=True)
    num_processed_tokens: int = field(
        default_factory=lambda: 0,
        repr=False,
        hash=False,
        init=False,
    )
    _matcher_lock: threading.Lock = field(
        default_factory=lambda: threading.Lock(),
        repr=False,
        init=False,
        hash=False,
    )

    def accept_token(self, token: int) -> bool:
        # NOTE: accept_token will determines whether we accept this token
        # and will also update the machine state
        with self._matcher_lock:
            self.num_processed_tokens += 1
            return self.matcher.accept_token(token)

    # this should be ran in parallel with model decoding
    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> bool:
        return self.matcher.fill_next_token_bitmask(bitmask, idx)

    def rollback(self, num_tokens: int):
        with self._matcher_lock:
            if self.num_processed_tokens > 0:
                self.num_processed_tokens -= num_tokens
                self.matcher.rollback(num_tokens)

    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()

    def __copy__(self):
        return Grammar(matcher=xgr.GrammarMatcher(self.ctx),
                       vocab_size=self.vocab_size,
                       ctx=self.ctx,
                       max_rollback_tokens=self.max_rollback_tokens)


class GuidedDecodingManager:

    def __init__(self, vllm_config: VllmConfig):
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

        self.request_key_to_grammar: Dict[GuidedDecodingKey, Grammar] = {}

        self.executor = ThreadPoolExecutor()
        self.requests: Set[Request] = set()
        self._requests_lock = threading.Lock()
        self.grammar_bitmask = xgr.allocate_token_bitmask(
            self.vllm_config.scheduler_config.max_num_seqs, self.vocab_size)

    def __getitem__(self, key: GuidedDecodingKey) -> Optional[Grammar]:
        return self.request_key_to_grammar.get(key)

    def remove_requests(self, request_ids: List[str]) -> None:
        with self._requests_lock:
            self.requests = {
                req
                for req in self.requests if req.request_id not in request_ids
            }

    def should_cache(self, request: Request):
        if not request.use_guided_decoding:
            return False
        request.grammar = self.request_key_to_grammar.get(
            request.guided_decoding_key)
        if not request.grammar:
            request.grammar = self.cache(request)
            return True
        return False

    def cache(self, request: Request):
        return self.executor.submit(self._executor_loop, request)

    def _executor_loop(self, request: Request) -> Grammar:
        key = request.guided_decoding_key
        with self._requests_lock:
            self.requests.add(request)
        if key in self.request_key_to_grammar:
            return self.request_key_to_grammar[key]

        self.request_key_to_grammar[key] = self.initialize_grammar(key)
        return self.request_key_to_grammar[key]

    def initialize_grammar(self, key: GuidedDecodingKey) -> Grammar:
        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request.
        #
        # TODO: we still need to handle xgrammar compilation failures
        request_type, grammar_spec = key

        if request_type == GuidedDecodingOptions.json:
            # TODO -- allow any_whitespace to be configurable
            # pending merge of https://github.com/vllm-project/vllm/pull/12744
            ctx = self.compiler.compile_json_schema(grammar_spec,
                                                    any_whitespace=False)
        elif request_type == GuidedDecodingOptions.json_object:
            ctx = self.compiler.compile_builtin_json_grammar()
        elif request_type == GuidedDecodingOptions.grammar:
            ctx = self.compiler.compile_grammar(grammar_spec)
        else:
            logger.error("Validation should have already occurred. "
                         "Please file an issue.")
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})")

        return Grammar(
            matcher=xgr.GrammarMatcher(ctx),
            vocab_size=self.vocab_size,
            ctx=ctx,
            max_rollback_tokens=self.vllm_config.speculative_config.
            num_lookahead_slots
            if self.vllm_config.speculative_config else MAX_ROLLBACK_TOKENS)

    def setup_grammars(self):
        with self._requests_lock:
            for req in self.requests:
                if req.grammar is not None:
                    continue

                # Check if grammar is ready in cache
                grammar = self[req.guided_decoding_key]
                if grammar is not None:
                    req.grammar = grammar
                    continue
