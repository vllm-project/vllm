# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import enum
import functools
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import torch
import xgrammar as xgr

from vllm.config import VllmConfig
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs

if TYPE_CHECKING:
    from vllm.v1.request import Request


class GuidedDecodingOptions(enum.Enum):
    json = enum.auto()
    regex = enum.auto()
    grammar = enum.auto()
    choice = enum.auto()


GuidedDecodingKey = Tuple[GuidedDecodingOptions, str]


def reset_bitmask(bitmask: torch.Tensor):
    # this calls bitmask.fill_(tensor([1, 1, ...], dtype=int32))
    xgr.reset_token_bitmask(bitmask)


def apply_bitmask(logits: torch.Tensor, vocab_mask: torch.Tensor,
                  indices: List[int]) -> None:
    xgr.apply_token_bitmask_inplace(logits, vocab_mask, indices=indices)


class Grammar:
    # NOTE: This would be a generic-enough class for
    # supporting different backends, in the future.
    # For now, just xgrammar.
    #
    # TODO: support max_rollback_tokens
    # https://xgrammar.mlc.ai/docs/api/python/index.html#xgrammar.GrammarMatcher.find_jump_forward_string
    # for jump-forward decoding

    def __init__(
        self,
        matcher: xgr.GrammarMatcher,
        vocab_size: int,
        ctx: xgr.CompiledGrammar,
    ) -> None:
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.ctx = ctx

    def accept_token(self, token: int) -> bool:
        # NOTE: accept_token will determines whether we accept this token
        # and will also update the machine state
        return self.matcher.accept_token(token)

    # this should be ran in parallel with model decoding
    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> bool:
        return self.matcher.fill_next_token_bitmask(bitmask, idx)

    def reset(self):
        self.matcher.reset()

    def copy(self):
        return Grammar(matcher=xgr.GrammarMatcher(self.ctx),
                       vocab_size=self.vocab_size,
                       ctx=self.ctx)

    def __copy__(self):
        return self.copy()


class GuidedDecodingManager:

    def __init__(self, vllm_config: VllmConfig):
        tokenizer_group = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            parallel_config=vllm_config.parallel_config,
            lora_config=vllm_config.lora_config)
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
        self._grammar_bitmask: Optional[Union[torch.Tensor,
                                              Future[torch.Tensor]]] = None

    def __getitem__(self, key: GuidedDecodingKey) -> Optional[Grammar]:
        return self.request_key_to_grammar.get(key)

    def allocate_bitmask(self) -> None:
        # NOTE: We will only want to allocate this once
        if self._grammar_bitmask is None:
            self._grammar_bitmask = self.executor.submit(
                xgr.allocate_token_bitmask,
                self.vllm_config.scheduler_config.max_num_seqs,
                self.vocab_size,
            )

    def _ensure_bitmask_ready(self) -> bool:
        if isinstance(self._grammar_bitmask, Future):
            try:
                self._grammar_bitmask = self._grammar_bitmask.result(
                    timeout=0.05)
            except TimeoutError:
                return False
        return True

    @functools.cached_property
    def grammar_bitmask(self) -> Optional[torch.Tensor]:
        self._ensure_bitmask_ready()
        return self._grammar_bitmask if not isinstance(self._grammar_bitmask,
                                                       Future) else None

    @property
    def is_bitmask_ready(self) -> bool:
        self._ensure_bitmask_ready()
        if isinstance(self._grammar_bitmask, Future):
            return not self._grammar_bitmask.running(
            ) and self._grammar_bitmask.done()
        return self._grammar_bitmask is not None

    def reset_bitmask(self):
        reset_bitmask(self.grammar_bitmask)

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
        request_type, grammar_spec = key

        if request_type == GuidedDecodingOptions.json:
            if not isinstance(grammar_spec, str):
                ctx = self.compiler.compile_builtin_json_grammar()
            else:
                # TODO -- allow any_whitespace to be configurable
                # pending merge of https://github.com/vllm-project/vllm/pull/12744
                ctx = self.compiler.compile_json_schema(grammar_spec,
                                                        any_whitespace=False)
        elif request_type == GuidedDecodingOptions.grammar:
            ctx = self.compiler.compile_grammar(grammar_spec)
        # elif request_type == GuidedDecodingOptions.regex:
        #     ctx = self.compiler.compile_regex(grammar_spec)
        else:
            raise ValueError(
                f"`grammar` is not of valid supported types. ({request_type!s})"
            )

        return Grammar(
            matcher=xgr.GrammarMatcher(ctx),
            vocab_size=self.vocab_size,
            ctx=ctx,
        )

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
        self.allocate_bitmask()
