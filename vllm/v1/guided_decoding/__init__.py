# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import enum
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Dict, Set, Tuple

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


class Grammar:
    # NOTE: This would be a generic-enough class for
    # supporting different backends, in the future.
    # For now, just xgrammar.
    #
    # https://xgrammar.mlc.ai/docs/api/python/index.html#xgrammar.GrammarMatcher.find_jump_forward_string
    # for jump-forward decoding
    # TODO: support max_rollback_tokens

    def __init__(self, matcher: xgr.GrammarMatcher, vocab_size: int,
                 ctx: xgr.CompiledGrammar) -> None:
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.ctx = ctx
        self.prefilled = False

    def accept_token(self, token: int) -> bool:
        # NOTE: accept_token will determines whether we accept this token
        # and will also update the machine state
        return self.matcher.accept_token(token)

    def allocate_bitmask(self, batch_size: int,
                         vocab_size: int) -> torch.Tensor:
        return xgr.allocate_token_bitmask(batch_size, vocab_size)

    # this should be ran in parallel with model decoding
    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(bitmask, idx)

    @staticmethod
    def apply_bitmask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        xgr.apply_token_bitmask_inplace(logits, vocab_mask)

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
        self.model_config = vllm_config.model_config
        self.vocab_size = vllm_config.model_config.get_vocab_size()

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            tokenizer_group.get_lora_tokenizer(None),
            vocab_size=self.vocab_size)
        self.compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)

        self.grammar_cache: Dict[GuidedDecodingKey, Grammar] = {}

        self.executor = ThreadPoolExecutor()
        self.requests: Set[Request] = set()

        self._lock = threading.Lock()

    def should_cache(self, request: Request):
        if not request.use_guided_decoding:
            return False
        request.grammar = self.get_grammar(request)
        if not request.grammar:
            request.grammar = self.cache(request)
            return True
        return False

    def cache(self, request: Request):
        return self.executor.submit(self._executor_loop, request)

    def _executor_loop(self, request: Request):
        key = request.guided_decoding_key
        self.requests.add(request)
        with self._lock:
            if key in self.grammar_cache:
                return self.grammar_cache[key]

        self.grammar_cache[key] = self.initialize_grammar(key)
        return self.grammar_cache[key]

    def initialize_grammar(self, key: GuidedDecodingKey) -> Grammar:
        request_type, grammar_spec = key

        if request_type == GuidedDecodingOptions.json:
            if not isinstance(grammar_spec, str):
                ctx = self.compiler.compile_builtin_json_grammar()
            else:
                ctx = self.compiler.compile_json_schema(grammar_spec)
        elif request_type == GuidedDecodingOptions.grammar:
            ctx = self.compiler.compile_grammar(grammar_spec)
        else:
            raise ValueError(
                f"`grammar` is not of valid supported types. ({request_type!s})"
            )

        return Grammar(matcher=xgr.GrammarMatcher(ctx),
                       vocab_size=self.vocab_size,
                       ctx=ctx)

    def get_grammar(self, request: Request):
        with self._lock:
            return self.grammar_cache.get(request.guided_decoding_key)
