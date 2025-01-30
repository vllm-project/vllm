from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import xgrammar as xgr

from vllm.config import VllmConfig
from vllm.v1.request import GuidedDecodingKey, Request, RequestStatus

if TYPE_CHECKING:
    from vllm.transformers_utils.tokenizer_group import BaseTokenizerGroup

__all__ = ["Grammar", "GuidedDecodingManager"]


class Grammar:
    # https://xgrammar.mlc.ai/docs/api/python/index.html#xgrammar.GrammarMatcher.find_jump_forward_string for jump-forward decoding

    def __init__(self, matcher: xgr.GrammarMatcher, vocab_size: int,
                 ctx: xgr.CompiledGrammar) -> None:
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.ctx = ctx

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


@dataclass
class GrammarCache:
    value: Optional[Grammar]
    event: threading.Event


class GuidedDecodingManager:

    def flush(self):
        with self._lock:
            self.grammar_cache.clear()

    def cache(self, request: Request):

        def _executor_loop(request: Request):
            key = request.guided_decoding_key
            with self._lock:
                cache_hit = False
                if key in self.grammar_cache:
                    cache_hit, entry = True, self.grammar_cache[key]
                else:
                    entry = GrammarCache(None, threading.Event())
                    self.grammar_cache[key] = entry

            if cache_hit:
                entry.event.wait()
            else:
                entry.value = self.initialize_cache(key)
                entry.event.set()
            return copy.copy(entry.value) if entry.value else None

        return self.executor.submit(_executor_loop, request)

    def get(self, request: Request):
        with self._lock:
            entry = self.grammar_cache.get(request.guided_decoding_key)
            if entry is None or not entry.event.is_set(): return None
            return copy.copy(entry.value) if entry.value else None

    def collect(self, request: Request):
        if not request.use_guided_decoding: return False
        request.grammar = self.get(request)
        if not request.grammar:
            request.grammar = self.cache(request)
            request.status = RequestStatus.WAITING_FOR_FSM
            return True
        return False

    def __init__(self, *, vllm_config: VllmConfig,
                 tokenizer_group: BaseTokenizerGroup):
        self.vllm_config = vllm_config
        self.tokenizer = tokenizer_group.get_lora_tokenizer(None)
        self.grammar_cache: dict[GuidedDecodingKey, GrammarCache] = {}
        self.executor = ThreadPoolExecutor()
        self._lock = threading.Lock()

    def initialize_cache(self, key: GuidedDecodingKey, max_threads: int = 8):
        request_type, grammar_spec = key
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(self.tokenizer)
        compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=max_threads)
        if request_type == "json":
            if type(grammar_spec) is not str:
                ctx = compiler.compile_builtin_json_grammar()
            else:
                ctx = compiler.compile_json_schema(grammar_spec)
        elif request_type == "grammar":
            ctx = compiler.compile_grammar(grammar_spec)
        else:
            raise ValueError("grammar is not of valid supported types.")
        return Grammar(
            matcher=xgr.GrammarMatcher(ctx),
            vocab_size=self.vllm_config.model_config.hf_text_config.vocab_size,
            ctx=ctx)
