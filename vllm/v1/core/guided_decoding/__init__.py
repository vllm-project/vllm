from __future__ import annotations

import copy, enum
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import xgrammar as xgr

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.v1.request import GuidedDecodingKey, Request, RequestStatus

from .grammar import Grammar

if TYPE_CHECKING:
    from typing_extensions import Self

    from vllm.transformers_utils.tokenizer_group import BaseTokenizerGroup

    from .grammar import XGrammar

logger = init_logger(__name__)

__all__ = ["Grammar", "GuidedDecodingManager"]


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

    def __init__(self, *, backend: str, tokenizer_group: BaseTokenizerGroup,
                 model_config: ModelConfig):
        self._backend = backend
        self.model_config = model_config
        self.tokenizer = tokenizer_group.get_lora_tokenizer(None)
        self.grammar_cache: dict[GuidedDecodingKey, GrammarCache] = {}
        self.executor = ThreadPoolExecutor()
        self._lock = threading.Lock()
        cls._registry[backend] = cls

    def initialize_cache(self, key: GuidedDecodingKey) -> Self:
        request_type, grammar_spec = key
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, stop_token_ids=stop_token_ids, vocab_size=vocab_size)
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
        return Grammar(matcher=xgr.GrammarMatcher(ctx),
                       vocab_size=self.model_config.hf_text_config.vocab_size,
                       ctx=ctx)
