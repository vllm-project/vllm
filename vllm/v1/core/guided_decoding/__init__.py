from __future__ import annotations

import copy
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, get_args

from transformers import PreTrainedTokenizer
import xgrammar as xgr

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.v1.request import GuidedDecodingKey, Request, RequestStatus

from .grammar import Grammar

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from typing_extensions import LiteralString

    from vllm.transformers_utils.tokenizer_group import BaseTokenizerGroup

    from .grammar import XGrammar

logger = init_logger(__name__)

__all__ = ["Grammar", "GuidedDecodingManager"]


@dataclass
class GrammarCache:
    value: Grammar | None
    event: threading.Event


T = TypeVar("T", bound=str)


class GuidedDecodingManager(ABC, Generic[T]):

    @abstractmethod
    def initialize_cache(self, key: GuidedDecodingKey) -> Grammar:
        ...

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

    @classmethod
    def from_backend(cls,
                     backend: LiteralString = "xgrammar",
                     /,
                     *,
                     tokenizer_group: BaseTokenizerGroup,
                     model_config: ModelConfig) -> GuidedDecodingManager[T]:
        manager_cls = cls._registry.get(backend)
        if manager_cls is None:
            raise ValueError(
                f"Backend '{backend}' not found in registry. Available backends: {list(cls._registry)}"
            )
        return manager_cls(tokenizer_group=tokenizer_group,
                           model_config=model_config)

    _registry: dict[str, type[GuidedDecodingManager[T]]] = {}
    _backend: T

    def __init__(self, *, tokenizer_group: BaseTokenizerGroup,
                 model_config: ModelConfig):
        self.model_config = model_config
        self.tokenizer = tokenizer_group.get_lora_tokenizer(None)
        self.grammar_cache: dict[GuidedDecodingKey, GrammarCache] = {}
        self.executor = ThreadPoolExecutor()
        self._lock = threading.Lock()

    def __init_subclass__(cls, **kwargs: Any):
        if not hasattr(cls, '__orig_bases__'):
            raise TypeError(
                f"{cls.__qualname__} must be subclass of GuidedDecodingManager"
            )

        backend = None
        for base in cls.__orig_bases__:
            if (origin := get_args(base)) and issubclass(
                    base.__origin__, GuidedDecodingManager):
                backend = get_args(origin[0])[0]
                break

        if backend is None:
            raise TypeError(
                f"Class {cls.__qualname__} must specify backend as a Literal type"
            )

        if backend in cls._registry:
            name = cls._registry[backend].__qualname__
            raise ValueError(
                f"Backend '{backend}' is already registered to {name}")

        # Set the backend value from the Literal type
        cls._backend = backend
        cls._registry[backend] = cls


class XGrammarManager(GuidedDecodingManager[Literal["xgrammar"]]):
    # cache GrammarCompiler instances based on given tokenizer
    _compiler_cache: dict[str, xgr.GrammarCompiler] = {}
    _compiler: xgr.GrammarCompiler | None = None

    def initialize_cache(self, key: GuidedDecodingKey) -> XGrammar:
        request_type, grammar_spec = key
        compiler = XGrammarManager.get_compiler(self.tokenizer)
        if request_type == "json":
            if type(grammar_spec) is not str:
                ctx = compiler.compile_builtin_json_grammar()
            else:
                ctx = compiler.compile_json_schema(grammar_spec)
        elif request_type == "grammar":
            ctx = compiler.compile_grammar(grammar_spec)
        else:
            raise ValueError("grammar is not of valid supported types.")
        return Grammar.from_backend(
            self._backend,
            matcher=xgr.GrammarMatcher(ctx),
            vocab_size=self.model_config.hf_text_config.vocab_size,
            ctx=ctx)

    def flush(self):
        super().flush()
        if self._compiler: self._compiler.clear_cache()
        for compiler in self._compiler_cache.values():
            compiler.clear_cache()
        self._compiler_cache.clear()

    @classmethod
    def get_compiler(
            cls,
            tokenizer: PreTrainedTokenizer,
            *,
            max_threads: int = 8,
            # passthrough to TokenizerInfo
            vocab_size: int | None = None,
            stop_token_ids: list[int] | int | None = None
    ) -> xgr.GrammarCompiler:
        cache_key = str(hash(tokenizer))
        if cache_key not in cls._compiler_cache:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                tokenizer,
                stop_token_ids=stop_token_ids,
                vocab_size=vocab_size)
            cls._compiler_cache[cache_key] = xgr.GrammarCompiler(
                tokenizer_info, max_threads=max_threads)
        return cls._compiler_cache[cache_key]
