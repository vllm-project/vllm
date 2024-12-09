# noqa: UP007
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from transformers import PreTrainedTokenizerFast

try:
    import xgrammar as xgr
    from xgrammar.base import _core as xgr_core
except ImportError:
    pass

from vllm.model_executor.guided_decoding.xgrammar_utils import (
    convert_lark_to_gbnf, grammar_is_likely_lark)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from vllm.config import ModelConfig
    from vllm.sampling_params import GuidedDecodingParams


# TODO: passing batch size to max threads here
def get_local_xgrammar_guided_decoding_logits_processor(
        guided_params: GuidedDecodingParams,
        tokenizer: PreTrainedTokenizer,
        model_config: ModelConfig,
        max_threads: int = 8):
    config = GrammarConfig.from_guided_params(guided_params=guided_params,
                                              model_config=model_config,
                                              tokenizer=tokenizer,
                                              max_threads=max_threads)
    return XGrammarLogitsProcessor(config)


class TokenizerData(NamedTuple):
    """Immutable container for cached tokenizer data."""
    encoded_vocab: list[str]
    stop_token_ids: list[int] | None
    backend_str: str


class TokenizerDataCache:
    """Cache manager for tokenizer data to avoid repeated processing."""
    _cache: dict[int, TokenizerData] = {}

    @classmethod
    def get_tokenizer_data(cls,
                           tokenizer: PreTrainedTokenizer) -> TokenizerData:
        tokenizer_hash = hash(tokenizer)

        if tokenizer_hash not in cls._cache:
            # Vendored from xgrammar logic since we cannot pickle the tokenizer
            # https://github.com/mlc-ai/xgrammar/blob/d77c0a0173ef14779c918e3be7966ba852f7910f/python/xgrammar/tokenizer_info.py#L98 # noqa: E501
            try:
                encoded_vocab = [
                    token for token, _ in sorted(tokenizer.get_vocab().items(),
                                                 key=lambda x: x[1])
                ]
            except AttributeError as e:
                raise ValueError(
                    f"Cannot get the vocabulary of the tokenizer "
                    f"{type(tokenizer)}. The tokenizer should have a "
                    "get_vocab method.") from e

            stop_token_ids = None
            backend_str = xgr.VocabType.RAW
            if isinstance(tokenizer, PreTrainedTokenizerFast):
                backend_str = tokenizer.backend_tokenizer.to_str()
                if stop_token_ids is None and hasattr(
                        tokenizer,
                        "eos_token_id") and tokenizer.eos_token_id is not None:
                    stop_token_ids = [tokenizer.eos_token_id]

            cls._cache[tokenizer_hash] = TokenizerData(
                encoded_vocab=encoded_vocab,
                stop_token_ids=stop_token_ids,
                backend_str=backend_str)

        return cls._cache[tokenizer_hash]


class GrammarCompilerCache:
    """
    Cache for GrammarCompiler instances based on tokenizer.

    This cache reduces the overhead of creating new compiler instances when
    using the same tokenizer configuration.
    """
    _cache: dict[str, xgr.GrammarCompiler] = {}

    @classmethod
    def get_compiler(cls, config: GrammarConfig) -> xgr.GrammarCompiler:
        cache_key = str(config.tokenizer_hash)

        if cache_key not in cls._cache:
            assert config.encoded_vocab is not None
            tokenizer_info = xgr.TokenizerInfo._create_from_handle(
                xgr_core.TokenizerInfo.from_huggingface(
                    config.encoded_vocab, config.backend_str,
                    config.vocab_size, config.stop_token_ids))
            cls._cache[cache_key] = xgr.GrammarCompiler(
                tokenizer_info, max_threads=config.max_threads)

        return cls._cache[cache_key]


@dataclass
class GrammarConfig:
    """Serializable configuration for grammar compilation"""
    tokenizer_hash: int
    vocab_size: int
    json_str: str | None = None
    grammar_str: str | None = None
    json_object: bool | None = None
    max_threads: int = 8
    # Only populated if tokenizer_hash not in cache
    encoded_vocab: list[str] | None = None
    stop_token_ids: list[int] | None = None
    backend_str: str | None = None

    @classmethod
    def from_guided_params(cls,
                           guided_params: GuidedDecodingParams,
                           model_config: ModelConfig,
                           tokenizer: PreTrainedTokenizer,
                           max_threads: int = 8) -> GrammarConfig:

        tokenizer_hash = hash(tokenizer)
        # Only get tokenizer data if not already cached
        if tokenizer_hash in TokenizerDataCache._cache:
            encoded_vocab = None
            stop_token_ids = None
            backend_str = None
        else:
            tokenizer_data = TokenizerDataCache.get_tokenizer_data(tokenizer)
            encoded_vocab = tokenizer_data.encoded_vocab
            stop_token_ids = tokenizer_data.stop_token_ids
            backend_str = tokenizer_data.backend_str

        if guided_params.json:
            if not isinstance(guided_params.json, str):
                json_str = json.dumps(guided_params.json)
            else:
                json_str = guided_params.json
            return cls(json_str=json_str,
                       vocab_size=model_config.hf_config.vocab_size,
                       encoded_vocab=encoded_vocab,
                       stop_token_ids=stop_token_ids,
                       backend_str=backend_str,
                       tokenizer_hash=tokenizer_hash,
                       max_threads=max_threads)
        elif guided_params.grammar:
            # XGrammar only supports GBNF grammars, so we must convert Lark
            if grammar_is_likely_lark(guided_params.grammar):
                try:
                    grammar_str = convert_lark_to_gbnf(guided_params.grammar)
                except ValueError as e:
                    raise ValueError(
                        "Failed to convert the grammar from Lark to GBNF. "
                        "Please either use GBNF grammar directly or specify"
                        " --guided-decoding-backend=outlines.\n"
                        f"Conversion error: {str(e)}") from e
            else:
                grammar_str = guided_params.grammar
            return cls(grammar_str=grammar_str,
                       vocab_size=model_config.hf_config.vocab_size,
                       encoded_vocab=encoded_vocab,
                       stop_token_ids=stop_token_ids,
                       backend_str=backend_str,
                       tokenizer_hash=tokenizer_hash,
                       max_threads=max_threads)
        elif guided_params.json_object:
            return cls(json_object=True,
                       vocab_size=model_config.hf_config.vocab_size,
                       encoded_vocab=encoded_vocab,
                       stop_token_ids=stop_token_ids,
                       backend_str=backend_str,
                       tokenizer_hash=tokenizer_hash,
                       max_threads=max_threads)
        else:
            raise ValueError(
                "Currently only support JSON and EBNF grammar mode for xgrammar"
            )


@dataclass
class XGrammarLogitsProcessor:
    """Wrapper class to support pickle protocol"""
    config: GrammarConfig

    ctx: xgr.CompiledGrammar | None = None
    token_bitmask: torch.Tensor = None  # type: ignore[assignment]
    matchers: list[xgr.GrammarMatcher] = field(default_factory=list)
    batch_size: int = field(default=1)
    prefilled: bool = field(default=False)

    def __getstate__(self) -> dict[str, Any]:
        return {'config': self.config}

    def __setstate__(self, state: dict[str, Any]):
        self.config = state['config']

        self.ctx = None
        self.matchers = []
        self.batch_size = 1
        self.token_bitmask = None  # type: ignore[assignment]
        self.prefilled = False

    def _ensure_ctx(self):
        """Lazily initialize the processor in the worker process"""
        if self.ctx is None:
            compiler = GrammarCompilerCache.get_compiler(self.config)
            if self.config.json_str is not None:
                self.ctx = compiler.compile_json_schema(self.config.json_str)
            elif self.config.grammar_str is not None:
                self.ctx = compiler.compile_grammar(self.config.grammar_str)
            elif self.config.json_object:
                self.ctx = compiler.compile_builtin_json_grammar()
            else:
                raise ValueError(
                    "Invalid configuration for xgrammar logits processor")

    def __call__(self, input_ids: list[int],
                 scores: torch.Tensor) -> torch.Tensor:
        if self.ctx is None:
            self._ensure_ctx()

        if len(self.matchers) == 0:
            self.matchers = [
                xgr.GrammarMatcher(self.ctx) for _ in range(self.batch_size)
            ]
            self.token_bitmask = xgr.allocate_token_bitmask(
                self.batch_size, self.config.vocab_size)

        if not self.prefilled:
            # Have not sampled a token yet
            self.prefilled = True
        else:
            for i, matcher in enumerate(self.matchers):
                if not matcher.is_terminated():
                    sampled_token = input_ids[-1]
                    assert self.matchers[i].accept_token(sampled_token)

        for i, matcher in enumerate(self.matchers):
            if not matcher.is_terminated():
                # @ubospica: ideally, fill_next_token_bitmask should be
                # parallelized with model decoding
                # See https://github.com/vllm-project/vllm/pull/10785/files#r1864278303
                matcher.fill_next_token_bitmask(self.token_bitmask, i)

        # token_bitmask is a CPU tensor for use with accept_token and
        # fill_next_token_bitmask so we move it to the device of scores
        device_type = scores.device.type
        if device_type != "cuda":
            scores = scores.to("cpu")
        xgr.apply_token_bitmask_inplace(scores,
                                        self.token_bitmask.to(scores.device))
        if device_type != "cuda":
            scores = scores.to(device_type)

        return scores
