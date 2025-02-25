# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import enum
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import torch
import xgrammar as xgr

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.v1.guided_decoding.utils import (
    has_xgrammar_unsupported_json_features)

if TYPE_CHECKING:
    from vllm.v1.request import Request

import json

logger = init_logger(__name__)


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
        self.grammar_bitmask: torch.Tensor = xgr.allocate_token_bitmask(
            self.vllm_config.scheduler_config.max_num_seqs, self.vocab_size)

    def __getitem__(self, key: GuidedDecodingKey) -> Optional[Grammar]:
        return self.request_key_to_grammar.get(key)

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
        self._validate_grammer_is_supported(request.guided_decoding_key)
        return self.executor.submit(self._executor_loop, request)

    def _executor_loop(self, request: Request) -> Grammar:
        key = request.guided_decoding_key
        with self._requests_lock:
            self.requests.add(request)
        if key in self.request_key_to_grammar:
            return self.request_key_to_grammar[key]

        self.request_key_to_grammar[key] = self.initialize_grammar(key)
        return self.request_key_to_grammar[key]

    def _validate_grammer_is_supported(self, key: GuidedDecodingKey):
        request_type, grammar_spec = key
        if request_type == GuidedDecodingOptions.json:
            try:
                schema = json.loads(grammar_spec)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON grammar specification.") from e

            if has_xgrammar_unsupported_json_features(schema):
                raise ValueError(
                    "The provided JSON schema contains features not "
                    "supported by xgrammar.")
            return
        elif request_type == GuidedDecodingOptions.grammar:
            return
        raise ValueError(
            f"grammar is not of valid supported types. ({request_type!s})")

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
        else:
            logger.error("Validation should have already occurred. "
                         "Please file an issue.")
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})")

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
