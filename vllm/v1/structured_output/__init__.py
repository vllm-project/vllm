# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import multiprocessing
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.utils import LazyLoader
from vllm.v1.structured_output.grammar import (Grammar, StructuredOutputKey,
                                               StructuredOutputOptions)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import xgrammar as xgr

    from vllm.v1.request import Request
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


class StructuredOutputManager:

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
        self.request_key_to_grammar: OrderedDict[StructuredOutputKey,
                                                 Grammar] = OrderedDict()

        # The default max_workers if not specified is the number of CPUs * 5,
        # which is way too high since these tasks are CPU-bound, not I/O bound.
        # We also know we would never dominate CPU usage with just grammar
        # compilation, so we set it to half the number of CPUs.
        max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._grammar_bitmask = xgr.allocate_token_bitmask(
            self.vllm_config.scheduler_config.max_num_seqs, self.vocab_size)

    def __getitem__(self, key: StructuredOutputKey) -> Optional[Grammar]:
        # We need to pop and re-insert the grammar here for LRU cache
        # of request_key_to_grammar
        if key in self.request_key_to_grammar:
            # Move accessed item to the end (most recently used)
            value = self.request_key_to_grammar.pop(key)
            if value is not None:
                self.request_key_to_grammar[key] = value
            return value
        return None

    def populate_cache(self, request: Request) -> None:
        if request.structured_output_request is None:
            return

        grammar = self.request_key_to_grammar.get(
            request.structured_output_request.structured_output_key)
        if grammar:
            request.structured_output_request.grammar = copy.copy(grammar)
            return
        request.structured_output_request.grammar = self.cache(request)

    def cache(self, request: Request):
        return self.executor.submit(self._executor_loop, request)

    def _executor_loop(self, request: Request) -> Grammar:
        # NOTE: The structured_output_request should never be
        # None in this case, but mypy can't infer this
        # correctly, so we need to ignore the error here.
        key = request.structured_output_request.structured_output_key  # type: ignore[union-attr]
        grammar = self.request_key_to_grammar.get(key)
        if grammar is not None:
            return copy.copy(grammar)
        grammar = self.initialize_grammar(key)
        # If cache is full, remove the least recently used item
        if len(self.request_key_to_grammar) >= self.max_cache_size:
            self.request_key_to_grammar.popitem(last=False)
        self.request_key_to_grammar[key] = grammar
        return copy.copy(grammar)

    def initialize_grammar(self, key: StructuredOutputKey) -> Grammar:
        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request.
        #
        # TODO: we still need to handle xgrammar compilation failures
        request_type, grammar_spec = key

        if request_type == StructuredOutputOptions.JSON:
            # TODO -- allow any_whitespace to be configurable
            # pending merge of https://github.com/vllm-project/vllm/pull/12744
            ctx = self.compiler.compile_json_schema(grammar_spec,
                                                    any_whitespace=False)
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            ctx = self.compiler.compile_builtin_json_grammar()
        elif request_type == StructuredOutputOptions.GRAMMAR:
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

    def grammar_bitmask(
        self,
        requests: dict[str, Request],
        structured_output_request_ids: dict[str, int],
        batch_len: int,
    ) -> Optional[npt.NDArray[np.int32]]:
        # Prepare the structured output bitmask for this batch.
        if not structured_output_request_ids:
            return None

        # Fill the bitmask using the index of each request equal to its
        # position in the batch. Resize the bitmask down to the size of
        # the batch.
        bitmask_tensor = self._grammar_bitmask
        for req_id, batch_index in structured_output_request_ids.items():
            request = requests[req_id].structured_output_request
            assert request is not None and request.grammar is not None
            if not request.grammar.matcher.is_terminated():
                request.grammar.fill_bitmask(bitmask_tensor, batch_index)
        if batch_len < self._grammar_bitmask.shape[0]:
            bitmask_tensor = self._grammar_bitmask[:batch_len]

        # After finishing with the xgrammar operations, we convert to
        # np.ndarray, because that is much more efficient for serialization
        # and deserialization when sending this to the GPU workers.
        return bitmask_tensor.numpy()
