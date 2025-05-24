# SPDX-License-Identifier: Apache-2.0
import copy
import os
from typing import Any

import llguidance
import llguidance.hf
import llguidance.torch
import torch
from transformers import PreTrainedTokenizerBase

from vllm.logger import init_logger

logger = init_logger(__name__)


class GuidanceLogitsProcessor:
    """Base Guidance Logits Processor"""

    cached_tokenizers: dict[str, Any] = {}

    def __init__(
        self,
        grammar: str,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """Base Guidance Logits Processor

        Args:
            grammar (str)
                grammar to guide the generation
            tokenizer (PreTrainedTokenizerBase)
                model's tokenizer
        """
        self.grammar = grammar
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer.name_or_path
        self.ll_tokenizer = None
        self.ll_matcher = None
        self.bitmask = None
        self.new_sampling = False
        self.initialized = False

    def clone(self) -> "GuidanceLogitsProcessor":
        cloned = copy.copy(self)
        if self.initialized:
            cloned.ll_matcher = llguidance.LLMatcher(
                self.ll_tokenizer,  # type: ignore[assignment]
                self.grammar,
                log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
            )
            self.bitmask = llguidance.torch.allocate_token_bitmask(
                1, self.ll_tokenizer.vocab_size)  # type: ignore[attr-defined]
        return cloned

    def _initialize(self):
        if self.initialized:
            return

        ll_tokenizer = self.cached_tokenizers.get(self.tokenizer.name_or_path,
                                                  None)
        if ll_tokenizer is None:
            ll_tokenizer = llguidance.hf.from_tokenizer(self.tokenizer, None)
            self.cached_tokenizers[self.tokenizer.name_or_path] = ll_tokenizer

        self.ll_tokenizer = ll_tokenizer
        self.ll_matcher = llguidance.LLMatcher(
            self.ll_tokenizer,
            self.grammar,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )

        # create reusable bitmask
        self.bitmask = llguidance.torch.allocate_token_bitmask(
            1, self.ll_tokenizer.vocab_size)  # type: ignore[attr-defined]

        self.initialized = True

    def __call__(
        self,
        input_ids: list[int],
        scores: torch.Tensor,
    ) -> torch.Tensor:
        # we initialize the guidance model here
        # to avoid pickling ll_tokenizer and ll_interpreter
        self._initialize()

        if self.new_sampling and len(input_ids) > 0:
            self.ll_matcher.consume_token(  # type: ignore[attr-defined]
                input_ids[-1])
            err = self.ll_matcher.get_error()  # type: ignore[attr-defined]
            if err:
                logger.warning("Error in LLMatcher: %s", err)

        llguidance.torch.fill_next_token_bitmask(self.ll_matcher, self.bitmask,
                                                 0)
        llguidance.torch.apply_token_bitmask_inplace(
            scores,
            self.bitmask.to(scores.device))  # type: ignore[attr-defined]

        self.new_sampling = True

        return scores
