# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend, StructuredOutputBatchMetaData,
    StructuredOutputGrammar, StructuredOutputOptions)

if TYPE_CHECKING:

    from vllm.reasoning import ReasoningParser
    from vllm.transformers_utils.tokenizer import AnyTokenizer
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class BitmaskSOBatchMetaData(StructuredOutputBatchMetaData):
    """
    This class is used to store the bitmask for structured output requests.
    It is used to pass the bitmask to the GPU workers.
    """

    grammar_bitmask: torch.Tensor


class BitmaskStructuredOutputBackend(StructuredOutputBackend):

    def __init__(self, vllm_config: VllmConfig, tokenizer: AnyTokenizer,
                 vocab_size: int, reasoner: ReasoningParser):
        super().__init__(vllm_config, tokenizer, vocab_size, reasoner)
        self._grammar_bitmask: Optional[torch.Tensor] = None
        self._full_mask = torch.tensor(-1, dtype=torch.int32)

    def grammar_bitmask(
        self,
        requests: dict[str, Request],
        structured_output_request_ids: dict[str, int],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> Optional[npt.NDArray[np.int32]]:
        """
        Method used by XGrammar and Guidance to process and filter all logits
        """

        max_num_spec_tokens = 0
        if self.vllm_config.speculative_config is not None:
            max_num_spec_tokens = \
                self.vllm_config.speculative_config.num_speculative_tokens

        if self._grammar_bitmask is None:
            max_batch_size = self.vllm_config.scheduler_config.max_num_seqs
            # Allocate a bitmask for each token needing to be checked:
            # one for each speculative position, and one more for the
            # bonus token / non-speculative token.
            self._grammar_bitmask = \
                self.allocate_token_bitmask(
                    max_batch_size * (1 + max_num_spec_tokens))

        bitmask_tensor = self._grammar_bitmask
        # Generate a batched bitmask for all structured output requests.
        # When speculative decoding is enabled, we need to include multiple
        # masks for each request, one for each possible bonus token position.
        # These are stored inline in the tensor and unpacked by the gpu runner.
        cumulative_index = 0
        ordered_seq = sorted(structured_output_request_ids.items(),
                             key=lambda x: x[1])

        # Note that for thinking support, we will need to
        # reset the relevant part of the bitmask for consequent
        # request here.
        bitmask_tensor[:(len(ordered_seq) * (1 + max_num_spec_tokens))].fill_(
            self._full_mask)

        # NOTE: This outer loop can likely be parallelized to improve
        # performance of bitmask generation for large batches.
        for req_id, _ in ordered_seq:
            request = requests[req_id]
            structured_output_request = request.structured_output_request
            if TYPE_CHECKING:
                assert structured_output_request is not None
                assert structured_output_request.grammar is not None
                assert isinstance(structured_output_request.grammar,
                                  BitmaskGrammar)

            apply_bitmask: bool = True
            if self.reasoner is not None:
                if structured_output_request.reasoning_ended is None:
                    structured_output_request.reasoning_ended = \
                        self.reasoner.is_reasoning_end(request.prompt_token_ids)
                apply_bitmask = structured_output_request.reasoning_ended

            state_advancements = 0
            req_tokens = scheduled_spec_decode_tokens.get(req_id, []) + [None]
            for i, token in enumerate(req_tokens):
                if apply_bitmask and not \
                    structured_output_request.grammar.is_terminated():
                    structured_output_request.grammar.fill_bitmask(
                        bitmask_tensor, cumulative_index)
                    if token is not None:
                        # In order to generate the correct bitmask for each
                        # position in the speculative sequence, we advance
                        # the FSM state for each speculative token and rollback
                        # to restore the previous state when we are finished.
                        assert structured_output_request.grammar.accept_tokens(
                            req_id, [token])
                        state_advancements += 1
                cumulative_index += 1
            if state_advancements > 0:
                structured_output_request.grammar.rollback(state_advancements)

        if cumulative_index < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:cumulative_index]

        # After finishing with the xgrammar operations, we convert to
        # np.ndarray, because that is much more efficient for serialization
        # and deserialization when sending this to the GPU workers.
        return bitmask_tensor.numpy()

    def init_batch(
        self, requests: dict[str, Request],
        structured_output_request_ids: dict[str, int],
        scheduled_spec_decode_tokens: dict[str, list[int]]
    ) -> StructuredOutputBatchMetaData:
        bitmask = self.grammar_bitmask(requests, structured_output_request_ids,
                                       scheduled_spec_decode_tokens)
        return BitmaskSOBatchMetaData(structured_output_request_ids, bitmask)

    @abstractmethod
    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        """
        Allocates a token bitmask for the specified maximum number of sequences.

        Args:
            max_num_seqs (int): The maximum number of sequences for which
              to allocate the bitmask.
        """

    @abstractmethod
    def compile_grammar(self, request_type: StructuredOutputOptions,
                        grammar_spec: str) -> StructuredOutputGrammar:
        """
        Compiles a grammar specification into a structured output grammar.

        Args:
            request_type (StructuredOutputOptions): The type of structured
              output request.
            grammar_spec (str): The grammar specification to compile.

        Returns:
            StructuredOutputGrammar: The compiled structured output grammar.
        """

    @abstractmethod
    def destroy(self):
        pass


class BitmaskGrammar(StructuredOutputGrammar):

    @abstractmethod
    def is_terminated(self) -> bool:
        """
        Checks whether the structured output process has terminated.

        Returns:
            bool: True if the process is terminated, False otherwise.
        """

    @abstractmethod
    def reset(self):
        """
        Resets the state of the structured output grammar.
        """

    @abstractmethod
    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        pass
