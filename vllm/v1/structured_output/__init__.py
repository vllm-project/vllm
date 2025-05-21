# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.utils import LazyLoader
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_types import (StructuredOutputBackend,
                                                     StructuredOutputGrammar)
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.reasoning import ReasoningParser
    from vllm.v1.request import Request
else:
    torch = LazyLoader("torch", globals(), "torch")

logger = init_logger(__name__)


class StructuredOutputManager:
    """Engine-level manager for structured output requests."""

    def __init__(self, vllm_config: VllmConfig):
        self.backend: Optional[StructuredOutputBackend] = None
        self.reasoner: Optional[ReasoningParser] = None
        self.vllm_config = vllm_config

        self._grammar_bitmask: Optional[torch.Tensor] = None
        self._full_mask = torch.tensor(-1, dtype=torch.int32)

        # The default max_workers if not specified is the number of CPUs * 5,
        # which is way too high since these tasks are CPU-bound, not I/O bound.
        # We also know we would never dominate CPU usage with just grammar
        # compilation, so we set it to half the number of CPUs.
        max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tokenizer = init_tokenizer_from_configs(
            model_config=self.vllm_config.model_config,
            scheduler_config=self.vllm_config.scheduler_config,
            lora_config=self.vllm_config.lora_config,
        ).get_lora_tokenizer(None)
        reasoning_backend = vllm_config.decoding_config.reasoning_backend
        if reasoning_backend:
            reasoner_cls = ReasoningParserManager.get_reasoning_parser(
                reasoning_backend)
            self.reasoner = reasoner_cls(tokenizer=self.tokenizer)

    def grammar_init(self, request: Request) -> None:
        if request.structured_output_request is None:
            return

        if TYPE_CHECKING:
            assert request.sampling_params.guided_decoding is not None

        # Initialize the backend the first time it is needed.
        #
        # NOTE: We only support a single backend. We do NOT support different
        # backends on a per-request basis in V1 (for now, anyway...).
        if self.backend is None:
            backend = request.sampling_params.guided_decoding.backend
            vocab_size = self.vllm_config.model_config.get_vocab_size()
            if backend == "xgrammar":
                self.backend = XgrammarBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "guidance":
                self.backend = GuidanceBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            else:
                raise ValueError(
                    f"Unsupported structured output backend: {backend}")

        grammar = self.executor.submit(self._async_create_grammar, request)
        request.structured_output_request.grammar = grammar  # type: ignore[assignment]

    def _async_create_grammar(
        self,
        request: Request,
    ) -> StructuredOutputGrammar:
        key = request.structured_output_request.structured_output_key  # type: ignore[union-attr]

        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request.
        #
        # TODO: we still need to handle xgrammar compilation failures,
        # though it should be unlikely as we test that up front as well.
        request_type, grammar_spec = key

        assert self.backend is not None
        return self.backend.compile_grammar(request_type, grammar_spec)

    def grammar_bitmask(
        self,
        requests: dict[str, Request],
        structured_output_request_ids: dict[str, int],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> Optional[npt.NDArray[np.int32]]:
        # Prepare the structured output bitmask for this batch.
        if not structured_output_request_ids:
            return None

        max_num_spec_tokens = 0
        if self.vllm_config.speculative_config is not None:
            max_num_spec_tokens = \
                self.vllm_config.speculative_config.num_speculative_tokens

        if self._grammar_bitmask is None:
            assert self.backend is not None
            max_batch_size = self.vllm_config.scheduler_config.max_num_seqs

            # Allocate a bitmask for each token needing to be checked:
            # one for each speculative position, and one more for the
            # bonus token / non-speculative token.
            self._grammar_bitmask = \
                self.backend.allocate_token_bitmask(
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
            request = requests[req_id].structured_output_request
            if TYPE_CHECKING:
                assert request is not None
                assert request.grammar is not None

            apply_bitmask = (
                request.reasoning_ended if self.reasoner is not None else True
            )  # noqa: E501

            state_advancements = 0
            req_tokens = scheduled_spec_decode_tokens.get(req_id, []) + [None]
            for i, token in enumerate(req_tokens):
                if apply_bitmask and not request.grammar.is_terminated():
                    request.grammar.fill_bitmask(bitmask_tensor,
                                                 cumulative_index)
                    if token is not None:
                        # In order to generate the correct bitmask for each
                        # position in the speculative sequence, we advance
                        # the FSM state for each speculative token and rollback
                        # to restore the previous state when we are finished.
                        assert request.grammar.accept_tokens(req_id, [token])
                        state_advancements += 1
                cumulative_index += 1
            if state_advancements > 0:
                request.grammar.rollback(state_advancements)

        if cumulative_index < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:cumulative_index]

        # After finishing with the xgrammar operations, we convert to
        # np.ndarray, because that is much more efficient for serialization
        # and deserialization when sending this to the GPU workers.
        return bitmask_tensor.numpy()

    def should_advance(self, request: Request) -> bool:
        if not request.use_structured_output:
            return False

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        if TYPE_CHECKING:
            assert request.structured_output_request is not None
            assert request.structured_output_request.grammar is not None
        # by default, we should always advance
        # for cases that doesn't uses thinking mode.
        if self.reasoner is not None:
            structured_req = request.structured_output_request

            if structured_req.reasoning_ended:
                return True

            # Check if reasoning ends in *this* step
            if self.reasoner.is_reasoning_end(request.all_token_ids):
                # Reasoning just ended, so we shouldn't advanced til
                # next pass
                structured_req.reasoning_ended = True

            return False
        else:
            return True

    def clear_backend(self) -> None:
        if self.backend is not None:
            self.backend.destroy()
